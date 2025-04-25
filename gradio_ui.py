import gradio as gr
import os
import traceback # For detailed error reporting in UI

# Import the analyzer class from main.py
try:
    from main import SimpleAudioAnalyzer
except ImportError:
    print("Error: Could not import SimpleAudioAnalyzer from main.py.")
    print("Make sure main.py is in the same directory.")
    # Define a dummy class to prevent NameError if import fails
    class SimpleAudioAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Failed to import SimpleAudioAnalyzer from main.py")
        def analyze_file(self, *args, **kwargs):
            raise ImportError("Failed to import SimpleAudioAnalyzer from main.py")
        def generate_with_musicgen(self, *args, **kwargs):
             raise ImportError("Failed to import SimpleAudioAnalyzer from main.py")

# --- State Variables ---
# These will hold data between steps
analyzer_instance_state = gr.State(None) # To store the initialized analyzer
analysis_results_state = gr.State(None) # To store the full analysis results dict

# --- Function for Step 1: Analysis Only ---
def analyze_audio_only(audio_file_path, use_gpu):
    """
    Initializes the analyzer and runs analysis WITHOUT generation.
    Updates UI with prompt and insights. Stores results in state.
    """
    if not audio_file_path:
        return None, None, "Error: No audio file provided.", "Please upload an audio file.", None, None # Update state outputs

    status_updates = ["Initializing analyzer..."]
    yield None, "", "", "\n".join(status_updates), None, None # Yield initial status

    analyzer_instance = None # Define analyzer_instance here
    results = None # Define results here

    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
             status_updates.append("Warning: OPENAI_API_KEY not set. Using fallback prompt generation.")
             yield None, "", "", "\n".join(status_updates), None, None

        analyzer_instance = SimpleAudioAnalyzer(use_gpu=use_gpu)
        status_updates.append("Analyzer initialized. Starting analysis...")
        yield None, "", "", "\n".join(status_updates), analyzer_instance, None # Update state

        # Run analysis WITHOUT generation
        results = analyzer_instance.analyze_file(audio_file_path, generate_audio=False)
        status_updates.append("Analysis complete.")

        if "error" in results:
            error_msg = f"Analysis failed: {results['error']}"
            status_updates.append(error_msg)
            # Clear outputs on error, update status, clear state
            return None, "", error_msg, "\n".join(status_updates), None, None

        # Format insights
        insights = results.get("insights", [])
        formatted_insights = "\n".join(f"• {line}" for line in insights) if insights else "No insights generated."

        # Get the prompt (use OpenAI description if available, otherwise fallback)
        prompt_for_ui = results.get("openai_description", "Error retrieving prompt.")
        if not analyzer_instance.openai_client or "error" in prompt_for_ui.lower() or "skipped" in prompt_for_ui.lower():
             # Construct the fallback prompt similarly to how analyze_file does it
             gen_info = results.get("generated_audio", {}) # analyze_file stores status here even if skipped
             prompt_for_ui = gen_info.get("prompt_used", "Fallback prompt could not be determined.")
             status_updates.append("Using fallback prompt.")


        status_updates.append("Analysis successful. Review prompt and click 'Generate Music'.")

        # Yield final results for UI and update state variables
        yield None, prompt_for_ui, formatted_insights, "\n".join(status_updates), analyzer_instance, results

    except ImportError as e:
         tb_str = traceback.format_exc(); error_msg = f"Import Error: {e}\nEnsure main.py is present.\n{tb_str}"
         status_updates.append(error_msg)
         return None, error_msg, "", "\n".join(status_updates), None, None # Clear state
    except Exception as e:
        tb_str = traceback.format_exc(); error_msg = f"An unexpected error occurred during analysis: {e}"
        status_updates.append(f"Error: {e}")
        print(f"Traceback:\n{tb_str}")
        # Return error to prompt output, clear insights, update status, clear state
        return None, f"{error_msg}\n\nCheck console for detailed traceback.", "", "\n".join(status_updates), None, None


# --- Function for Step 2: Generation Only ---
def generate_audio_from_state(prompt_from_ui, current_analyzer, current_results):
    """
    Uses the stored analyzer instance and prompt to generate music.
    """
    status_updates = ["Starting music generation..."]
    yield None, "\n".join(status_updates) # Update status, clear audio output initially

    if current_analyzer is None or current_results is None:
        status_updates.append("Error: Analysis must be run successfully before generating music.")
        yield None, "\n".join(status_updates)
        return

    if not prompt_from_ui:
         status_updates.append("Error: Prompt is empty. Cannot generate music.")
         yield None, "\n".join(status_updates)
         return

    try:
        # Get original duration to match generation length
        duration = int(current_results.get("basic_info", {}).get("duration_s", 10))

        # Use the stored analyzer instance to generate
        output_audio_path = current_analyzer.generate_with_musicgen(
            prompt=prompt_from_ui,
            duration=duration
        )

        if output_audio_path and os.path.exists(output_audio_path):
             status_updates.append(f"Generation complete. Audio saved to: {os.path.basename(output_audio_path)}")
             yield output_audio_path, "\n".join(status_updates)
        else:
            status_updates.append("Generation failed. No output file produced (check console logs).")
            yield None, "\n".join(status_updates)

    except Exception as e:
        tb_str = traceback.format_exc(); error_msg = f"An unexpected error occurred during generation: {e}"
        status_updates.append(f"Error: {e}")
        print(f"Traceback:\n{tb_str}")
        yield None, "\n".join(status_updates)


# Define the Gradio UI structure
def generate_ui():
    """Creates and launches the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Caption That Song!
            *Created by Oliver Camp and Charlie Tran (PAT 498 Final Project)*

            **Step 1:** Upload an audio file and click "Analyze Audio".
            **Step 2:** Review the generated prompt (edit if desired) and click "Generate Music".
            *(Requires `OPENAI_API_KEY` environment variable for best prompt generation)*
            """
        )
        # Use State components to store data between steps
        analyzer_state = gr.State(None)
        results_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="Upload Audio File", type="filepath")
                use_gpu_checkbox = gr.Checkbox(label="Use GPU (if available)", value=True)
                analyze_btn = gr.Button("1. Analyze Audio", variant="secondary") # Changed label/variant
                generate_btn = gr.Button("2. Generate Music", variant="primary") # Changed label
                status_output = gr.Textbox(label="Status / Logs", lines=8, interactive=False, placeholder="Processing status...")

            with gr.Column(scale=2):
                # Prompt output is now interactive so user can edit it (optional)
                prompt_output = gr.Textbox(label="Generated Prompt (Editable)", lines=3, interactive=True)
                insights_output = gr.Textbox(label="Analysis Insights", lines=10, interactive=False)
                audio_output = gr.Audio(label="Generated Music (.wav)", type="filepath", interactive=False)


        # Wire Step 1: Analyze Button
        analyze_btn.click(
            fn=analyze_audio_only,
            inputs=[audio_input, use_gpu_checkbox],
            # Outputs: Clear audio, update prompt, insights, status, AND update state variables
            outputs=[audio_output, prompt_output, insights_output, status_output, analyzer_state, results_state]
        )

        # Wire Step 2: Generate Button
        generate_btn.click(
            fn=generate_audio_from_state,
            # Inputs: Get the potentially edited prompt from UI, and the stored analyzer/results from state
            inputs=[prompt_output, analyzer_state, results_state],
            # Outputs: Update the audio player and the status box
            outputs=[audio_output, status_output]
        )

        gr.Markdown("""
        ### Notes
        - Analysis (Step 1) can take time. Generation (Step 2) also takes time.
        - Generation uses the prompt currently shown in the 'Generated Prompt' box.
        - Ensure required libraries are installed and check console for errors.
        """)

    print("Launching Gradio interface...")
    demo.launch(share=False, debug=True)

# Allow running directly
if __name__ == "__main__":
    generate_ui()