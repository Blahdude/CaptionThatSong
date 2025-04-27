# Caption That Song!

## Project Overview
**Caption That Song!** is a final project for PAT 498, created by Oliver Camp and Charlie Tran. This system analyzes audio files to extract human-interpretable features (e.g., genre, mood, tempo) and generates a natural language description using OpenAI’s API. The description can be edited by the user and used to generate a similar music loop with the MusicGen model from Audiocraft. The project provides an interpretable interface for music generation, making it easier for users to understand and control the AI-generated output.

The system is built with two main components:
- **`main.py`**: Handles audio analysis, feature extraction, and music generation.
- **`gradio_ui.py`**: Provides a user-friendly Gradio interface for interacting with the system.

## Motivation
This system is designed for **music producers**, **AI researchers**, and **educators** who want to:
- Analyze audio files and extract human-interpretable descriptions for better understanding of music features.
- Generate similar music loops for creative purposes, such as producing new tracks or experimenting with AI-generated music.
- Create pseudo training data for music GenAI models by reconstructing songs with controlled parameters.
- Use the system as an educational tool to teach students about music features (e.g., tempo, mood, genre) and AI-based music generation.

By providing an interpretable and editable interface, the system improves transparency, controllability, and interpretability for reference-based music GenAI applications, allowing users to bridge the gap between audio analysis and AI-driven music creation.

## Features
- **Audio Analysis**: Extracts features like genre, mood, tempo, key, and timbre using libraries such as Librosa, CLAP, and Transformers.
- **Human-Interpretable Insights**: Generates readable insights (e.g., "Tempo: 120 BPM (moderate)") and a prose description using OpenAI’s API.
- **Editable Prompt**: Users can modify the generated description before music generation.
- **Music Generation**: Uses MusicGen to create a new audio loop based on the description.
- **Gradio UI**: A user-friendly interface for uploading audio, analyzing it, editing prompts, and generating music.

## Setup Instructions
### Prerequisites
- Python 3.8 or higher
- An OpenAI API key (for generating prose descriptions)
- **Note**: CUDA is not set up, so GPU usage is not available. The system will run on CPU only.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Blahdude/CaptionThatSong.git
   cd CaptionThatSong
   ```

2. **Install Dependencies**:
   Create a virtual environment and install the required packages listed in `requirements.txt`:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   - Set your OpenAI API key as an environment variable:
     ```bash
     export OPENAI_API_KEY="your-openai-api-key"  # On Windows: set OPENAI_API_KEY=your-openai-api-key
     ```
   - Without the API key, the system will use a fallback prompt for music generation, which may be less accurate.

### Dependencies
The `requirements.txt` file includes all necessary packages, such as:
- **Gradio**: For the user interface.
- **Librosa**: For audio feature extraction.
- **NumPy**: For numerical computations.
- **PyTorch & Torchaudio**: For audio processing and MusicGen.
- **Transformers**: For audio and genre classification.
- **MSCLAP**: For zero-shot classification.
- **OpenAI**: For generating prose descriptions.
- **Audiocraft**: For MusicGen model.

## Usage
### Running the Gradio UI
1. Launch the Gradio interface:
   ```bash
   python main.py audio.wav --mode gradio
   ```
2. Open the provided URL in your browser to access the UI.
3. **Step 1**: Upload an audio file (preferably in WAV format) and click "Analyze Audio".
4. **Step 2**: Review the generated prompt and insights. Edit the prompt if desired.
5. **Step 3**: Click "Generate Music" to create a new audio loop based on the prompt.

### Running in Terminal Mode
1. Analyze an audio file and optionally generate music:
   ```bash
   python main.py audio.wav --mode terminal --generate
   ```
   - Omit `--generate` to skip music generation and only perform analysis.

## Example
1. Upload an audio file (e.g., a 10-second WAV clip).
2. The system analyzes it and outputs:
   - **Insights**: "Tempo: 120 BPM (moderate). Dominant mood: happy (score=0.85). Predicted genre: pop (score=0.90)."
   - **Prompt**: "A happy pop track with a moderate tempo of 120 BPM, featuring guitar and vocals."
3. Edit the prompt if desired (e.g., change "happy" to "dreamy").
4. Generate a new audio loop, which will be saved as a WAV file (e.g., `generated_dreamy_pop_20250427_123456.wav`).

## Notes
- Audio files shorter than 12.5 seconds may skip tempo analysis due to duration constraints.
- Generation can take time, especially since CUDA is not set up (running on CPU only).
- Ensure the OpenAI API key is set for the best prompt generation experience.
- Check the console or UI status logs for detailed error messages if something goes wrong.

## Project Structure
- **`main.py`**: Core logic for audio analysis, feature extraction, and music generation.
- **`gradio_ui.py`**: Gradio interface for user interaction.
- **`requirements.txt`**: List of required Python packages.
- **Generated Files**: Output audio files are saved in the working directory with names like `generated_<prompt>_<timestamp>.wav`.

## Limitations
- Requires an OpenAI API key for optimal prompt generation.
- CUDA is not set up, so the system runs on CPU, which may be slower.
- Some features (e.g., tempo estimation) are skipped for very short audio files.
- The quality of generated music depends on the accuracy of the analysis and the MusicGen model.

## Future Improvements
- Add more detailed human-interpretable insights (e.g., separate melody, rhythm, and mood descriptions).
- Allow users to control specific parameters (e.g., adjust tempo or mood before generation).
- Conduct systematic evaluation of the system’s performance (e.g., accuracy of analysis, quality of generated music).
- Improve transparency with better logging and visualization in the UI.

## Acknowledgments
- Thanks to our teacher, Herman, for providing valuable feedback.
- Built with open-source libraries: Gradio, Librosa, Transformers, MSCLAP, Audiocraft, and more.

## License
This project is for educational purposes as part of PAT 498. Please contact the authors for usage permissions outside this context.