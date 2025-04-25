import os
import json
import warnings
from datetime import datetime
import argparse # Added for command-line arguments
import traceback # Added for potential tracebacks

import librosa
import numpy as np
import torch
# Removed scipy.stats import as prior is removed
from msclap import CLAP   # core CLAP class for embeddings/similarity
from openai import OpenAI
from audiocraft.models import MusicGen  # Import MusicGen from audiocraft
import torchaudio  # For saving the generated audio

warnings.filterwarnings('ignore')


class SimpleAudioAnalyzer:
    """Analyzer with audio-classification, genre-classification, CLAP zero‑shot, mood, perceptual features, and MusicGen generation"""

    def __init__(self, use_gpu=True, openai_api_key=None):
        self.device_type = 'cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu'
        self.device_num = 0 if self.device_type == 'cuda' else -1
        print(f"Using device: {self.device_type.upper()}")
        self._load_audio_classifier()
        self._load_genre_classifier()
        self._load_clap_model()
        self._initialize_openai(openai_api_key)
        self._load_musicgen()

    # --- Helper methods for loading models ---
    def _load_audio_classifier(self):
        try:
            from transformers import pipeline
            print("Loading standard audio-classification pipeline...")
            self.audio_classifier = pipeline(
                "audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=self.device_num, return_all_scores=True, function_to_apply="softmax"
            )
            print("Audio classification pipeline ready")
        except Exception as e: print(f"Error loading HF audio pipeline: {e}"); self.audio_classifier = None

    def _load_genre_classifier(self):
        try:
            from transformers import pipeline
            print("Loading music-genre pipeline...")
            self.genre_classifier = pipeline(
                "audio-classification", model="m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres",
                device=self.device_num
            )
            print("Genre pipeline ready")
        except Exception as e: print(f"Could not load genre pipeline: {e}"); self.genre_classifier = None

    def _load_clap_model(self):
        try:
            print("Loading CLAP zero-shot model...")
            self.clap_model = CLAP(version="2023", use_cuda=(self.device_type == 'cuda'))
            print("CLAP model loaded successfully")
        except Exception as e: print(f"Error loading CLAP model: {e}"); self.clap_model = None

    def _initialize_openai(self, openai_api_key):
        if openai_api_key is None: openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key: print("Warning: OpenAI API key not found. Description generation will be skipped."); self.openai_client = None
        else: self.openai_client = OpenAI(api_key=openai_api_key)

    def _load_musicgen(self):
        try:
            print("Loading MusicGen model...")
            self.musicgen = MusicGen.get_pretrained('facebook/musicgen-small')
            print("MusicGen model loaded successfully")
        except Exception as e: print(f"Error loading MusicGen model: {e}"); self.musicgen = None
    # --- End Helper methods ---

    def describe_with_openai(self, results):
        """Generates a natural language description using OpenAI based on analysis results."""
        if not self.openai_client: return "OpenAI description skipped (API key missing)."

        # Filter results - using filtered classifications from analyze_file
        filtered_results = {
            "insights": results.get("insights", []),
            "hf_classifications": results.get("hf_classifications", []),
            "genre": results.get("genre", []),
            "mood_classification": results.get("mood_classification", []),
            "key": results.get("key", {}),
            "rhythm": results.get("rhythm", {}), # Tempo or skipped status
            "clap_zero_shot": results.get("clap_zero_shot", []),
            "basic_info": results.get("basic_info", {}),
            "perceptual_features": {
                "f0_mean_hz": results.get("perceptual_features", {}).get("f0_mean_hz"),
                "f0_median_hz": results.get("perceptual_features", {}).get("f0_median_hz"),
            }
        }
        # Clean up potential error/status fields and empty lists
        keys_to_process = ["key", "perceptual_features", "mood_classification", "rhythm", "hf_classifications", "genre", "clap_zero_shot"]
        keys_to_remove = []
        for key in keys_to_process:
             val = filtered_results.get(key)
             if isinstance(val, dict):
                 if 'error' in val: del val['error']
                 # Keep status only if it's the only key (e.g., rhythm skipped)
                 if 'status' in val and len(val.keys()) > 1: del val['status']
                 if not val: keys_to_remove.append(key) # Remove if dict becomes empty
             elif isinstance(val, list):
                 filtered_results[key] = [item for item in val if not (isinstance(item, dict) and ('error' in item or 'status' in item))]
                 if not filtered_results[key]: keys_to_remove.append(key) # Remove if list becomes empty
        for key in keys_to_remove:
            if key in filtered_results: del filtered_results[key]

        analysis_str = json.dumps(filtered_results, indent=2)

        # Simplified system prompt (no mention of reliability)
        system = (
            "You are an expert audio analyst. "
            "Given the below JSON of features and insights from an audio clip, "
            "write a concise description of the sound in natural language. "
            "Focus on the most confident data provided for genre, instrumentation, tempo (if available), key, and overall mood. "
            "Avoid technical audio terms like spectral rolloff, MFCC, or tonnetz. Describe the sound in terms a music generation model can understand (e.g., 'bright tone,' 'warm melody,' 'fast rhythm'). "
            "The description should be suitable for generating a similar audio loop."
        )
        user = f"Audio analysis:\n{analysis_str}\n\nDescription:"

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=150, temperature=0.7, top_p=0.9
            )
            description = response.choices[0].message.content.strip()
            return description.split("Description:")[-1].strip()
        except Exception as e: print(f"Error calling OpenAI API: {e}"); return f"Error generating description: {e}"


    def generate_with_musicgen(self, prompt, duration=10, output_file=None):
        """Generate a new audio clip using MusicGen based on the given prompt."""
        if not self.musicgen: print("MusicGen model not loaded. Skipping generation."); return None
        print(f"\n--- Generating audio with MusicGen using prompt: '{prompt}' ---")
        self.musicgen.set_generation_params(duration=duration, temperature=0.9)
        try:
            wav = self.musicgen.generate([prompt], progress=True)
            audio_tensor = wav[0].cpu()
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
                output_file = f"generated_{safe_prompt}_{timestamp}.wav"
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
            torchaudio.save(output_file, audio_tensor, self.musicgen.sample_rate)
            print(f"Generated audio saved to {output_file}")
            return output_file
        except Exception as e: print(f"Error generating audio with MusicGen: {e}"); traceback.print_exc(); return None


    def analyze_file(self, audio_file, generate_audio=True):
        """Analyzes the audio file and optionally generates a new one based on description."""
        if not os.path.exists(audio_file): raise FileNotFoundError(f"Audio file not found: {audio_file}")
        print(f"\n==== ANALYSIS START: {os.path.basename(audio_file)} ====\n")
        results = {}

        # --- Step 1: Load waveform & basic info ---
        try:
            waveform, sr = librosa.load(audio_file, sr=None)
            target_sr = 32000
            if sr != target_sr: waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr); sr = target_sr
            duration = librosa.get_duration(y=waveform, sr=sr)
            results["basic_info"] = {"filename": os.path.basename(audio_file), "duration_s": float(duration), "sample_rate": sr, "num_samples": len(waveform)}
            print(f"Loaded audio: {duration:.2f} seconds, {sr} Hz")
        except Exception as e: print(f"Error loading audio file: {e}"); return {"error": f"Failed to load audio: {e}"}

        # --- Step 2: Extended low‑level features ---
        onset_env = None
        try:
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sr) # Use full waveform
            rms_env = librosa.feature.rms(y=waveform)[0]; rms = float(np.mean(rms_env)); peak = float(np.max(np.abs(waveform))); dynamic_range = float(peak / rms) if rms > 1e-6 else 0.0
            cent = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]; bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]; zcr = librosa.feature.zero_crossing_rate(y=waveform)[0]
            mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13); mfcc_mean = np.mean(mfcc, axis=1).tolist(); mfcc_var = np.var(mfcc, axis=1).tolist()
            chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr); chroma_mean = np.mean(chroma, axis=1).tolist()
            spec_con = librosa.feature.spectral_contrast(y=waveform, sr=sr); spec_con_mean = np.mean(spec_con, axis=1).tolist()
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr); onset_density = len(onsets) / duration if duration > 0 else 0.0
            loud_db = librosa.amplitude_to_db(rms_env, ref=np.max); loud_summary = {"min_db": float(np.min(loud_db)), "mean_db": float(np.mean(loud_db)), "max_db": float(np.max(loud_db))}
            results["acoustic_features"] = {"rms_mean": rms, "peak_amplitude": peak, "dynamic_range": dynamic_range, "spectral_centroid_mean": float(np.mean(cent)), "spectral_bandwidth_mean": float(np.mean(bw)), "zero_crossing_rate_mean": float(np.mean(zcr)), "mfcc_mean": mfcc_mean, "mfcc_var": mfcc_var, "chroma_mean": chroma_mean, "spectral_contrast_mean": spec_con_mean, "onset_density_per_sec": onset_density, "loudness_db_summary": loud_summary }
            print("Extended acoustic features extracted successfully")
        except Exception as e: print(f"Error extracting acoustic features: {e}"); results["acoustic_features"] = {"error": str(e)}; onset_env = None

        # --- Step 3 & 4 Combined: Tempo/Beat Estimation (Simplified) ---
        MIN_DURATION_FOR_TEMPO = 12.5
        tempo = None
        beat_times = []
        onset_times = [] # Onset times calculated here for consistency
        results["rhythm"] = {}

        if duration < MIN_DURATION_FOR_TEMPO:
            print(f"Audio duration ({duration:.2f}s) < {MIN_DURATION_FOR_TEMPO}s. Skipping tempo/beat analysis.")
            results["rhythm"] = {"status": f"Skipped (duration < {MIN_DURATION_FOR_TEMPO}s)"}
        elif onset_env is None:
             print("Skipping tempo/beat analysis due to missing onset envelope.")
             results["rhythm"] = {"status": "Skipped (missing onset envelope)"}
        else:
            try:
                print("Estimating tempo and beats using librosa.beat.beat_track...")
                # Use beat_track directly, NO prior, NO octave correction
                tempo_est_from_bt, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                tempo = float(tempo_est_from_bt[0]) # Use the tempo estimated by beat_track
                beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
                results["rhythm"]["tempo_bpm"] = tempo
                print(f"Detected tempo: {tempo:.2f} BPM")
                print(f"Found {len(beat_frames)} beats.")
            except Exception as e:
                print(f"Error during tempo/beat estimation: {e}")
                results["rhythm"]["error"] = str(e)

        # Calculate onset times regardless of tempo success (if onset_env available)
        try:
            if onset_env is not None:
                onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
                onset_times = librosa.frames_to_time(onsets, sr=sr).tolist()
            else:
                print("Cannot calculate onset times (missing onset envelope).")
        except Exception as e:
             print(f"Error calculating onset times: {e}")


        # --- Step 4b: Other Perceptual Features ---
        try:
            # --- Pitch (pYIN) ---
            f0, _, _ = librosa.pyin(waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            f0_clean = f0[~np.isnan(f0)]
            f0_stats = {"f0_mean_hz": float(np.mean(f0_clean)) if len(f0_clean) > 0 else None, "f0_median_hz": float(np.median(f0_clean)) if len(f0_clean) > 0 else None, "f0_var": float(np.var(f0_clean)) if len(f0_clean) > 1 else None}

            # --- Tonal centroid (tonnetz) ---
            tonnetz_mean = None
            if "chroma_mean" in results.get("acoustic_features", {}) and results["acoustic_features"]["chroma_mean"]:
                try: harmonic = librosa.effects.harmonic(waveform); tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr); tonnetz_mean = np.mean(tonnetz, axis=1).tolist()
                except Exception as tonnetz_e: print(f"Warning: Error calculating tonnetz: {tonnetz_e}")
            else: print("Skipping tonnetz calculation due to missing chroma features.")

            results["perceptual_features"] = {"beat_times_s": beat_times, "onset_times_s": onset_times, **f0_stats, "tonnetz_mean": tonnetz_mean}
            print("Other perceptual features extracted successfully")
        except Exception as e: print(f"Error extracting other perceptual features: {e}"); results["perceptual_features"] = {"error": str(e)}


        # --- Step 5a: HF audio classification (WITH FILTERING) ---
        HF_SCORE_THRESHOLD = 0.1 # Threshold for filtering
        if self.audio_classifier:
            try:
                hf_preds_raw = self.audio_classifier(audio_file)
                filtered_hf_preds = []
                if isinstance(hf_preds_raw, list) and hf_preds_raw:
                    preds_to_filter = hf_preds_raw[0] if isinstance(hf_preds_raw[0], list) else hf_preds_raw
                    filtered_hf_preds = [pred for pred in preds_to_filter if pred['score'] >= HF_SCORE_THRESHOLD]
                results["hf_classifications"] = filtered_hf_preds
                print(f"HF audio classifications obtained (kept {len(filtered_hf_preds)} above threshold {HF_SCORE_THRESHOLD}).")
            except Exception as e: print(f"Error during HF audio classification: {e}"); results["hf_classifications"] = {"error": str(e)}
        else: results["hf_classifications"] = {"status": "Pipeline not loaded"}

        # --- Step 5b: CLAP zero‑shot classification ---
        # ... (No changes - Keep top 5 CLAP results) ...
        audio_embs = None
        if self.clap_model:
            try:
                labels = ["piano", "guitar", "drum kit", "bass guitar", "synthesizer", "vocals", "orchestra", "electronic beat", "ambient", "sound effect"]
                text_embs = self.clap_model.get_text_embeddings(labels)
                audio_embs = self.clap_model.get_audio_embeddings([audio_file])
                sims = self.clap_model.compute_similarity(audio_embs, text_embs)[0].tolist()
                top5 = sorted(zip(labels, sims), key=lambda x: x[1], reverse=True)[:5]
                results["clap_zero_shot"] = [{"label": lbl, "score": float(score)} for lbl, score in top5]
                print("CLAP zero-shot classifications obtained.")
            except Exception as e: print(f"Error during CLAP zero-shot classification: {e}"); results["clap_zero_shot"] = {"error": str(e)}; audio_embs = None
        else: results["clap_zero_shot"] = {"status": "CLAP model not loaded"}; audio_embs = None


        # --- Step 5c: CLAP Mood Classification ---
        # ... (No changes - Keep top 2 moods) ...
        if self.clap_model and audio_embs is not None:
            try:
                mood_labels = ["happy", "sad", "energetic", "calm", "aggressive", "dreamy", "epic", "mysterious", "tense"]
                mood_text_embs = self.clap_model.get_text_embeddings(mood_labels)
                mood_sims = self.clap_model.compute_similarity(audio_embs, mood_text_embs)[0].tolist()
                top_moods = sorted(zip(mood_labels, mood_sims), key=lambda x: x[1], reverse=True)[:2]
                results["mood_classification"] = [{"label": lbl, "score": float(score)} for lbl, score in top_moods]
                print(f"CLAP mood classification obtained: {results['mood_classification']}")
            except Exception as e: print(f"Error during CLAP mood classification: {e}"); results["mood_classification"] = {"error": str(e)}
        elif not self.clap_model: results["mood_classification"] = {"status": "CLAP model not loaded"}
        else: results["mood_classification"] = {"status": "Skipped (audio embedding failed)"}


        # --- Step 6: Music genre classification (WITH FILTERING) ---
        GENRE_SCORE_THRESHOLD = 0.1 # Threshold for filtering
        if self.genre_classifier:
            try:
                 genre_preds_raw = self.genre_classifier(audio_file)
                 filtered_genre_preds = [pred for pred in genre_preds_raw if pred['score'] >= GENRE_SCORE_THRESHOLD]
                 results["genre"] = filtered_genre_preds
                 # 1. Create the list of formatted strings first
                 genre_strings = [f'{g["label"]} ({g["score"]:.2f})' for g in filtered_genre_preds]
                 # 2. Then print the list
                 print(f"Top genres obtained (kept {len(filtered_genre_preds)} above threshold {GENRE_SCORE_THRESHOLD}): {genre_strings}")
            except Exception as e: print(f"Error during music genre classification: {e}"); results["genre"] = {"error": str(e)}
        else: results["genre"] = {"status": "Pipeline not loaded"}

        # --- Step 7: Key / mode detection ---
        # ... (No changes) ...
        try:
            if "chroma_mean" in results.get("acoustic_features", {}) and results["acoustic_features"]["chroma_mean"]:
                chroma_arr = np.array(results["acoustic_features"]["chroma_mean"])
                major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
                minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
                best = (-np.inf, None, None)
                for mode, profile in [("major", major_profile), ("minor", minor_profile)]:
                    for i in range(12):
                        corr = np.corrcoef(chroma_arr, np.roll(profile, i))[0,1]
                        if corr > best[0]: best = (corr, i, mode)
                names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                corr, idx, mode = best
                if idx is not None: results["key"] = {"root": names[idx], "mode": mode, "correlation": float(corr)}; print(f"Estimated key: {names[idx]} {mode} (corr={corr:.2f})")
                else: results["key"] = {"status": "Could not determine key"}; print("Could not determine key.")
            else: results["key"] = {"status": "Skipped (missing chroma features)"}; print("Skipping key detection due to missing chroma features.")
        except Exception as e: print(f"Error detecting key: {e}"); results["key"] = {"error": str(e)}

        # --- Step 8: Timbre features ---
        # ... (No changes) ...
        try:
            rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)[0]; rolloff_mean = float(np.mean(rolloff))
            flatness = librosa.feature.spectral_flatness(y=waveform)[0]; flatness_mean = float(np.mean(flatness))
            flux_mean = float(np.mean(onset_env)) if onset_env is not None else None
            mfcc_delta_mean = None
            if "mfcc_mean" in results.get("acoustic_features", {}) and "error" not in results.get("acoustic_features", {}):
                mfcc_delta = librosa.feature.delta(mfcc); mfcc_delta_mean = np.mean(mfcc_delta, axis=1).tolist()
            else: print("Skipping MFCC delta calculation (missing/failed MFCCs).")
            results["timbre_features"] = {"spectral_rolloff_85pct_mean": rolloff_mean, "spectral_flatness_mean": flatness_mean, "spectral_flux_mean": flux_mean, "mfcc_delta_mean": mfcc_delta_mean}
            print("Extended timbre features extracted successfully")
        except Exception as e: print(f"Error extracting timbre features: {e}"); results["timbre_features"] = {"error": str(e)}


        # --- Step 9: Generate human‑readable insights ---
        insights = self._generate_insights(results)
        results["insights"] = insights
        # ... (Print insights) ...


        # --- Step 10: Generate prose description with OpenAI ---
        print("\n--- Generating Prose Description (OpenAI) ---")
        description = self.describe_with_openai(results)
        results["openai_description"] = description
        print(description)


        # --- Step 11: Generate new audio clip with MusicGen ---
        if generate_audio:
            # ... (Fallback prompt logic - uses filtered genre now) ...
            generation_prompt = description
            if not self.openai_client or "error" in description.lower() or "skipped" in description.lower():
                fallback_parts = []
                mood_info = results.get("mood_classification", [])
                if isinstance(mood_info, list) and mood_info and isinstance(mood_info[0], dict) and 'label' in mood_info[0]: fallback_parts.append(mood_info[0]['label'])
                genre_info = results.get("genre", []) # Filtered list
                if isinstance(genre_info, list) and genre_info and isinstance(genre_info[0], dict) and 'label' in genre_info[0]: fallback_parts.append(genre_info[0]['label'])
                rhythm_info = results.get("rhythm", {})
                if isinstance(rhythm_info, dict) and rhythm_info.get("tempo_bpm"):
                    tempo_val = rhythm_info["tempo_bpm"]; desc = "slow" if tempo_val<60 else "moderate" if tempo_val<120 else "fast"; fallback_parts.append(f"{desc} tempo {int(tempo_val)} bpm")
                key_info = results.get("key", {})
                if isinstance(key_info, dict) and "root" in key_info and "mode" in key_info: fallback_parts.append(f"in {key_info['root']} {key_info['mode']}")
                clap_info = results.get("clap_zero_shot", [])
                if isinstance(clap_info, list) and clap_info and isinstance(clap_info[0], dict) and 'label' in clap_info[0]: fallback_parts.append(f"featuring {clap_info[0]['label']}")
                generation_prompt = ", ".join(filter(None, fallback_parts))
                if not generation_prompt: generation_prompt = "instrumental music loop"
                print(f"Using fallback prompt for generation: '{generation_prompt}'")

            generated_audio_file = self.generate_with_musicgen(generation_prompt, duration=int(results.get("basic_info", {}).get("duration_s", 10)))
            if generated_audio_file: results["generated_audio"] = {"prompt_used": generation_prompt, "file": generated_audio_file}
            else: results["generated_audio"] = {"error": "Music generation failed."}
        else: results["generated_audio"] = {"status": "Generation skipped by request."}

        print(f"\n==== ANALYSIS COMPLETE: {os.path.basename(audio_file)} ====\n")
        return results


    def _generate_insights(self, results):
        """Generates a list of human-readable insights from the analysis results."""
        insights = []
        def get_nested(data, keys, default=None): # Basic helper
            val = data;
            for key in keys:
                if isinstance(val, dict) and key in val: val = val[key]
                else: return default
            if isinstance(val, dict) and ('error' in val or 'status' in val) and default != val: return default
            if isinstance(default, list) and not isinstance(val, list): return default
            return val

        # --- Get data ---
        feats = results.get("acoustic_features", {}); zs = get_nested(results, ["clap_zero_shot"], default=[]); moods = get_nested(results, ["mood_classification"], default=[])
        rhythm_info = results.get("rhythm", {}); tempo = get_nested(rhythm_info, ["tempo_bpm"]) # No reliability fetched
        duration = get_nested(results, ["basic_info", "duration_s"]); perf = results.get("perceptual_features", {}); timb = results.get("timbre_features", {})
        hf_classes = get_nested(results, ["hf_classifications"], default=[]); genre = get_nested(results, ["genre"], default=[]); key = results.get("key", {})

        # 1) CLAP sound
        if zs and isinstance(zs[0], dict) and "label" in zs[0]: insights.append(f'Most associated sound: "{zs[0]["label"]}" (score={zs[0]["score"]:.2f}).')
        # 2) Tempo (Simplified)
        if tempo is not None: desc = "slow" if tempo<80 else "moderate" if tempo<120 else "fast"; insights.append(f'Tempo: {tempo:.0f} BPM ({desc}).')
        elif "status" in rhythm_info: insights.append(f'Tempo: {rhythm_info["status"]}.')
        # 3) Mood
        if moods and isinstance(moods[0], dict) and "label" in moods[0]: top_mood = moods[0]; insight_str = f'Dominant mood: "{top_mood["label"]}" (score={top_mood["score"]:.2f})'; insights.append(insight_str + '.') # Simplified mood insight slightly
        # 4) Timbre/Brightness
        cent = get_nested(feats, ["spectral_centroid_mean"]);
        if cent is not None: tone = "dark/mellow" if cent < 1500 else "neutral" if cent < 3000 else "bright/sharp"; insights.append(f'Timbre: {tone} (centroid ≈{cent:.0f} Hz).')
        # 5) Dynamics
        dr = get_nested(feats, ["dynamic_range"]);
        if dr is not None: dyn = "compressed/narrow range" if dr < 10 else "moderate range" if dr < 18 else "dynamic/wide range"; insights.append(f'Dynamics: {dyn} (peak/RMS ≈{dr:.1f}).')
        # 6) Texture/Percussiveness
        od = get_nested(feats, ["onset_density_per_sec"]);
        if od is not None: desc = "sustained/sparse" if od < 2 else "moderately percussive" if od < 6 else "highly percussive/dense"; insights.append(f'Texture: {desc} ({od:.1f} onsets/sec).')
        # 7) Loudness
        mean_db = get_nested(feats, ["loudness_db_summary", "mean_db"]);
        if mean_db is not None: lvl = "very quiet" if mean_db < -30 else "quiet" if mean_db < -18 else "moderate" if mean_db < -9 else "loud"; insights.append(f'Loudness: {lvl} (avg ≈{mean_db:.1f} dBFS).')
        # 8) Pitch Variation
        f0_var = get_nested(perf, ["f0_var"]); f0_mean = get_nested(perf, ["f0_mean_hz"])
        if f0_var is not None: pitch_var = "stable pitch" if f0_var < 500 else "moderate pitch variation" if f0_var < 5000 else "highly varying pitch/melody"; insights.append(f'Pitch: {pitch_var} (F0 variance ≈{f0_var:.0f}).')
        elif f0_mean is not None: insights.append(f'Detected fundamental frequency (avg F0 ≈{f0_mean:.1f} Hz).')
        # 9) Tonal Character
        flat = get_nested(timb, ["spectral_flatness_mean"]);
        if flat is not None:
            if flat < 0.05: insights.append("Sound character: strongly tonal.")
            elif flat < 0.2: insights.append("Sound character: mix of tone and noise.")
            else: insights.append("Sound character: noisy or complex.")
        # 10) HF Class (Top 1 filtered)
        if hf_classes and isinstance(hf_classes[0], dict) and "label" in hf_classes[0]: insights.append(f'Top associated sound event (HF): "{hf_classes[0]["label"]}" (score={hf_classes[0]["score"]:.2f}).')
        # 11) Genre (Top 1 filtered)
        if genre and isinstance(genre[0], dict) and "label" in genre[0]: insights.append(f'Predicted genre: "{genre[0]["label"]}" (score={genre[0]["score"]:.2f}).')
        elif "status" in results.get("genre", {}): insights.append(f'Predicted genre: {results["genre"]["status"]}.')
        # 12) Key
        if isinstance(key, dict) and "root" in key and "mode" in key: insights.append(f'Estimated key: {key["root"]} {key["mode"]} (corr={key["correlation"]:.2f}).')

        return insights


# --- Main execution block ---
if __name__ == "__main__":
    import traceback # Ensure traceback is imported here or globally

    parser = argparse.ArgumentParser(description="Analyze an audio file and optionally generate a similar one.")
    parser.add_argument("audio_file", help="Path to the input audio file.")
    parser.add_argument("--mode", choices=['terminal', 'gradio'], default='terminal', help="Run in terminal or launch Gradio UI.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage.")
    # REMOVED: parser.add_argument("--skip-generation", action="store_true", help="Only analyze, do not generate new audio (Terminal mode only).")
    # ADDED: --generate flag
    parser.add_argument("--generate", action="store_true", default=False, help="Generate new audio based on analysis (Terminal mode only).") # Default is False

    args = parser.parse_args()

    if args.mode == 'gradio':
        # ... (Gradio launch code remains the same) ...
        try:
            from gradio_ui import generate_ui
            print("Launching Gradio UI...")
            generate_ui()
        except ImportError: print("Error: Could not import gradio_ui.py."); traceback.print_exc()
        except Exception as e: print(f"Error launching Gradio UI: {e}"); traceback.print_exc()

    elif args.mode == 'terminal':
        print("Running in terminal mode...")
        try:
            analyzer = SimpleAudioAnalyzer(use_gpu=not args.no_gpu)

            # Perform analysis, pass the new generate flag directly
            results = analyzer.analyze_file(
                args.audio_file,
                generate_audio=args.generate # Use the --generate flag value
            )

            # --- Terminal Summary ---
            print("\n--- Terminal Summary ---")
            if "error" in results:
                print(f"Analysis failed: {results['error']}")
            else:
                # Define helper locally or ensure it's accessible if moved outside class
                def get_nested(data, keys, default=None): # Local helper for summary
                    val = data;
                    for key in keys:
                        if isinstance(val, dict) and key in val: val = val[key]
                        else: return default
                    if isinstance(val, dict) and ('error' in val or 'status' in val): return default
                    return val

                print(f"Analyzed: {get_nested(results, ['basic_info', 'filename'], args.audio_file)}")
                duration = get_nested(results, ['basic_info', 'duration_s'])
                print(f"Duration: {duration:.2f}s" if duration is not None else "Duration: N/A")

                print(f"Insights:")
                insights = results.get('insights', [])
                if insights:
                    for insight in insights: print(f"  • {insight}")
                else: print("  No insights generated.")

                # Update generation status reporting based on the new flag
                gen_info = results.get('generated_audio', {})
                if args.generate: # Check if generation was requested
                    print(f"\nAudio Generation (requested):") # Update message
                    if "file" in gen_info and gen_info["file"]:
                        print(f"  Prompt Used: {gen_info.get('prompt_used', 'N/A')}")
                        print(f"  Saved to: {gen_info['file']}")
                    elif "error" in gen_info:
                         print(f"  Generation Failed: {gen_info['error']}")
                    elif "status" in gen_info: # Should contain 'Generation skipped by request' if analyze_file was called with generate_audio=False
                         print(f"  Status: {gen_info['status']}")
                    else:
                         print(f"  Generation did not produce an output file (check logs).")
                else:
                    # Message when --generate flag is NOT used
                    print("\nAudio generation not requested (use --generate flag to enable).")

        except FileNotFoundError as e:
            print(f"Error: Input audio file not found: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during terminal execution: {e}")
            traceback.print_exc()

    else:
        # This case should not be reachable due to argparse choices
        print(f"Unknown mode: {args.mode}")

# The rest of the SimpleAudioAnalyzer class remains unchanged.