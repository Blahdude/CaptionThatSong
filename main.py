import os
import json
import warnings
from datetime import datetime

import librosa
import numpy as np
import torch
from msclap import CLAP   # core CLAP class for embeddings/similarity
warnings.filterwarnings('ignore')


class SimpleAudioAnalyzer:
    """Analyzer with audio-classification, genre-classification, CLAP zero‑shot, and perceptual features"""

    def __init__(self, use_gpu=True):
        # Select device
        self.device = 0 if (torch.cuda.is_available() and use_gpu) else -1
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")

        # Load standard audio-classification (AudioSet) pipeline
        self._load_audio_classifier()

        # Load music‑genre classification pipeline
        try:
            from transformers import pipeline
            print("Loading music-genre pipeline...")
            self.genre_classifier = pipeline(
                "audio-classification",
                model="m3hrdadfi/wav2vec2-base-100k-gtzan-music-genres",
                device=self.device
            )
            print("Genre pipeline ready")
        except Exception as e:
            print(f"Could not load genre pipeline: {e}")
            self.genre_classifier = None

        # Initialize CLAP for zero-shot
        print("Loading CLAP zero-shot model...")
        self.clap_model = CLAP(version="2023", use_cuda=(self.device == 0))
        print("CLAP model loaded successfully")

    def _load_audio_classifier(self):
        try:
            from transformers import pipeline
            print("Loading standard audio-classification pipeline...")
            self.audio_classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=self.device,
                return_all_scores=True,
                function_to_apply="softmax"
            )
            print("Audio classification pipeline ready")
        except Exception as e:
            print(f"Error loading HF audio pipeline: {e}")
            self.audio_classifier = None

    def analyze_file(self, audio_file):
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        print(f"\n==== ANALYSIS START: {audio_file} ====\n")
        results = {}

        # Step 1: load waveform & basic info
        waveform, sr = librosa.load(audio_file, sr=22050)
        duration = librosa.get_duration(y=waveform, sr=sr)
        results["basic_info"] = {
            "duration_s": float(duration),
            "sample_rate": sr,
            "num_samples": len(waveform)
        }
        print(f"Loaded audio: {duration:.2f} seconds, {sr} Hz")

        # Step 2: extended low‑level features
        rms_env = librosa.feature.rms(y=waveform)[0]
        rms = float(np.mean(rms_env))
        peak = float(np.max(np.abs(waveform)))
        dynamic_range = float(peak / rms) if rms > 0 else None

        cent = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        bw = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=waveform)[0]

        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        mfcc_var = np.var(mfcc, axis=1).tolist()

        chroma = librosa.feature.chroma_cqt(y=waveform, sr=sr)
        chroma_mean = np.mean(chroma, axis=1).tolist()

        spec_con = librosa.feature.spectral_contrast(y=waveform, sr=sr)
        spec_con_mean = np.mean(spec_con, axis=1).tolist()

        onset_env = librosa.onset.onset_strength(y=waveform, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_density = len(onsets) / duration if duration > 0 else 0.0

        loud_db = librosa.amplitude_to_db(rms_env, ref=np.max)
        loud_summary = {
            "min_db": float(np.min(loud_db)),
            "mean_db": float(np.mean(loud_db)),
            "max_db": float(np.max(loud_db))
        }

        results["acoustic_features"] = {
            "rms_mean": rms,
            "peak_amplitude": peak,
            "dynamic_range": dynamic_range,
            "spectral_centroid_mean": float(np.mean(cent)),
            "spectral_bandwidth_mean": float(np.mean(bw)),
            "zero_crossing_rate_mean": float(np.mean(zcr)),
            "mfcc_mean": mfcc_mean,
            "mfcc_var": mfcc_var,
            "chroma_mean": chroma_mean,
            "spectral_contrast_mean": spec_con_mean,
            "onset_density_per_sec": onset_density,
            "loudness_db_summary": loud_summary
        }
        print("Extended acoustic features extracted successfully")

        # Step 3: tempo & rhythm
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
        tempo = float(tempo)
        results["rhythm"] = { "tempo_bpm": tempo }
        print(f"Detected tempo: {tempo:.2f} BPM")

        # Step 4: mid‑level perceptual features
        try:
            # beat and onset times
            _, beat_frames = librosa.beat.beat_track(y=waveform, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

            # pitch (pYIN)
            f0, _, _ = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            f0_clean = f0[~np.isnan(f0)]
            f0_stats = {
                "f0_mean_hz": float(np.mean(f0_clean)) if len(f0_clean) else None,
                "f0_median_hz": float(np.median(f0_clean)) if len(f0_clean) else None,
                "f0_var": float(np.var(f0_clean)) if len(f0_clean) else None
            }

            # tonal centroid (tonnetz)
            harmonic = librosa.effects.harmonic(waveform)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1).tolist()

            results["perceptual_features"] = {
                "beat_times_s": beat_times,
                "onset_times_s": onset_times,
                **f0_stats,
                "tonnetz_mean": tonnetz_mean
            }
            print("Mid-level perceptual features extracted successfully")
        except Exception as e:
            print(f"Error extracting perceptual features: {e}")
            results["perceptual_features"] = {"error": str(e)}

        # Step 5a: HF audio classification
        if self.audio_classifier:
            results["hf_classifications"] = self.audio_classifier(audio_file)

        # Step 5b: CLAP zero‑shot classification
        labels = ["piano","guitar","drum","trumpet","vocals"]
        text_embs = self.clap_model.get_text_embeddings(labels)
        audio_embs = self.clap_model.get_audio_embeddings([audio_file])
        sims = self.clap_model.compute_similarity(audio_embs, text_embs)[0]
        top5 = sorted(zip(labels, sims), key=lambda x: x[1], reverse=True)[:5]
        results["clap_zero_shot"] = [
            {"label": lbl, "score": float(score)} for lbl, score in top5
        ]

        # Step 6: Music genre classification
        if self.genre_classifier:
            genre_preds = self.genre_classifier(audio_file)
            results["genre"] = genre_preds[:3]
            print("Top genres:", results["genre"])

        # Step 7: Key / mode detection (Krumhansl–Schmuckler)
        try:
            chroma_arr = np.array(results["acoustic_features"]["chroma_mean"])
            major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
            minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
            best = (-1,None,None)
            for mode, profile in [("major", major_profile), ("minor", minor_profile)]:
                for i in range(12):
                    corr = np.corrcoef(chroma_arr, np.roll(profile, i))[0,1]
                    if corr > best[0]:
                        best = (corr, i, mode)
            names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            corr, idx, mode = best
            results["key"] = {"root": names[idx], "mode": mode, "correlation": float(corr)}
            print(f"Estimated key: {names[idx]} {mode} (corr={corr:.2f})")
        except Exception as e:
            print(f"Error detecting key: {e}")
            results["key"] = {"error": str(e)}

        # Spectral rolloff (85% energy cutoff)
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)[0]
        rolloff_mean = float(np.mean(rolloff))

        # Spectral flatness (tonal vs noisy)
        flatness = librosa.feature.spectral_flatness(y=waveform)[0]
        flatness_mean = float(np.mean(flatness))

        # Spectral flux (frame‑to‑frame difference)
        # here we reuse onset_env from before – it's exactly the spectral flux
        flux_mean = float(np.mean(onset_env))

        # MFCC deltas (1st derivative)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1).tolist()

        # Add into your results dict under a new “timbre_features” key
        results["timbre_features"] = {
            "spectral_rolloff_85pct_mean": rolloff_mean,
            "spectral_flatness_mean": flatness_mean,
            "spectral_flux_mean": flux_mean,
            "mfcc_delta_mean": mfcc_delta_mean,
        }
        print("Extended timbre features extracted successfully")

        # Step 8: generate human‑readable insights
        insights = self._generate_insights(results)
        results["insights"] = insights
        print("Insights generated successfully:")
        for line in insights:
            print(" -", line)

        # Save full JSON
        out_name = f"audio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_name, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Analysis results saved to {out_name}")

        return results

    def _generate_insights(self, results):
        insights = []
        feats = results.get("acoustic_features", {})
        zs = results.get("clap_zero_shot", [])
        tempo = results.get("rhythm", {}).get("tempo_bpm")
        duration = results.get("basic_info", {}).get("duration_s")

        # 1) top instrument
        if zs:
            top = zs[0]
            insights.append(f'The clip is most strongly associated with "{top["label"]}" (score={top["score"]:.2f}).')

        # 2) tempo
        if tempo is not None:
            desc = "slow" if tempo<60 else "moderate" if tempo<120 else "fast"
            insights.append(f'Its tempo is about {tempo:.0f} BPM, which is {desc}.')

        # 3) brightness
        cent = feats.get("spectral_centroid_mean")
        if cent is not None:
            tone = "warm/mellow" if cent<1500 else "bright"
            insights.append(f'The spectrum sounds {tone} (centroid ≈{cent:.0f} Hz).')

        # 4) dynamics
        dr = feats.get("dynamic_range")
        if dr is not None:
            dyn = "wide dynamic range" if dr>20 else "narrow dynamic range"
            insights.append(f'It has a {dyn} (peak/RMS ≈{dr:.1f}).')

        # 5) percussiveness
        od = feats.get("onset_density_per_sec")
        if od is not None:
            desc = "very percussive" if od>5 else "more sustained"
            insights.append(f'Onset density is {od:.1f}/s → it’s {desc}.')

        # 6) loudness
        mean_db = feats.get("loudness_db_summary", {}).get("mean_db")
        if mean_db is not None:
            lvl = "very quiet" if mean_db< -30 else "quiet" if mean_db< -15 else "moderately loud" if mean_db< -5 else "loud"
            insights.append(f'Mean loudness ≈{mean_db:.1f} dB (it sounds {lvl}).')

        perf = results.get("perceptual_features", {})

        # 7) beats
        beats = perf.get("beat_times_s", [])
        if beats and duration:
            insights.append(f'There are {len(beats)} beat events over {duration:.1f}s (~{duration/len(beats):.2f}s between beats).')

        # 8) onsets
        onsets = perf.get("onset_times_s", [])
        if onsets and duration:
            insights.append(f'{len(onsets)} transients detected (~{len(onsets)/duration:.1f} onsets per second).')

        # 9) pitch stats
        f0_mean = perf.get("f0_mean_hz")
        f0_med = perf.get("f0_median_hz")
        f0_var = perf.get("f0_var")
        if f0_mean is not None and f0_med is not None:
            insights.append(f'Average F0 ≈{f0_mean:.1f}Hz (median {f0_med:.1f}Hz).')
        if f0_var is not None:
            insights.append("High F0 variance suggests melodic movement." if f0_var>1000 else "Low F0 variance indicates a stable tone.")

        # 10) tonnetz
        if perf.get("tonnetz_mean"):
            insights.append("Tonal-centroid (tonnetz) features extracted, reflecting harmonic color.")

        # 11) genre
        genre = results.get("genre", [])
        if genre:
            topg = genre[0]
            insights.append(f'Predicted music genre: "{topg["label"]}" (score={topg["score"]:.2f}).')

        # 12) key
        key = results.get("key", {})
        if "root" in key:
            insights.append(f'Estimated key: {key["root"]} {key["mode"]} (correlation={key["correlation"]:.2f}).')

        # 13) Timbre / Texture insights
        timb = results.get("timbre_features", {})

        # Spectral flatness
        flat = timb.get("spectral_flatness_mean")
        if flat is not None:
            if flat < 0.1:
                insights.append("The sound is very tonal (low spectral flatness).")
            elif flat < 0.3:
                insights.append("The sound has a balanced mix of tone and noise.")
            else:
                insights.append("The sound is noisy or percussive (high spectral flatness).")

        # Spectral rolloff
        roll = timb.get("spectral_rolloff_85pct_mean")
        if roll is not None:
            insights.append(f"The high‑frequency rolloff (85%) is at ≈{roll:.0f} Hz.")

        # Spectral flux
        flux = timb.get("spectral_flux_mean")
        if flux is not None:
            if flux > 0.1:
                insights.append("There are rapid spectral changes (high flux).")
            else:
                insights.append("Spectral content is relatively stable (low flux).")

        # MFCC delta
        md = timb.get("mfcc_delta_mean", [])
        if md:
            avg_md = float(np.mean(md))
            insights.append(f"Average MFCC delta is {avg_md:.2f}, indicating timbral movement.")

        return insights


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", required=True)
    parser.add_argument("--no_gpu", action="store_true")
    args = parser.parse_args()

    analyzer = SimpleAudioAnalyzer(use_gpu=not args.no_gpu)
    analyzer.analyze_file(args.audio_file)


if __name__ == "__main__":
    main()