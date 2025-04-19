import os
import librosa
import numpy as np
import json
import torch
from transformers import pipeline
from datetime import datetime
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

class SimpleAudioAnalyzer:
    """A robust analyzer for instrumental audio files with minimal dependencies"""
    
    def __init__(self, use_gpu=True):
        """Initialize the analyzer with GPU support if available"""
        self.device = 0 if (torch.cuda.is_available() and use_gpu) else -1
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Pre-load model
        self._load_models()
        
    def _load_models(self):
        """Load ML models for audio analysis"""
        try:
            print("Loading audio classification model...")
            self.audio_classifier = pipeline(
                "audio-classification", 
                model="MIT/ast-finetuned-audioset-10-10-0.4593", 
                device=self.device
            )
            print("Audio classification model loaded successfully")
        except Exception as e:
            print(f"Error loading audio classification model: {e}")
            self.audio_classifier = None
    
    def analyze_file(self, audio_file):
        """Analyze an audio file and return results"""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        print(f"\n==== STARTING AUDIO ANALYSIS: {audio_file} ====\n")
        results = {}
        
        # Step 1: Extract waveform and basic properties
        try:
            # Load audio
            waveform, sample_rate = librosa.load(audio_file, sr=22050)
            duration = librosa.get_duration(y=waveform, sr=sample_rate)
            results["basic_info"] = {
                "duration": float(duration),
                "sample_rate": sample_rate,
                "num_samples": len(waveform)
            }
            print(f"Loaded audio: {duration:.2f} seconds, {sample_rate} Hz")
        except Exception as e:
            print(f"Error loading audio: {e}")
            results["basic_info"] = {"error": str(e)}
            
        # Step 2: Extract simple features that work reliably
        try:
            # Volume/amplitude
            rms = float(np.mean(librosa.feature.rms(y=waveform)))
            peak = float(np.max(np.abs(waveform)))
            
            # Spectral features
            spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]))
            spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)[0]))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(waveform)[0]))
            
            results["acoustic_features"] = {
                "rms_volume": rms,
                "peak_amplitude": peak,
                "spectral_centroid": spec_cent,
                "spectral_bandwidth": spec_bw,
                "zero_crossing_rate": zcr
            }
            print("Basic acoustic features extracted successfully")
        except Exception as e:
            print(f"Error extracting features: {e}")
            results["acoustic_features"] = {"error": str(e)}
            
        # Step 3: Detect tempo & rhythm
        try:
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate)
            results["rhythm"] = {
                "tempo_bpm": float(tempo)
            }
            print(f"Detected tempo: {tempo} BPM")
        except Exception as e:
            print(f"Error detecting tempo: {e}")
            results["rhythm"] = {"error": str(e)}
            
        # Step 4: Audio classification
        try:
            if self.audio_classifier:
                print("Classifying audio content...")
                predictions = self.audio_classifier(audio_file)
                
                # Extract top predictions
                results["classifications"] = {
                    "top_classes": [
                        {"label": pred["label"], "confidence": float(pred["score"])} 
                        for pred in predictions[:5]  # Get top 5
                    ]
                }
                
                print("Top classifications:")
                for i, pred in enumerate(predictions[:3]):
                    print(f"  {i+1}. {pred['label']}: {pred['score']:.4f}")
            else:
                results["classifications"] = {"status": "classifier not available"}
        except Exception as e:
            print(f"Error in classification: {e}")
            results["classifications"] = {"error": str(e)}
            
        # Step 5: Generate insights
        try:
            insights = self._generate_insights(results)
            results["insights"] = insights
            print("Insights generated successfully")
        except Exception as e:
            print(f"Error generating insights: {e}")
            results["insights"] = {"error": str(e)}
            
        print("\n==== ANALYSIS COMPLETE ====\n")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"audio_analysis_{timestamp}.json"
        
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Analysis results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        return results
    
    def _generate_insights(self, results):
        """Generate human-readable insights from analysis results"""
        insights = {}
        
        # Get data from results
        basic = results.get("basic_info", {})
        acoustic = results.get("acoustic_features", {})
        rhythm = results.get("rhythm", {})
        classes = results.get("classifications", {}).get("top_classes", [])
        
        # Basic audio quality
        if "duration" in basic:
            duration = basic["duration"]
            if duration < 30:
                length_type = "Short"
            elif duration < 180:
                length_type = "Medium"
            else:
                length_type = "Long"
                
            # Format as mm:ss
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            insights["duration_formatted"] = f"{minutes}:{seconds:02d}"
        
        # Volume analysis
        if "rms_volume" in acoustic:
            rms = acoustic["rms_volume"]
            if rms < 0.05:
                quality = "Low"
            elif rms < 0.1:
                quality = "Medium"
            else:
                quality = "High"
                
            insights["quality_assessment"] = f"{quality} quality {length_type} audio"
        
        # Musical qualities  
        if "tempo_bpm" in rhythm:
            tempo = rhythm["tempo_bpm"]
            if tempo < 70:
                tempo_desc = "Slow"
            elif tempo < 120:
                tempo_desc = "Medium"
            else:
                tempo_desc = "Fast"
                
            insights["tempo"] = f"{tempo_desc} tempo ({int(tempo)} BPM)"
        
        # Instrument/content type
        if classes:
            top_class = classes[0]["label"]
            insights["specific_sound"] = top_class
            
            # Check for instrument related terms
            instruments = ["piano", "guitar", "drum", "bass", "violin", "cello", 
                          "trumpet", "saxophone", "flute", "organ"]
            
            found_instrument = False
            for instrument in instruments:
                if instrument in top_class.lower():
                    insights["instrument_type"] = instrument
                    found_instrument = True
                    break
            
            if not found_instrument:
                # Check for music related terms
                music_terms = ["music", "audio", "sound", "song"]
                if any(term in top_class.lower() for term in music_terms):
                    insights["instrument_type"] = "unidentified music"
                else:
                    insights["instrument_type"] = "non-musical audio"
        
        # Sound characteristics based on spectral features
        if "spectral_centroid" in acoustic:
            centroid = acoustic["spectral_centroid"]
            if centroid < 1000:
                tone = "Dark/warm"
            elif centroid < 3000:
                tone = "Balanced"
            else:
                tone = "Bright/sharp"
                
            insights["tonal_quality"] = tone
            
        # Estimate a simple genre or style
        if classes and "tempo_bpm" in rhythm:
            top_label = classes[0]["label"].lower()
            tempo = rhythm["tempo_bpm"]
            
            # Very simple genre estimation
            if "piano" in top_label:
                if tempo < 80:
                    genre = "classical/ambient"
                else:
                    genre = "jazz/contemporary"
            elif "guitar" in top_label:
                if tempo > 120:
                    genre = "rock"
                else:
                    genre = "folk/acoustic"
            elif "electronic" in top_label or "synth" in top_label:
                genre = "electronic/ambient"
            elif "drum" in top_label or "percussion" in top_label:
                genre = "percussion/rhythmic"
            else:
                # Default based on tempo
                if tempo < 70:
                    genre = "ambient/meditation"
                elif tempo < 100:
                    genre = "easy listening"
                elif tempo < 130:
                    genre = "pop/rock"
                else:
                    genre = "upbeat/dance"
                    
            insights["estimated_style"] = genre
            
        return insights


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze instrumental audio files')
    parser.add_argument('--audio_file', type=str, required=True, 
                       help='Path to the audio file to analyze')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SimpleAudioAnalyzer(use_gpu=not args.no_gpu)
    
    # Run analysis
    results = analyzer.analyze_file(args.audio_file)
    
    # Print insights
    print("\nINSIGHTS SUMMARY:")
    for key, value in results.get('insights', {}).items():
        print(f"- {key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    main()