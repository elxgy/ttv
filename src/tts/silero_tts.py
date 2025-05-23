"""
Optimized Silero TTS implementation, focused on English with maximum performance.
This module provides a high-quality neural TTS engine using Silero models.
"""

import os
import sys
import torch
import time
import hashlib
import tempfile
import threading
import requests
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm


class SileroTTS:
    """Optimized neural text-to-speech engine using Silero models."""
    
    _model_load_lock = threading.Lock()
    _models_cache = {}
    _download_progress_shown = False
    _MODEL_URL = "https://models.silero.ai/models/tts/en/v3_en.pt"
    
    def __init__(self, model_dir: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize Silero TTS engine optimized for English.
        
        Args:
            model_dir: Directory to store downloaded models (None = default location)
            device: Device to run models on ('cpu' or 'cuda')
        """
        if model_dir is None:
            self.model_dir = os.path.join(os.path.expanduser('~'), '.ttv_models')
        else:
            self.model_dir = model_dir
            
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.model_marker_file = os.path.join(self.model_dir, "silero_model_downloaded.txt")
        self.default_speaker = 'en_0'
        self.available_speakers = ['en_0', 'en_1', 'en_2']
        self.sample_rate = 48000
        
        threading.Thread(target=self._preload_model, daemon=True).start()
        
        print(f"Initialized Silero TTS engine (device: {self.device}, optimized for English)")
    
    def _preload_model(self):
        """Preload the model in background thread to speed up first use."""
        try:
            self._load_model()
        except Exception as e:
            print(f"Model preloading in background failed: {e}")
    
    def _get_cached_model(self):
        """Get model from global cache if available."""
        cache_key = f"en_{self.device}"
        return self._models_cache.get(cache_key)
    
    def _set_cached_model(self, model):
        """Store model in global cache for reuse across instances."""
        cache_key = f"en_{self.device}"
        self._models_cache[cache_key] = model
    
    def _show_download_progress(self):
        """Show a fake progress bar for torch.hub downloads."""
        try:
            SileroTTS._download_progress_shown = True
            
            print("Downloading Silero TTS model (one-time setup, ~105MB)...")
            progress_bar = tqdm(
                desc="Downloading model",
                total=100,
                unit='%',
            )
            
            for i in range(0, 101, 5):
                time.sleep(0.5)
                progress_bar.update(5)
                
            progress_bar.close()
            
            with open(self.model_marker_file, 'w') as f:
                f.write(f"Model downloaded via torch.hub on {time.ctime()}")
                
        except Exception as e:
            print(f"Error showing progress: {e}")
        finally:
            SileroTTS._download_progress_shown = False
    
    def _download_model(self) -> bool:
        """
        Download the model directly from Silero's servers.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_path = os.path.join(self.model_dir, "silero_en_v3.pt")
            
            if os.path.exists(model_path):
                print(f"Model already exists at {model_path}")
                return model_path
                
            url = self._MODEL_URL
            print(f"Downloading Silero TTS model from {url}")
            
            SileroTTS._download_progress_shown = True
            
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            progress_bar = tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc="Downloading model"
            )
            
            with open(model_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            
            progress_bar.close()
            
            if total_size != 0 and progress_bar.n != total_size:
                print("Error: Download incomplete")
                return False
                
            print(f"Model downloaded to {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            return False
        finally:
            SileroTTS._download_progress_shown = False

    def _load_model(self) -> bool:
        """
        Load Silero model for English.
        
        Returns:
            bool: True if successful, False otherwise
        """
        model = self._get_cached_model()
        if model is not None:
            return True
            
        with self._model_load_lock:
            model = self._get_cached_model()
            if model is not None:
                return True
                
            try:
                model_downloaded = os.path.exists(self.model_marker_file)
                progress_thread = None
                
                if not model_downloaded and not SileroTTS._download_progress_shown:
                    progress_thread = threading.Thread(
                        target=self._show_download_progress,
                        daemon=True
                    )
                    progress_thread.start()
                
                print("Loading Silero TTS model...")
                
                try:
                    hub_result = torch.hub.load(
                        repo_or_dir='snakers4/silero-models',
                        model='silero_tts',
                        language='en',
                        speaker='v3_en',
                        trust_repo=True
                    )
                    
                    model = None
                    apply_tts = None
                    
                    if isinstance(hub_result, tuple):
                        if len(hub_result) == 5:
                            model, _, _, _, apply_tts = hub_result
                        elif len(hub_result) == 2:
                            model = hub_result[0]
                            apply_tts = lambda text, speaker, sample_rate: model.apply_tts(text, speaker=speaker, sample_rate=sample_rate)
                    else:
                        model = hub_result
                        apply_tts = lambda text, speaker, sample_rate: model.apply_tts(text, speaker=speaker, sample_rate=sample_rate)
                    
                    if model is None:
                        print("Error: Could not extract model from torch.hub result")
                        return False
                        
                    model.to(self.device)
                    
                    self._set_cached_model((model, apply_tts))
                    
                    if not os.path.exists(self.model_marker_file):
                        with open(self.model_marker_file, 'w') as f:
                            f.write(f"Model downloaded via torch.hub on {time.ctime()}")
                    
                    print("Silero TTS model loaded successfully")
                    return True
                    
                except Exception as e:
                    print(f"Error loading model via torch.hub: {str(e)}")
                    return False
                
            except Exception as e:
                print(f"Error loading Silero model: {str(e)}")
                return False
    
    def _clear_torch_hub_cache(self):
        """Clear torch hub cache to fix potential corruption issues."""
        try:
            import shutil
            cache_dir = os.path.expanduser("~/.cache/torch/hub")
            if os.path.exists(cache_dir):
                print(f"Clearing torch hub cache at {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error clearing torch hub cache: {str(e)}")
    
    def get_available_speakers(self, lang: str = 'en') -> List[str]:
        """
        Get list of available speakers.
        
        Args:
            lang: Language code (only 'en' supported in this optimized version)
            
        Returns:
            List[str]: List of speaker IDs
        """
        if lang != 'en':
            print(f"Warning: Language {lang} is not supported, using English")
            
        return self.available_speakers
    
    def generate_audio(self, text: str, output_file: str, lang: str = 'en', 
                      speaker: Optional[str] = None, verbose: bool = True) -> bool:
        """
        Generate audio file for the given text.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file
            lang: Language code (only 'en' supported in this optimized version)
            speaker: Speaker ID (if None, uses default)
            verbose: Whether to print timing information
            
        Returns:
            bool: True if successful, False otherwise
        """
        if lang != 'en':
            print(f"Warning: Language {lang} is not supported, using English")
            lang = 'en'
        
        try:
            if not self._load_model():
                return False
                
            if speaker is None:
                speaker = self.default_speaker
                if verbose:
                    print(f"No speaker specified, using default: {speaker}")
            elif speaker not in self.available_speakers:
                if verbose:
                    print(f"Warning: Speaker '{speaker}' not in available speakers: {self.available_speakers}")
                    print(f"Using default speaker: {self.default_speaker}")
                speaker = self.default_speaker
            else:
                if verbose:
                    print(f"Using specified speaker: {speaker}")
            
            cached_model = self._get_cached_model()
            if cached_model is None:
                print("Error: Model not loaded")
                return False
                
            model, apply_tts = cached_model
            
            start_time = time.time()
            audio = apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=self.sample_rate
            )
            
            try:
                import torchaudio
            except ImportError:
                print("Error: torchaudio not installed. Install with 'pip install torchaudio'")
                return False
            
            if output_file.lower().endswith('.mp3'):
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
                torchaudio.save(temp_wav, audio.unsqueeze(0), self.sample_rate)
                
                try:
                    import subprocess
                    subprocess.call(['ffmpeg', '-i', temp_wav, '-codec:a', 'libmp3lame', '-qscale:a', '2', output_file, '-y', '-loglevel', 'quiet'])
                    os.remove(temp_wav)
                except Exception as e:
                    print(f"MP3 conversion failed: {e}, falling back to WAV format")
                    output_file = output_file.replace('.mp3', '.wav')
                    torchaudio.save(output_file, audio.unsqueeze(0), self.sample_rate)
            else:
                torchaudio.save(output_file, audio.unsqueeze(0), self.sample_rate)
            
            end_time = time.time()
            audio_length = len(audio) / self.sample_rate
            
            if verbose and end_time - start_time > 1.0:
                print(f"Audio generated in {end_time - start_time:.2f}s (RT factor: {audio_length/(end_time-start_time):.2f}x)")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"Error generating audio with Silero: {str(e)}")
            return False
    
    def generate_audio_batch(self, texts: List[str], output_files: List[str], 
                           lang: str = 'en', speaker: Optional[str] = None) -> List[bool]:
        """
        Generate multiple audio files in batch mode.
        
        Args:
            texts: List of texts to convert to speech
            output_files: List of output file paths
            lang: Language code (only 'en' supported)
            speaker: Speaker ID (if None, uses default)
            
        Returns:
            List[bool]: List of success flags for each file
        """
        if not texts or len(texts) != len(output_files):
            return [False] * len(texts) if texts else []
        
        if not self._load_model():
            return [False] * len(texts)
            
        if speaker is None:
            speaker = self.default_speaker
            print(f"No speaker specified, using default: {speaker}")
        elif speaker not in self.available_speakers:
            print(f"Warning: Speaker '{speaker}' not in available speakers: {self.available_speakers}")
            print(f"Using default speaker: {self.default_speaker}")
            speaker = self.default_speaker
        else:
            print(f"Using specified speaker: {speaker}")
        
        cached_model = self._get_cached_model()
        if cached_model is None:
            print("Error: Model not loaded")
            return [False] * len(texts)
            
        model, apply_tts = cached_model
        
        try:
            import torchaudio
        except ImportError:
            print("Error: torchaudio not installed. Install with 'pip install torchaudio'")
            return [False] * len(texts)
        
        progress_bar = tqdm(
            total=len(texts),
            desc="Generating audio",
            unit="file",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        results = []
        try:
            for text, output_file in zip(texts, output_files):
                try:
                    audio = apply_tts(
                        text=text,
                        speaker=speaker,
                        sample_rate=self.sample_rate
                    )
                    
                    if output_file.lower().endswith('.mp3'):
                        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
                        torchaudio.save(temp_wav, audio.unsqueeze(0), self.sample_rate)
                        
                        try:
                            import subprocess
                            subprocess.call(['ffmpeg', '-i', temp_wav, '-codec:a', 'libmp3lame', '-qscale:a', '2', output_file, '-y', '-loglevel', 'quiet'])
                            os.remove(temp_wav)
                        except Exception as e:
                            output_file = output_file.replace('.mp3', '.wav')
                            torchaudio.save(output_file, audio.unsqueeze(0), self.sample_rate)
                    else:
                        torchaudio.save(output_file, audio.unsqueeze(0), self.sample_rate)
                    
                    results.append(True)
                except Exception as e:
                    print(f"Error generating audio: {str(e)}")
                    results.append(False)
                finally:
                    progress_bar.update(1)
        finally:
            progress_bar.close()
            
        return results
    
    def cleanup(self):
        """Release resources and models."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 