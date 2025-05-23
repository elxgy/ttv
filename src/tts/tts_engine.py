import os
import time
import sys
import tempfile
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import threading
from tqdm import tqdm

from .language_detector import detect_language
try:
    from .silero_tts import SileroTTS
    SILERO_AVAILABLE = True
except ImportError as e:
    print(f"Silero TTS not available: {str(e)}")
    SILERO_AVAILABLE = False

class TTSEngine:
    """Text-to-Speech engine with multiple backend options."""
    
    def __init__(self, engine_type: str = "pyttsx3", max_workers: int = None,
                enable_cache: bool = True, cache_dir: Optional[str] = None,
                cache_size_limit_mb: int = 500, silero_speaker: Optional[str] = None,
                silero_model_dir: Optional[str] = None, silero_device: str = 'cpu'):
        """
        Initialize the TTS engine.
        
        Args:
            engine_type: Type of TTS engine to use ("pyttsx3", "gtts", or "silero")
            max_workers: Maximum number of workers for parallel processing (None = auto)
            enable_cache: Whether to enable audio caching
            cache_dir: Directory to store cached audio files (None = auto)
            cache_size_limit_mb: Maximum cache size in MB
            silero_speaker: Speaker ID for Silero TTS (None = language default)
            silero_model_dir: Directory to store Silero models (None = auto)
            silero_device: Device for Silero inference ('cpu' or 'cuda')
        """
        self.engine_type = engine_type.lower()
        self.engine = None
        self.max_workers = max_workers
        
        self.silero_speaker = silero_speaker
        self.silero_model_dir = silero_model_dir
        self.silero_device = silero_device
        
        self.enable_cache = enable_cache
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser('~'), '.ttv_cache')
        else:
            self.cache_dir = cache_dir
        self.cache_size_limit_mb = cache_size_limit_mb
        self.cache_index_file = os.path.join(self.cache_dir, 'cache_index.json')
        self.cache_index: Dict[str, Dict] = {}
        
        self._progress_lock = threading.Lock()
        self._progress = {"completed": 0, "total": 0, "pbar": None}
        
        if self.enable_cache:
            self._initialize_cache()
        
        self._initialize_engine()
    
    def _initialize_cache(self):
        """Initialize the audio cache directory and index."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            if os.path.exists(self.cache_index_file):
                with open(self.cache_index_file, 'r') as f:
                    self.cache_index = json.load(f)
            else:
                self.cache_index = {}
                self._save_cache_index()
                
            print(f"Audio cache initialized. Using: {self.cache_dir}")
            print(f"Cache entries: {len(self.cache_index)}")
            
            self._manage_cache_size()
        except Exception as e:
            print(f"Warning: Failed to initialize cache: {str(e)}")
            self.enable_cache = False
    
    def _save_cache_index(self):
        """Save the cache index to disk."""
        if not self.enable_cache:
            return
            
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            print(f"Warning: Failed to save cache index: {str(e)}")
    
    def _manage_cache_size(self):
        """Check cache size and remove oldest entries if over limit."""
        if not self.enable_cache:
            return
            
        try:
            total_size = 0
            for entry in self.cache_index.values():
                if 'size' in entry:
                    total_size += entry['size']
            
            total_size_mb = total_size / (1024 * 1024)
            print(f"Current cache size: {total_size_mb:.2f} MB")
            
            if total_size_mb > self.cache_size_limit_mb:
                print(f"Cache over size limit ({self.cache_size_limit_mb} MB). Cleaning up...")
                
                sorted_entries = sorted(
                    self.cache_index.items(),
                    key=lambda x: x[1].get('last_access', 0)
                )
                
                removed_count = 0
                removed_size = 0
                for cache_key, entry in sorted_entries:
                    if total_size_mb <= self.cache_size_limit_mb * 0.9:
                        break
                        
                    cache_path = os.path.join(self.cache_dir, entry.get('filename', ''))
                    if os.path.exists(cache_path):
                        try:
                            os.remove(cache_path)
                            entry_size = entry.get('size', 0)
                            total_size -= entry_size
                            total_size_mb = total_size / (1024 * 1024)
                            removed_size += entry_size
                            removed_count += 1
                            del self.cache_index[cache_key]
                        except:
                            pass
                
                print(f"Removed {removed_count} cache entries ({removed_size/(1024*1024):.2f} MB)")
                self._save_cache_index()
        except Exception as e:
            print(f"Warning: Failed to manage cache size: {str(e)}")
    
    def _get_cache_key(self, text: str, lang: str) -> str:
        """
        Generate a unique cache key for the text-language combination.
        
        Args:
            text: The text to convert to speech
            lang: The language code
            
        Returns:
            str: A unique hash key for caching
        """
        normalized_text = ' '.join(text.split())
        
        cache_string = f"{normalized_text}|{lang}|{self.engine_type}"
        
        cache_key = hashlib.md5(cache_string.encode('utf-8')).hexdigest()
        return cache_key
    
    def _get_cached_audio(self, text: str, lang: str) -> Optional[str]:
        """
        Check if audio for this text-language combination is in cache.
        
        Args:
            text: The text to convert to speech
            lang: The language code
            
        Returns:
            Optional[str]: Path to cached audio file if found, None otherwise
        """
        if not self.enable_cache:
            return None
            
        cache_key = self._get_cache_key(text, lang)
        if cache_key in self.cache_index:
            entry = self.cache_index[cache_key]
            cache_path = os.path.join(self.cache_dir, entry.get('filename', ''))
            
            if os.path.exists(cache_path):
                self.cache_index[cache_key]['last_access'] = time.time()
                self._save_cache_index()
                return cache_path
            else:
                del self.cache_index[cache_key]
                self._save_cache_index()
                
        return None
    
    def _add_to_cache(self, text: str, lang: str, audio_path: str) -> str:
        """
        Add generated audio to the cache.
        
        Args:
            text: The original text
            lang: The language code
            audio_path: Path to the generated audio file
            
        Returns:
            str: Path to the cached audio file
        """
        if not self.enable_cache:
            return audio_path
            
        try:
            cache_key = self._get_cache_key(text, lang)
            
            file_size = os.path.getsize(audio_path)
            
            filename = f"{cache_key}.mp3"
            cache_path = os.path.join(self.cache_dir, filename)
            
            if audio_path != cache_path:
                with open(audio_path, 'rb') as src_file:
                    audio_data = src_file.read()
                
                with open(cache_path, 'wb') as dst_file:
                    dst_file.write(audio_data)
            
                self.cache_index[cache_key] = {
                'filename': filename,
                'text': text[:100],  
                'lang': lang,
                'engine': self.engine_type,
                'size': file_size,
                'created': time.time(),
                'last_access': time.time()
            }
            
            self._save_cache_index()
            
            return cache_path
        except Exception as e:
            print(f"Warning: Failed to add to cache: {str(e)}")
            return audio_path
    
    def _initialize_engine(self):
        """Initialize the selected TTS engine."""
        if self.engine_type == "pyttsx3":
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)  
                self.engine.setProperty('volume', 1.0)  
            except ImportError:
                print("Error: pyttsx3 is not installed. Install it using 'pip install pyttsx3'")
                self.engine = None
        elif self.engine_type == "gtts":
            try:
                import gtts
                self.gtts = gtts.gTTS
            except ImportError:
                print("Error: gtts is not installed. Install it using 'pip install gtts'")
                self.engine = None
        elif self.engine_type == "silero":
            if not SILERO_AVAILABLE:
                print("Error: Silero TTS dependencies not available. Please install pytorch: 'pip install torch torchaudio'")
                self.engine = None
            else:
                try:
                    self.silero = SileroTTS(
                        model_dir=self.silero_model_dir,
                        device=self.silero_device
                    )
                    self.engine = self.silero
                except Exception as e:
                    print(f"Error initializing Silero TTS: {str(e)}")
                    self.engine = None
        else:
            raise ValueError(f"Unsupported engine type: {self.engine_type}")
    
    def set_properties(self, rate: Optional[int] = None, volume: Optional[float] = None, 
                      voice: Optional[str] = None):
        """
        Set TTS engine properties.
        
        Args:
            rate: Speaking rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice: Voice ID to use
        """
        if self.engine_type == "pyttsx3" and self.engine is not None:
            if rate is not None:
                self.engine.setProperty('rate', rate)
            if volume is not None:
                self.engine.setProperty('volume', volume)
            if voice is not None:
                voices = self.engine.getProperty('voices')
                for v in voices:
                    if voice.lower() in v.id.lower():
                        self.engine.setProperty('voice', v.id)
                        break
        elif self.engine_type == "silero" and self.engine is not None:
            if voice is not None:
                print(f"Setting Silero speaker to: {voice}")
                self.silero_speaker = voice
                if voice not in self.silero.available_speakers:
                    print(f"Warning: '{voice}' is not in available speakers: {self.silero.available_speakers}")
                    print(f"Using default speaker: {self.silero.default_speaker}")
                    self.silero_speaker = self.silero.default_speaker
    
    def get_available_voices(self):
        """Get list of available voices for the current engine."""
        if self.engine_type == "pyttsx3" and self.engine is not None:
            voices = []
            for voice in self.engine.getProperty('voices'):
                voices.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages,
                    'gender': voice.gender
                })
            return voices
        elif self.engine_type == "silero" and self.engine is not None:
            voices = []
            for lang in ['en', 'pt-br']:
                for speaker_id in self.silero.get_available_speakers(lang):
                    gender = "female" if speaker_id.endswith("_1") else "male"
                    voices.append({
                        'id': f"{lang}_{speaker_id}",
                        'name': f"Silero {speaker_id} ({lang})",
                        'languages': [lang],
                        'gender': gender
                    })
            return voices
        else:
            return []
    
    def detect_language_from_text(self, text: str) -> str:
        """
        Detect whether the text is in English or Brazilian Portuguese.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: The language code ('en' or 'pt-br')
        """
        return detect_language(text)
    
    def speak(self, text: str, save_to_file: Optional[str] = None, lang: Optional[str] = None):
        """
        Speak the given text or save it to a file.
        
        Args:
            text: Text to speak
            save_to_file: Path to save audio to instead of speaking
            lang: Language code (if None, auto-detected)
        """
        if not text:
            print("No text provided")
            return

        if lang is None:
            lang = self.detect_language_from_text(text)
            print(f"Detected language: {lang}")
        
        if self.engine_type == "silero" and lang != "en":
            print(f"Language {lang} is not directly supported by Silero, using English")
            lang = "en"
            
        cached_file = self._get_cached_audio(text, lang)
        if cached_file is not None:
            if save_to_file:
                try:
                    import shutil
                    shutil.copy2(cached_file, save_to_file)
                    print(f"Audio saved to {save_to_file} (from cache)")
                    return
                except Exception as e:
                    print(f"Error copying from cache: {str(e)}")
            else:
                self._play_audio_file(cached_file)
                return
        
        if save_to_file:
            success = self._generate_audio_file(text, save_to_file, lang)
            if success:
                print(f"Audio saved to {save_to_file}")
                
                if self.enable_cache:
                    self._add_to_cache(text, lang, save_to_file)
            else:
                print("Failed to generate audio")
        else:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                
            try:
                success = self._generate_audio_file(text, temp_path, lang)
                if success:
                    cached_path = None
                    if self.enable_cache:
                        cached_path = self._add_to_cache(text, lang, temp_path)
                        
                    self._play_audio_file(temp_path)
                else:
                    print("Failed to generate audio")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
    
    def _generate_audio_file(self, text: str, output_file: str, lang: Optional[str] = None, verbose: bool = True) -> bool:
        """
        Generate an audio file for the given text.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the audio file
            lang: Language code (if None, auto-detected)
            verbose: Whether to print timing information
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            return False
            
        if lang is None:
            lang = self.detect_language_from_text(text)
        
        if self.engine_type == "silero" and lang != "en":
            print(f"Language {lang} is not directly supported by Silero, using English")
            lang = "en"
            
        if self.engine_type == "pyttsx3" and self.engine is not None:
            if lang == "pt-br":
                voices = self.get_available_voices()
                for voice in voices:
                    if "brazil" in voice['name'].lower() or "portuguese" in voice['name'].lower():
                        self.engine.setProperty('voice', voice['id'])
                        break
                        
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
            return True
            
        elif self.engine_type == "gtts":
            try:
                from gtts import gTTS
                gTTS(text=text, lang=lang).save(output_file)
                return True
            except Exception as e:
                print(f"Error in Google TTS: {str(e)}")
                return False
                
        elif self.engine_type == "silero" and hasattr(self, 'silero'):
            if verbose:
                print(f"Using Silero speaker: {self.silero_speaker}")
            
            return self.silero.generate_audio(
                text=text,
                output_file=output_file,
                lang="en",
                speaker=self.silero_speaker,
                verbose=verbose
            )
            
        else:
            print(f"Unsupported engine type: {self.engine_type}")
            return False
    
    def _generate_audio_worker(self, args: Tuple[str, str, str, int, int]) -> Tuple[bool, str, int]:
        """
        Worker function for parallel audio generation.
        
        Args:
            args: Tuple containing (text, output_file, lang, index, total)
            
        Returns:
            Tuple[bool, str, int]: (success, output_file, index)
        """
        text, output_file, lang, index, total = args
        
        try:
            if lang is None:
                lang = self.detect_language_from_text(text)
            
            if self.engine_type == "silero":
                if lang != "en":
                    print(f"Language {lang} is not directly supported by Silero, using English")
                    lang = "en"
            
            if self.engine_type == "silero" and hasattr(self, 'silero'):
                success = self.silero.generate_audio(
                    text=text,
                    output_file=output_file,
                    lang=lang,
                    speaker=self.silero_speaker,
                    verbose=False
                )
            else:
                success = self._generate_audio_file(text, output_file, lang)
            
            with self._progress_lock:
                self._progress["completed"] += 1
                if self._progress["pbar"] is not None:
                    self._progress["pbar"].update(1)
            
            return (success, output_file, index)
        except Exception as e:
            print(f"\nError in worker thread: {str(e)}")
            return (False, output_file, index)
    
    def _init_progress_bar(self, total: int):
        """Initialize a tqdm progress bar for tracking generation progress."""
        with self._progress_lock:
            self._progress["completed"] = 0
            self._progress["total"] = total
            
            self._progress["pbar"] = tqdm(
                total=total,
                desc="Generating audio",
                unit="file",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
    
    def _close_progress_bar(self):
        """Close the progress bar."""
        with self._progress_lock:
            if self._progress["pbar"] is not None:
                self._progress["pbar"].close()
                self._progress["pbar"] = None
    
    def _play_audio_file(self, file_path: str) -> bool:
        """
        Play audio file using appropriate player for the platform.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        import platform
        system = platform.system()
        
        try:
            if system == "Windows":
                os.system(f'start "{file_path}"')
                return True
            elif system == "Darwin":  
                os.system(f"afplay {file_path}")
                return True
            else:  
                players = ["mpg123", "mpg321", "mplayer", "cvlc", "aplay"]
                
                for player in players:
                    cmd = f"{player} {file_path} > /dev/null 2>&1"
                    exit_code = os.system(cmd)
                    
                    if exit_code == 0:
                        return True
                
                sys.stdout.write("Could not find a suitable audio player.\n")
                sys.stdout.flush()
                return False
                
        except Exception as e:
            print(f"Error playing audio file: {str(e)}")
            return False
            
    def _play_audio_files_continuously(self, file_paths: List[str]) -> bool:
        """
        Play multiple audio files continuously without breaks.
        
        Args:
            file_paths: List of paths to audio files
            
        Returns:
            bool: True if successful, False otherwise
        """
        import platform
        system = platform.system()
        
        try:
            if system == "Windows" or system == "Darwin":
                for file_path in file_paths:
                    if system == "Windows":
                        os.system(f"start /wait {file_path}")
                    elif system == "Darwin":
                        os.system(f"afplay {file_path}")
                return True
            else:  
                players = ["mpg123", "mpg321", "mplayer", "cvlc"]
                
                for player in players:
                    sys.stdout.write(f"Using {player} for continuous playback... ")
                    sys.stdout.flush()
                    
                    files_str = " ".join([f'"{f}"' for f in file_paths])
                    
                    if player in ["mpg123", "mpg321"]:
                        cmd = f"{player} -q {files_str} > /dev/null 2>&1"
                    elif player == "mplayer":
                        cmd = f"{player} -really-quiet {files_str} > /dev/null 2>&1"
                    elif player == "cvlc":
                        cmd = f"{player} --play-and-exit {files_str} > /dev/null 2>&1"
                    else:
                        continue
                        
                    exit_code = os.system(cmd)
                    if exit_code == 0:
                        sys.stdout.write("Completed!\n")
                        sys.stdout.flush()
                        return True
                    else:
                        sys.stdout.write("failed. Trying another player.\n")
                        sys.stdout.flush()
                
                sys.stdout.write("Falling back to sequential playback.\n")
                sys.stdout.flush()
                
                for file_path in file_paths:
                    for player in ["mpg123", "mpg321", "mplayer", "cvlc", "aplay"]:
                        exit_code = os.system(f"{player} {file_path} > /dev/null 2>&1")
                        if exit_code == 0:
                            break
                
                return True
                
        except Exception as e:
            print(f"Error playing audio files continuously: {str(e)}")
            return False
    
    def speak_chunks(self, chunks: List[str], save_to_files: bool = False, 
                    output_dir: str = "output", lang: Optional[str] = None):
        """
        Process and speak a list of text chunks.
        
        Args:
            chunks: List of text chunks to speak
            save_to_files: Whether to save audio files instead of speaking
            output_dir: Directory to save audio files (if save_to_files is True)
            lang: Language code for TTS (if None, auto-detected from content)
        """
        if not chunks:
            print("No text chunks provided")
            return
            
        if lang is None and chunks:
            sample_text = " ".join(chunks[:min(5, len(chunks))])
            lang = self.detect_language_from_text(sample_text)
            print(f"Detected language: {lang}")
            
        if self.engine_type == "silero" and lang != "en":
            print(f"Language {lang} is not directly supported by Silero, using English")
            lang = "en"
            
        self._init_progress_bar(len(chunks))
            
        if save_to_files:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            print(f"Generating {len(chunks)} audio files...")
            
            if self.engine_type == "silero" and self.engine is not None:
                try:
                    file_names = [f"{output_dir}/chunk_{i}.mp3" for i in range(1, len(chunks) + 1)]
                    
                    success_flags = self.silero.generate_audio_batch(
                        texts=chunks,
                        output_files=file_names,
                        lang="en",
                        speaker=self.silero_speaker
                    )
                    
                    with self._progress_lock:
                        if self._progress["pbar"] is not None:
                            self._progress["pbar"].update(len(chunks) - self._progress["completed"])
                    
                    self._close_progress_bar()
                        
                    print(f"Successfully generated {sum(success_flags)} audio files.")
                    return
                except Exception as e:
                    print(f"Batch processing failed: {e}, falling back to standard processing")
            
            tasks = []
            for i, chunk in enumerate(chunks, 1):
                chunk_lang = lang
                if chunk_lang is None:
                    chunk_lang = self.detect_language_from_text(chunk)
                    
                file_name = f"{output_dir}/chunk_{i}.mp3"
                tasks.append((chunk, file_name, chunk_lang, i, len(chunks)))
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self._generate_audio_worker, tasks))
            
            self._close_progress_bar()
                
            print(f"Successfully generated {sum(1 for r in results if r[0])} audio files.")
                
        else:
            print("Generating audio files...")
            start_time = time.time()
            
            temp_dir = tempfile.mkdtemp()
            temp_files = []
            
            try:
                tasks = []
                for i, chunk in enumerate(chunks, 1):
                    chunk_lang = lang
                    if chunk_lang is None:
                        chunk_lang = self.detect_language_from_text(chunk)
                        
                    temp_file = os.path.join(temp_dir, f"chunk_{i}.mp3")
                    tasks.append((chunk, temp_file, chunk_lang, i, len(chunks)))
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(self._generate_audio_worker, tasks))

                self._close_progress_bar()
                
                ordered_files = []
                for success, file_path, index in sorted(results, key=lambda x: x[2]):
                    if success:
                        ordered_files.append(file_path)
                
                end_time = time.time()
                generation_time = end_time - start_time
                print(f"Generated {len(ordered_files)} audio files in {generation_time:.2f} seconds")
                
                if ordered_files:
                    print("\nPlaying all audio continuously...")
                    self._play_audio_files_continuously(ordered_files)
                
            finally:
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
    
    def clear_cache(self):
        """Clear all cached audio files."""
        if not self.enable_cache:
            return
            
        try:
            count = 0
            size = 0
            
            for cache_entry in self.cache_index.values():
                file_path = os.path.join(self.cache_dir, cache_entry.get('filename', ''))
                if os.path.exists(file_path):
                    size += os.path.getsize(file_path)
                    os.remove(file_path)
                    count += 1
            
            self.cache_index = {}
            self._save_cache_index()
            
            print(f"Cleared {count} cached audio files ({size/(1024*1024):.2f} MB)")
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        if not self.enable_cache:
            return {"enabled": False}
            
        total_size = 0
        count = len(self.cache_index)
        
        for entry in self.cache_index.values():
            if 'size' in entry:
                total_size += entry['size']
        
        return {
            "enabled": True,
            "location": self.cache_dir,
            "entries": count,
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024),
            "limit_mb": self.cache_size_limit_mb
        } 