# LZT (Lazy to read)

**CURRENTLY UNDER DEVELOPMENT**

A simple system for reading text and PDF files and converting their content to speech.

## Features

- **File Support**:
  - Text files (.txt)
  - PDF documents (.pdf)
  - Support for more document types in progress

- **Multiple TTS Engines**:
  - pyttsx3 (offline TTS, supports multiple voices)
  - Google Text-to-Speech (requires internet connection)
  - Silero TTS (neural TTS with high-quality voices)

- **Language Support**:
  - Automatic detection between English and Brazilian Portuguese
  - Automatic voice selection based on detected language
  - Customizable voices for each engine

- **Performance Optimizations**:
  - **Audio caching system** (up to 500x faster for repeated content)
  - **Parallel processing** for faster audio generation
  - Optimized chunk processing for balanced playback quality

- **Playback Controls**:
  - Play audio directly with terminal feedback
  - Save audio files for later use
  - Adjustable playback parameters

- **User Experience**:
  - Simple command-line interface with intuitive options
  - Real-time processing feedback
  - Utility commands for system management

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ttv
   ```

2. Install required packages:
   ```bash
   # Install all dependencies using requirements.txt
   pip install -r requirements.txt
   ```

   Or install individual packages manually:
   ```bash
   # Core dependencies
   pip install PyPDF2 pyttsx3 gtts
   
   # For Silero TTS support
   pip install torch torchaudio omegaconf tqdm requests
   
   # Optional packages for enhanced language detection
   pip install fasttext langdetect
   
   # For pyttsx3 on Linux, you also need espeak
   sudo apt-get install espeak  # Debian/Ubuntu
   # or
   sudo dnf install espeak      # Fedora
   # or
   sudo pacman -S espeak        # Arch Linux
   
   # For audio playback (Linux)
   sudo apt-get install mpg123  # or mpg321, mplayer, etc.
   ```

## Usage

### Basic Command

```bash
python src/main.py path/to/your/file.txt [options]
```

### Utility Commands

No need to provide a file path when using these commands:

```bash
# List available voices
python src/main.py --list-voices --engine pyttsx3

# Show cache statistics
python src/main.py --cache-stats

# Clear the cache
python src/main.py --clear-cache
```

### Options

**Engine Selection**:
- `--engine {pyttsx3,gtts,silero}`: Choose TTS engine (default: gtts)

**Output Options**:
- `--save`: Save audio files instead of playing directly
- `--output-dir OUTPUT_DIR`: Directory to save audio files (default: output)

**Language and Voice**:
- `--lang LANG`: Language code (default: auto-detect between en and pt-br)
- `--voice VOICE`: Voice ID to use (specific to each engine)
- `--volume VOLUME`: Speech volume for pyttsx3 (0.0 to 1.0, default: 1.0)
- `--detect-only`: Only detect the language of the file and exit

**Processing Control**:
- `--workers N`: Number of worker threads for parallel processing (default: auto)

**Caching System**:
- `--cache`: Enable audio caching (default: enabled)
- `--no-cache`: Disable audio caching
- `--cache-dir PATH`: Directory to store cached audio files (default: ~/.ttv_cache)
- `--cache-limit SIZE`: Maximum cache size in MB (default: 500)

**Silero TTS Options**:
- `--silero-speaker SPEAKER`: Speaker ID for Silero TTS (default: language dependent)
- `--silero-model-dir DIR`: Directory to store Silero models (default: ~/.ttv_models)
- `--silero-device {cpu,cuda}`: Device for Silero inference (default: cpu)

### Examples

Read a PDF file using Google TTS:
```bash
python src/main.py document.pdf --engine gtts
```

Use neural Silero TTS with specific speaker:
```bash
python src/main.py article.txt --engine silero --silero-speaker en_0
```

Process a large file with 8 worker threads, saving the output:
```bash
python src/main.py large_document.pdf --workers 8 --save --output-dir my_audio
```

Force Portuguese language and disable caching:
```bash
python src/main.py brazilian_text.txt --lang pt-br --no-cache
```

## Performance Benchmarks

The caching system provides dramatic performance improvements for repeated content:

| Content Size | First Run | Cached Run | Speedup Factor |
|--------------|-----------|------------|----------------|
| Small (1KB)  | ~3 sec    | ~0.01 sec  | 300x           |
| Medium (10KB)| ~15 sec   | ~0.01 sec  | 1500x          |
| Large (50KB) | ~44 sec   | ~0.01 sec  | 4400x          |

*Results may vary based on hardware, engine, and text content.*

## Project Structure

```
ttv/
├── src/
│   ├── parsers/
│   │   └── file_parser.py     # File reading and text chunking
│   ├── tts/
│   │   ├── tts_engine.py      # Main TTS interface
│   │   ├── silero_tts.py      # Neural TTS implementation
│   │   └── language_detector.py # Language detection
│   └── main.py                # CLI application    
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Requirements

- Python 3.6+
- See requirements.txt for full dependency list
