# LZT (Lazy to read)

**CURRENTLY UNDER DEVELOPMENT**

A simple system for reading text and PDF files and converting their content to speech.

## Features

- Read text files (.txt) and PDF files (.pdf)
- Process text content for TTS by splitting into manageable chunks
- Multiple TTS engine options:
  - pyttsx3 (offline TTS, supports multiple voices)
  - Google Text-to-Speech (requires internet connection)
- **Parallel processing** for faster audio generation
- **Audio caching** for improved performance on repeated text
- Auto-detection of English and Brazilian Portuguese language
- Play audio directly with terminal feedback or save to files
- Command-line interface for easy usage

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
   
   # Optional packages for enhanced language detection
   pip install langdetect langid
   
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

### Simple Example

Run the example script for a quick demo:

```bash
python src/example.py
```

This will read the sample file and convert it to speech using Google TTS, playing the audio directly with terminal feedback.

### Command-Line Interface

The system provides a command-line interface for more customization:

```bash
python src/main.py path/to/your/file.txt [options]
```

#### Options:

**File Processing Options:**
- `--engine {pyttsx3,gtts}`: Choose TTS engine (default: gtts)
- `--save`: Save audio files instead of playing directly
- `--output-dir OUTPUT_DIR`: Directory to save audio files (default: output)
- `--lang LANG`: Language code for TTS (default: auto-detect between en and pt-br)
- `--volume VOLUME`: Speech volume for pyttsx3 (default: 1.0)
- `--voice VOICE`: Voice to use for pyttsx3
- `--workers N`: Number of worker threads for parallel processing (default: auto)
- `--detect-only`: Only detect the language of the file and exit

**Caching Options:**
- `--cache`: Enable audio caching (default: enabled)
- `--no-cache`: Disable audio caching
- `--cache-dir PATH`: Directory to store cached audio files (default: ~/.ttv_cache)
- `--cache-limit SIZE`: Maximum cache size in MB (default: 500)

**Utility Commands:**
- `--list-voices`: List available voices for pyttsx3 and exit
- `--clear-cache`: Clear the audio cache and exit
- `--cache-stats`: Show cache statistics and exit

#### Examples:

List available voices:
```bash
python src/main.py --list-voices --engine pyttsx3
```

Read a file using Google TTS in Spanish:
```bash
python src/main.py path/to/file.pdf --engine gtts --lang es
```

Save audio files instead of playing directly:
```bash
python src/main.py path/to/file.txt --save --output-dir my_audio_files
```

Speed up processing with more worker threads:
```bash
python src/main.py path/to/large_file.pdf --workers 8
```

Disable caching:
```bash
python src/main.py path/to/file.txt --no-cache
```

Show cache statistics:
```bash
python src/main.py --cache-stats
```

Clear the cache:
```bash
python src/main.py --clear-cache
```

## Performance Optimization

The system uses parallel processing to generate audio files simultaneously, which can significantly speed up the conversion process, especially for large documents. You can adjust the number of worker threads with the `--workers` option to optimize performance for your specific hardware.

Audio caching is enabled by default, which stores previously generated audio files. This dramatically improves performance when processing the same text multiple times.

## Terminal Feedback

When playing audio directly, the program provides real-time feedback in the terminal:
- Shows which chunk is being played
- Displays the text content being read
- Indicates which audio player is being used (on Linux)
- Shows progress as each chunk is processed
- Shows timing information for performance monitoring

## Project Structure

```
ttv/
├── src/
│   ├── parsers/
│   │   └──  file_parser.py  
│   ├── tts/
│   │   ├── tts_engine.py
│   │   └── language_detector.py
│   ├── main.py             
│   └── example.py          
└── README.md               
```

## Requirements

- Python 3.6+
- pyttsx3 (for offline TTS)
  - On Linux: espeak system package
- PyPDF2 (for PDF support)
- gtts (for Google TTS)
- Audio player: mpg123, mpg321, mplayer, or similar (for Linux)
