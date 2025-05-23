#!/usr/bin/env python3
import os
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.file_parser import FileParser
from src.tts.tts_engine import TTSEngine

def main():
    """Main function to parse text files and convert to speech."""
    parser = argparse.ArgumentParser(description='Text-to-Speech File Reader')
    
    utility_group = parser.add_argument_group('Utility Commands')
    utility_group.add_argument('--list-voices', action='store_true',
                        help='List available voices and exit (for pyttsx3 only)')
    utility_group.add_argument('--clear-cache', action='store_true',
                        help='Clear the audio cache and exit')
    utility_group.add_argument('--cache-stats', action='store_true',
                        help='Show cache statistics and exit')
    
    parser.add_argument('file_path', nargs='?',
                        help='Path to the text file to read (optional for utility commands)')
    parser.add_argument('--engine', choices=['pyttsx3', 'gtts'], default='gtts',
                        help='TTS engine to use (pyttsx3 for offline, gtts for Google TTS)')
    parser.add_argument('--save', action='store_true',
                        help='Save audio files instead of playing directly')
    parser.add_argument('--output-dir', default='output',
                        help='Directory to save audio files (if --save is specified)')
    parser.add_argument('--lang', default=None,
                        help='Language code for TTS (default: auto-detect between en and pt-br)')
    parser.add_argument('--volume', type=float, default=1.0,
                        help='Speech volume (for pyttsx3 only, 0.0 to 1.0)')
    parser.add_argument('--voice', default=None,
                        help='Voice to use (for pyttsx3 only)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker threads for parallel processing (default: auto)')
    parser.add_argument('--detect-only', action='store_true',
                        help='Only detect the language of the file and exit')
    
    cache_group = parser.add_argument_group('Caching Options')
    cache_group.add_argument('--cache', action='store_true', default=True,
                        help='Enable audio caching (default: enabled)')
    cache_group.add_argument('--no-cache', action='store_true',
                        help='Disable audio caching')
    cache_group.add_argument('--cache-dir', default=None,
                        help='Directory to store cached audio files (default: ~/.ttv_cache)')
    cache_group.add_argument('--cache-limit', type=int, default=500,
                        help='Maximum cache size in MB (default: 500)')
    
    args = parser.parse_args()
    
    utility_mode = args.list_voices or args.clear_cache or args.cache_stats
    
    if not utility_mode and not args.file_path:
        parser.error("file_path is required unless running a utility command")
    
    if args.no_cache:
        args.cache = False
    
    tts = TTSEngine(engine_type=args.engine, max_workers=args.workers,
                   enable_cache=args.cache, cache_dir=args.cache_dir,
                   cache_size_limit_mb=args.cache_limit)
    
    if args.clear_cache:
        tts.clear_cache()
        print("Cache cleared.")
        return
        
    if args.cache_stats:
        stats = tts.get_cache_stats()
        print("Cache statistics:")
        print("-" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("-" * 50)
        return
    
    if args.list_voices:
        if args.engine != 'pyttsx3':
            print("Voice listing is only supported for pyttsx3 engine")
            return
            
        voices = tts.get_available_voices()
        if not voices:
            print("No voices available or pyttsx3 not initialized properly")
            return
            
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"{i+1}. ID: {voice['id']}")
            print(f"   Name: {voice['name']}")
            print(f"   Languages: {voice['languages']}")
            print(f"   Gender: {voice['gender']}")
            print()
        return
    
    if args.engine == 'pyttsx3':
        tts.set_properties(rate=150, volume=args.volume, voice=args.voice)
    
    file_parser = FileParser()
    
    print(f"Reading file: {args.file_path}")
    content = file_parser.read_file(args.file_path)
    
    if content is None:
        print("Failed to read the file")
        return
    
    print("Processing content for TTS...")
    chunks = file_parser.process_text_for_tts(content)
    
    print(f"Found {len(chunks)} text chunks")
    
    if args.lang is None:
        print("Detecting document language...")
        sample_text = " ".join(chunks[:min(10, len(chunks))])
        args.lang = tts.detect_language_from_text(sample_text)
        print(f"Detected language: {args.lang}")
    
    if args.detect_only:
        return
    
    if args.save:
        print(f"Saving audio files to directory: {args.output_dir}")
        tts.speak_chunks(chunks, save_to_files=True, output_dir=args.output_dir, 
                        lang=args.lang)
        print(f"Audio files saved to {args.output_dir}")
    else:
        print("Playing audio directly...")
        tts.speak_chunks(chunks, lang=args.lang)
    
    print("Done!")

if __name__ == '__main__':
    main() 