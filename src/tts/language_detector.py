"""
Language detection module for Text-to-Speech applications.
Detects language from text using a variety of methods.
"""

import os
import re
import math
import pickle
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
import threading

FASTTEXT_MODEL_PATH = os.path.join(os.path.expanduser('~'), '.ttv_models', 'lid.176.bin')
FASTTEXT_LOCK = threading.Lock()
FASTTEXT_MODEL = None

ENGLISH_COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for', 'not', 'on', 'with',
    'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
    'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
    'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than',
    'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
    'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
    'day', 'most', 'us', 'very', 'such'
}

PORTUGUESE_COMMON_WORDS = {
    'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os',
    'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem',
    'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também',
    'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos',
    'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 'tinha', 'foram', 'essa',
    'num', 'nem', 'suas', 'meu', 'às', 'minha', 'têm', 'numa', 'pelos', 'elas', 'havia', 'seja',
    'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse',
    'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas',
    'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele',
    'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo'
}

def _normalize_text(text: str) -> str:
    """
    Normalize text for language detection: lowercase, remove punctuation,
    normalize unicode, etc.
    
    Args:
        text: Text to normalize
        
    Returns:
        str: Normalized text
    """
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
    
def _count_common_words(text: str) -> Tuple[int, int]:
    """
    Count the number of common English and Portuguese words in the text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple[int, int]: Count of (English, Portuguese) common words
    """
    words = set(_normalize_text(text).split())
    en_count = len(words.intersection(ENGLISH_COMMON_WORDS))
    pt_count = len(words.intersection(PORTUGUESE_COMMON_WORDS))
    return en_count, pt_count

def _detect_language_heuristic(text: str) -> str:
    """
    Use heuristic methods (common word counting) to detect language.
    
    Args:
        text: Text to analyze
        
    Returns:
        str: Language code ('en' or 'pt-br')
    """
    if not text or len(text.strip()) < 10:
        return 'en'
        
    en_count, pt_count = _count_common_words(text)
    
    if en_count > pt_count:
        return 'en'
    elif pt_count > en_count:
        return 'pt-br'
    else:
        return 'en'

def _detect_language_fasttext(text: str) -> str:
    """
    Use FastText model for language detection.
    
    Args:
        text: Text to analyze
        
    Returns:
        str: Language code ('en' or 'pt-br')
    """
    global FASTTEXT_MODEL
    
    if not text or len(text.strip()) < 10:
        return 'en'
        
    with FASTTEXT_LOCK:
        if FASTTEXT_MODEL is None:
            try:
                import fasttext
                print("Downloading FastText language identification model (first run only)...")
                os.makedirs(os.path.dirname(FASTTEXT_MODEL_PATH), exist_ok=True)
                FASTTEXT_MODEL = fasttext.load_model(FASTTEXT_MODEL_PATH)
            except Exception as e:
                print(f"Error loading FastText model: {str(e)}")
                return _detect_language_heuristic(text)
    
    try:
        normalized_text = _normalize_text(text)
        lang, prob = FASTTEXT_MODEL.predict(normalized_text, k=1)
        lang_code = lang[0].replace('__label__', '')
        
        if lang_code == 'en':
            return 'en'
        elif lang_code in ['pt', 'br']:
            return 'pt-br'
        else:
            return 'en'
    except Exception as e:
        print(f"Error detecting language with FastText: {str(e)}")
        return _detect_language_heuristic(text)

def detect_language(text: str) -> str:
    """
    Detect language from text (English or Brazilian Portuguese).
    
    Args:
        text: Text to analyze
        
    Returns:
        str: Language code ('en' or 'pt-br')
    """
    if not text or len(text.strip()) < 20:
        return 'en'
        
    try:
        return _detect_language_fasttext(text)
    except Exception:
        return _detect_language_heuristic(text) 