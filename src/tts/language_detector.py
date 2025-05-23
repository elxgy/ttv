"""
Language detector module for TTS engine.
Identifies whether text is in Brazilian Portuguese or English.
"""

import re
from typing import Tuple, Dict, List, Set
from collections import Counter

class LanguageDetector:
    """
    Detects whether text is in Brazilian Portuguese or English.
    Uses multiple detection strategies for accuracy.
    """
    
    def __init__(self):
        """Initialize the language detector with language signatures."""
        self.en_common_words: Set[str] = {
            'the', 'and', 'is', 'in', 'it', 'you', 'that', 'was', 'for', 'on',
            'are', 'with', 'as', 'this', 'be', 'at', 'have', 'from', 'or', 'had',
            'by', 'but', 'what', 'not', 'they', 'he', 'she', 'we', 'an', 'were',
            'which', 'there', 'can', 'all', 'their', 'has', 'would', 'will', 'been',
            'if', 'more', 'when', 'who', 'also', 'people', 'no', 'my', 'than', 'about'
        }

        self.pt_common_words: Set[str] = {
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
            'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
            'as', 'dos', 'como', 'mas', 'foi', 'ao', 'das', 'tem', 'seu', 'sua',
            'ou', 'ser', 'pelo', 'pela', 'são', 'você', 'quando', 'me', 'já', 'nos',
            'eu', 'também', 'só', 'esse', 'isso', 'ela', 'entre', 'era', 'até', 'ele'
        }
        
        self.pt_character_patterns: List[str] = [
            'ão', 'ç', 'õe', 'á', 'é', 'ê', 'í', 'ó', 'ú', 'ü', 
            'à', 'â', 'ô', 'nh', 'lh', 'ç'
        ]

        try:
            import langdetect
            self.langdetect = langdetect
            self.has_langdetect = True
        except ImportError:
            self.has_langdetect = False
            
        try:
            import langid
            self.langid = langid
            self.has_langid = True
        except ImportError:
            self.has_langid = False
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect whether the text is in English or Brazilian Portuguese.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Tuple[str, float]: A tuple containing the language code ('en' or 'pt-br')
                              and a confidence score (0.0 to 1.0)
        """
        if not text or len(text) < 10:      
            return 'en', 0.5
            
        results = []
        confidence_weights = []
        
        lang_code, confidence = self._detect_by_common_words(text)
        results.append((lang_code, confidence))
        confidence_weights.append(3.0)  
        
        lang_code, confidence = self._detect_by_character_patterns(text)
        results.append((lang_code, confidence))
        confidence_weights.append(2.0)
        
        if self.has_langdetect:
            try:
                detected = self.langdetect.detect(text)
                if detected == 'pt':
                    results.append(('pt-br', 0.9))
                elif detected == 'en':
                    results.append(('en', 0.9))
                confidence_weights.append(2.5)
            except:
                pass
                
        if self.has_langid:
            try:
                lang, confidence = self.langid.classify(text)
                if lang == 'pt':
                    results.append(('pt-br', min(confidence, 1.0)))
                elif lang == 'en':
                    results.append(('en', min(confidence, 1.0)))
                confidence_weights.append(2.5)
            except:
                pass
                
        if not results:
            return 'en', 0.5  
            
        en_score = 0
        pt_score = 0
        total_weight = sum(confidence_weights)
        
        for i, (lang, conf) in enumerate(results):
            weight = confidence_weights[i]
            if lang == 'en':
                en_score += conf * weight
            else:
                pt_score += conf * weight
                
        en_score /= total_weight
        pt_score /= total_weight
        
        if en_score >= pt_score:
            return 'en', en_score
        else:
            return 'pt-br', pt_score
    
    def _detect_by_common_words(self, text: str) -> Tuple[str, float]:
        """
        Detect language by counting common words.
        """
        # Clean and normalize the text
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        if not words:
            return 'en', 0.5
        
        en_matches = sum(1 for word in words if word in self.en_common_words)
        pt_matches = sum(1 for word in words if word in self.pt_common_words)
        
        total_words = len(words)
        en_percentage = en_matches / total_words if total_words > 0 else 0
        pt_percentage = pt_matches / total_words if total_words > 0 else 0
        
        if en_percentage >= pt_percentage:
            confidence = max(0.5, min(en_percentage * 2, 1.0))
            return 'en', confidence
        else:
            confidence = max(0.5, min(pt_percentage * 2, 1.0))
            return 'pt-br', confidence
    
    def _detect_by_character_patterns(self, text: str) -> Tuple[str, float]:
        """
        Detect language by analyzing character patterns.
        """
        text = text.lower()
        
        pt_pattern_count = sum(text.count(pattern) for pattern in self.pt_character_patterns)
        
        text_length = len(text)
        if text_length < 1:
            return 'en', 0.5
            
        pt_pattern_ratio = pt_pattern_count / text_length
        
        if pt_pattern_ratio > 0.01:  
            confidence = min(0.5 + pt_pattern_ratio * 10, 1.0)
            return 'pt-br', confidence
        else:
            confidence = min(0.5 + (0.01 - pt_pattern_ratio) * 10, 1.0)
            return 'en', confidence

def detect_language(text: str) -> str:
    """
    Convenience function to detect language from text.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: Language code ('en' or 'pt-br')
    """
    detector = LanguageDetector()
    lang_code, _ = detector.detect_language(text)
    return lang_code 