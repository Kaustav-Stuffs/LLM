import re
from typing import List, Tuple
from difflib import SequenceMatcher

class GreetingHandler:
    """Handles detection and processing of greetings and closings with fuzzy matching for typos."""
    
    def __init__(self):
        # Core greeting patterns (exact matches)
        self.greeting_patterns = [
            r"^h+[iae]*l*o*w*\b",  # hi, hello, helo, hiiii, etc.
            r"^h+[eay]+y*\b",      # hey, hay, heyy, etc.
            r"^good\s*(morning|afternoon|evening|day)\b",
            r"^greetings?\b",
            r"^welcome\b",
            r"^howdy\b",
            r"^yo+\b",
            r"^sup\b",
            r"^what'?s?\s*up\b",
            r"^salutations?\b"
        ]
        
        # Common greeting words for fuzzy matching
        self.greeting_words = [
            "hi", "hello", "hey", "helo", "hallo", "hiya", "howdy", "yo", "sup",
            "good morning", "good afternoon", "good evening", "good day",
            "greetings", "welcome", "salutations"
        ]
        
        # Closing patterns
        self.closing_patterns = [
            r"\bth?ank\s*y?ou?\b", r"\bthanks?\b", r"\bthx\b", r"\bty\b", 
            r"\bthnks?\b", r"\bthank\s*u\b",
            r"\bappreciate\s*it\b", r"\bthanx\b",
            r"\bgot\s*it\b", r"\bi\s*got\s*it\b", r"\bunderstood\b", r"\bgot\s*this\b",
            r"\bbye+\b", r"\bgood\s*bye\b", r"\bsee\s*y?ou?\b", r"\bsee\s*ya\b",
            r"\bcatch\s*y?ou?\s*later\b", r"\blater\b", r"\btake\s*care\b",
            r"\bok\b", r"\bokay\b", r"\bpeace\b", r"\bcheers\b", r"\bciao\b",
            r"\bdone\b", r"\bfinished\b", r"\bbye\s*now\b"
        ]
        
        # Common closing words for fuzzy matching
        self.closing_words = [
            "thank you", "thanks", "thx", "ty", "bye", "goodbye", "see you", 
            "see ya", "later", "ok", "okay", "done", "finished", "peace", 
            "cheers", "ciao", "take care"
        ]
    
    def _fuzzy_match_word(self, word: str, target_words: List[str], threshold: float = 0.7) -> bool:
        """Check if a word fuzzy matches any target words."""
        word = word.lower().strip()
        if len(word) < 2:
            return False
            
        for target in target_words:
            similarity = SequenceMatcher(None, word, target.lower()).ratio()
            if similarity >= threshold:
                return True
        return False
    
    def _extract_first_words(self, text: str, max_words: int = 3) -> str:
        """Extract first few words from text for greeting detection."""
        words = re.findall(r'\b\w+\b', text.lower())
        return ' '.join(words[:max_words])
    
    def is_greeting(self, text: str) -> bool:
        """Check if text contains a greeting."""
        text_clean = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
        
        # Check exact patterns first
        for pattern in self.greeting_patterns:
            if re.search(pattern, text_clean):
                return True
        
        # Check fuzzy matching for first few words
        first_words = self._extract_first_words(text_clean, 2)
        if self._fuzzy_match_word(first_words, self.greeting_words, threshold=0.6):
            return True
        
        # Check individual words
        words = text_clean.split()
        if words and self._fuzzy_match_word(words[0], ["hi", "hello", "hey", "yo"], threshold=0.6):
            return True
            
        return False
    
    def is_greeting_only(self, text: str) -> bool:
        """Check if text is only a greeting (with minimal additional words)."""
        text_clean = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
        words = text_clean.split()
        
        if len(words) == 0:
            return False
        
        # Single word greetings
        if len(words) == 1:
            return self._fuzzy_match_word(words[0], ["hi", "hello", "hey", "yo", "sup"], threshold=0.6)
        
        # Two word greetings
        if len(words) == 2:
            first_two = ' '.join(words)
            if self._fuzzy_match_word(first_two, ["good morning", "good afternoon", "good evening"], threshold=0.7):
                return True
            if self._fuzzy_match_word(words[0], ["hi", "hello", "hey"], threshold=0.6):
                # Allow small filler words after greeting
                filler_words = {"there", "everyone", "all", "guys", "folks", "friend", "buddy"}
                return words[1] in filler_words or self._fuzzy_match_word(words[1], list(filler_words), threshold=0.6)
        
        # Three words maximum for greeting-only
        if len(words) <= 3:
            if self._fuzzy_match_word(words[0], ["hi", "hello", "hey"], threshold=0.6):
                # Check if remaining words are just filler
                remaining = ' '.join(words[1:])
                filler_phrases = ["there", "everyone", "all", "guys", "folks", "friend", "buddy", "how are", "whats up"]
                return self._fuzzy_match_word(remaining, filler_phrases, threshold=0.5)
        
        return False
    
    def is_closing(self, text: str) -> bool:
        """Check if text contains a closing."""
        text_clean = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
        
        # Check exact patterns
        for pattern in self.closing_patterns:
            if re.search(pattern, text_clean):
                return True
        
        # Check fuzzy matching
        words = text_clean.split()
        for word in words:
            if self._fuzzy_match_word(word, self.closing_words, threshold=0.6):
                return True
        
        # Check multi-word closings
        if len(words) >= 2:
            for i in range(len(words) - 1):
                two_words = ' '.join(words[i:i+2])
                if self._fuzzy_match_word(two_words, self.closing_words, threshold=0.6):
                    return True
        
        return False
    
    def is_closing_only(self, text: str) -> bool:
        """Check if text is only a closing."""
        text_clean = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', ' ', text_clean)
        words = text_clean.split()
        
        if len(words) == 0:
            return False
        
        # Single word closings
        if len(words) == 1:
            return self._fuzzy_match_word(words[0], ["thanks", "bye", "ok", "done"], threshold=0.6)
        
        # Two word closings
        if len(words) == 2:
            two_words = ' '.join(words)
            return self._fuzzy_match_word(two_words, ["thank you", "good bye", "see ya"], threshold=0.6)
        
        # Allow up to 3 words for closings
        if len(words) <= 3:
            full_text = ' '.join(words)
            return self._fuzzy_match_word(full_text, ["thank you", "see you later", "take care"], threshold=0.5)
        
        return False
    
    def get_greeting_response(self) -> str:
        """Get a standard greeting response."""
        return "Hello! How can I help you with your documents today?"
    
    def get_closing_response(self) -> str:
        """Get a standard closing response."""
        return "Goodbye! Feel free to come back if you have more questions about your documents."
    
    def should_add_greeting_prefix(self, user_input: str, response: str) -> str:
        """Add greeting prefix to response if user started with greeting but asked a question."""
        if self.is_greeting(user_input) and not self.is_greeting_only(user_input):
            if not re.match(r"^\s*hello[!.,\s]*", response, re.IGNORECASE):
                return f"Hello! {response}"
        return response