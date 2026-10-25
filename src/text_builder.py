"""
src/text_builder.py
Intelligent Word Reconstruction & Fingerspelling Buffer for Indian Sign Language.
Includes temporal hold-stabilization, closed-fist SPACE gesture detection,
Groq LLM semantic reconstruction, and offline dictionary spell correction.
"""

import os
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
import difflib

logger = logging.getLogger("ISLTextBuilder")

COMMON_WORDS = [
    "A", "ABOUT", "AFTER", "AGAIN", "ALL", "ALMOST", "ALSO", "ALWAYS", "AM", "AN", "AND",
    "ANY", "ARE", "AS", "AT", "BE", "BECAUSE", "BEEN", "BEFORE", "BEST", "BETTER", "BIG",
    "BOY", "BUS", "BUT", "BY", "CALL", "CAN", "CAR", "CAT", "COME", "COULD", "DAY", "DEAF",
    "DID", "DO", "DOG", "DOWN", "DRINK", "EAT", "FAMILY", "FATHER", "FEEL", "FIND", "FIRST",
    "FOOD", "FOR", "FRIEND", "FROM", "GET", "GIRL", "GIVE", "GO", "GOOD", "GREAT", "HAPPY",
    "HAVE", "HE", "HELLO", "HELP", "HER", "HERE", "HIM", "HIS", "HOME", "HOT", "HOUSE",
    "HOW", "I", "IF", "IN", "IS", "IT", "ITS", "JUST", "KNOW", "LEARN", "LIKE", "LITTLE",
    "LIVE", "LOOK", "LOVE", "MAKE", "MAN", "MANY", "ME", "MORE", "MORNING", "MOTHER", "MUCH",
    "MY", "NAME", "NEED", "NEW", "NIGHT", "NO", "NOT", "NOW", "OF", "OFF", "OLD", "ON",
    "ONE", "ONLY", "OPEN", "OR", "OTHER", "OUR", "OUT", "OVER", "PEOPLE", "PLEASE", "PUT",
    "READ", "RIGHT", "ROOM", "RUN", "SAD", "SAY", "SEE", "SHE", "SIGN", "SLEEP", "SMALL",
    "SO", "SOME", "SORRY", "STAND", "STOP", "TAKE", "TELL", "THANK", "THANKS", "THAT",
    "THE", "THEIR", "THEM", "THEN", "THERE", "THESE", "THEY", "THING", "THINK", "THIS",
    "TIME", "TO", "TODAY", "TOGETHER", "TOMORROW", "TOO", "TWO", "UP", "US", "USE", "VERY",
    "WANT", "WATER", "WAY", "WE", "WELL", "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHY",
    "WILL", "WITH", "WOMAN", "WORD", "WORK", "WORLD", "WOULD", "WRITE", "YES", "YOU",
    "YOUR", "ZERO"
]


def is_fist_closed(hand_landmarks) -> bool:
    """
    Checks if all four finger tips (index, middle, ring, pinky) are folded
    inward toward their respective base/PIP joints, representing a closed fist (SPACE).
    """
    if hand_landmarks is None:
        return False

    lms = hand_landmarks.landmark
    # In MediaPipe, y points downward (0 is top, 1 is bottom)
    # Folded finger means tip y is below (greater than) the PIP joint y
    index_folded = lms[8].y > lms[6].y
    middle_folded = lms[12].y > lms[10].y
    ring_folded = lms[16].y > lms[14].y
    pinky_folded = lms[20].y > lms[18].y

    return index_folded and middle_folded and ring_folded and pinky_folded


class TextBuilder:
    def __init__(self, confirm_threshold: float = 1.0, space_threshold: float = 1.2):
        self.confirm_threshold = confirm_threshold
        self.space_threshold = space_threshold

        self.current_sign: Optional[str] = None
        self.sign_start_time: Optional[float] = None
        self.fist_start_time: Optional[float] = None
        self.last_confirmed_sign: Optional[str] = None

        self.raw_char_buffer: List[str] = []
        self.sentence_words: List[str] = []
        self.last_reconstructed_word: str = ""

        # Initialize Groq client if API key is present
        self.groq_client = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=api_key)
                logger.info("Groq LLM online client initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

    def update(
        self,
        predicted_sign: Optional[str],
        fist_detected: bool = False
    ) -> Dict[str, Any]:
        """
        Processes streaming sign prediction per frame.
        Returns state dictionary with buffers and event flags.
        """
        now = time.time()
        event = None
        new_word = None

        # 1. Handle Closed Fist / SPACE Gesture
        if fist_detected or predicted_sign == "SPACE":
            if self.fist_start_time is None:
                self.fist_start_time = now
            elif (now - self.fist_start_time) >= self.space_threshold:
                # Commit active word buffer
                if self.raw_char_buffer:
                    new_word = self.commit_word()
                    event = "WORD_COMMITTED"
                self.fist_start_time = None
                self.current_sign = None
                self.last_confirmed_sign = None
        else:
            self.fist_start_time = None

        # 2. Handle Letter / Digit Signs
        if predicted_sign and predicted_sign != "SPACE" and not fist_detected:
            if predicted_sign != self.current_sign:
                self.current_sign = predicted_sign
                self.sign_start_time = now
            else:
                elapsed = now - (self.sign_start_time or now)
                if elapsed >= self.confirm_threshold:
                    if predicted_sign != self.last_confirmed_sign:
                        self.raw_char_buffer.append(predicted_sign)
                        self.last_confirmed_sign = predicted_sign
                        event = "CHAR_ADDED"
                        self.sign_start_time = now
        elif not predicted_sign and not fist_detected:
            # Neutral hand position: reset active sign debounce
            self.current_sign = None
            self.last_confirmed_sign = None
            self.sign_start_time = None

        raw_str = "".join(self.raw_char_buffer)
        sentence_str = " ".join(self.sentence_words)

        return {
            "event": event,
            "confirmed_char": self.last_confirmed_sign,
            "raw_word": raw_str,
            "committed_word": new_word or self.last_reconstructed_word,
            "sentence": sentence_str,
            "word_buffer": list(self.raw_char_buffer),
            "sentence_words": list(self.sentence_words)
        }

    def commit_word(self) -> str:
        """
        Reconstructs the raw characters into a clean English word using
        online Groq LLM with offline dictionary fallback.
        """
        if not self.raw_char_buffer:
            return ""

        raw_str = "".join(self.raw_char_buffer).upper()

        # Step A: Deduplicate consecutive repeats (e.g., 'HHEELLO' -> 'HELO')
        deduped = []
        for ch in raw_str:
            if not deduped or deduped[-1] != ch:
                deduped.append(ch)
        deduped_str = "".join(deduped)

        reconstructed = None

        # Step B: Online Groq LLM Reconstruction
        if self.groq_client:
            try:
                prompt = (
                    f"The user has fingerspelled an Indian Sign Language sequence: '{raw_str}' "
                    f"(condensed: '{deduped_str}'). "
                    f"Infer the single intended uppercase English word. "
                    f"Return ONLY the single uppercase word with no punctuation, no greetings, no explanation."
                )
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )
                word = response.choices[0].message.content.strip().upper()
                # Sanitize response
                word = "".join(c for c in word if c.isalpha())
                if word:
                    reconstructed = word
            except Exception as e:
                logger.warning(f"Groq API call failed: {e}. Falling back to offline dictionary.")

        # Step C: Offline Spell Correction Fallback
        if not reconstructed:
            if deduped_str in COMMON_WORDS:
                reconstructed = deduped_str
            else:
                matches = difflib.get_close_matches(deduped_str, COMMON_WORDS, n=1, cutoff=0.55)
                if matches:
                    reconstructed = matches[0]
                else:
                    reconstructed = deduped_str

        self.last_reconstructed_word = reconstructed
        self.sentence_words.append(reconstructed)
        self.raw_char_buffer = []
        self.last_confirmed_sign = None
        return reconstructed

    def backspace(self) -> None:
        """Removes the last raw character, or the last committed word if buffer is empty."""
        if self.raw_char_buffer:
            self.raw_char_buffer.pop()
        elif self.sentence_words:
            self.sentence_words.pop()

    def clear(self) -> None:
        """Clears all character and sentence buffers."""
        self.raw_char_buffer = []
        self.sentence_words = []
        self.last_reconstructed_word = ""
        self.current_sign = None
        self.last_confirmed_sign = None
