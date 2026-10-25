"""
src/word_builder.py
Dedicated English Word Construction & Fingerspelling State Machine.
Strictly in-scope:
  - English text only
  - Raw letter preservation (no cloud LLM substitution)
  - Release / Re-arm transition logic for repeated letters (e.g. 'L' in 'HELLO')
  - Manual and gesture-based word boundary delimiter controls
  - Completed word list accumulator
"""

import time
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger("ISLWordBuilder")


class EnglishWordBuilder:
    """
    Manages stateful assembly of confirmed fingerspelled letters into English words and sentences.
    Preserves raw recognized letters to ensure model classification errors remain fully measurable.
    """

    def __init__(
        self,
        confirm_duration: float = 1.0,
        rearm_duration: float = 0.25,
        space_gesture_duration: float = 1.2
    ):
        self.confirm_duration = confirm_duration
        self.rearm_duration = rearm_duration
        self.space_gesture_duration = space_gesture_duration

        # Active character tracking
        self.candidate_char: Optional[str] = None
        self.candidate_start_time: Optional[float] = None
        self.last_confirmed_char: Optional[str] = None

        # Re-arm state for repeated letters (e.g., L-L in HELLO)
        self.is_armed: bool = True
        self.neutral_start_time: Optional[float] = None

        # Gesture space tracking
        self.fist_start_time: Optional[float] = None

        # Text buffers
        self.current_word_letters: List[str] = []
        self.completed_words: List[str] = []

    def feed_frame(
        self,
        predicted_char: Optional[str],
        confidence: float,
        is_reliable: bool,
        is_fist_closed: bool = False
    ) -> Dict[str, Any]:
        """
        Feeds the real-time classification result from the current frame into the state machine.
        
        Args:
            predicted_char: Predicted sign label (e.g., 'A', 'B', '1', etc.) or None
            confidence: Float confidence score [0.0, 1.0]
            is_reliable: Whether confidence passed threshold and majority voting
            is_fist_closed: Whether closed fist (SPACE) gesture is active
            
        Returns:
            Dictionary with current word, completed words, and event notifications.
        """
        now = time.time()
        event = None
        confirmed_char = None
        hold_progress = 0.0

        # 1. Closed Fist SPACE Gesture Check
        if is_fist_closed:
            if self.fist_start_time is None:
                self.fist_start_time = now
            elif (now - self.fist_start_time) >= self.space_gesture_duration:
                if self.current_word_letters:
                    event = self.finalize_word()
                self.fist_start_time = None
                self.candidate_char = None
                self.candidate_start_time = None
        else:
            self.fist_start_time = None

        # 2. Re-Arm Transition Logic & Hand Leaving Frame Reset
        if not is_reliable or predicted_char is None or is_fist_closed:
            # Immediately clear uncommitted candidate sign
            self.candidate_char = None
            self.candidate_start_time = None

            if self.neutral_start_time is None:
                self.neutral_start_time = now
            elif (now - self.neutral_start_time) >= self.rearm_duration:
                self.is_armed = True
                self.last_confirmed_char = None
        else:
            # If we were previously neutral and sufficient time elapsed, re-arm!
            if self.neutral_start_time is not None:
                if (now - self.neutral_start_time) >= self.rearm_duration:
                    self.is_armed = True
                    self.last_confirmed_char = None
                self.neutral_start_time = None

            # If user transitions to a DIFFERENT sign from the last confirmed sign: re-arm immediately!
            if predicted_char != self.last_confirmed_char:
                self.is_armed = True

            # 3. Candidate Letter Stability Accumulation
            if self.is_armed:
                if predicted_char != self.candidate_char:
                    # New candidate sign detected: restart hold timer
                    self.candidate_char = predicted_char
                    self.candidate_start_time = now
                else:
                    # Same candidate sign held continuously
                    elapsed = now - (self.candidate_start_time or now)
                    hold_progress = min(1.0, elapsed / max(0.001, self.confirm_duration))

                    if elapsed >= self.confirm_duration:
                        # Confirm and commit letter
                        self.current_word_letters.append(predicted_char)
                        self.last_confirmed_char = predicted_char
                        confirmed_char = predicted_char
                        event = "LETTER_ADDED"

                        # Disarm until hand relaxes or transitions to a different sign
                        self.is_armed = False
                        self.candidate_char = None
                        self.candidate_start_time = None

        current_word_str = "".join(self.current_word_letters)
        sentence_str = " ".join(self.completed_words)

        return {
            "event": event,
            "confirmed_char": confirmed_char,
            "current_word": current_word_str,
            "completed_words": list(self.completed_words),
            "full_text": f"{sentence_str} {current_word_str}".strip() if current_word_str else sentence_str,
            "hold_progress": round(hold_progress, 2),
            "is_armed": self.is_armed,
            "fist_detected": is_fist_closed
        }

    def finalize_word(self) -> str:
        """Finalizes the current word buffer into the completed words list."""
        if not self.current_word_letters:
            return "WORD_EMPTY"

        word = "".join(self.current_word_letters)
        self.completed_words.append(word)
        self.current_word_letters = []
        self.is_armed = True
        self.candidate_char = None
        self.candidate_start_time = None
        return "WORD_FINALIZED"

    def backspace(self) -> str:
        """Deletes the last letter from the current word, or the last completed word."""
        if self.current_word_letters:
            self.current_word_letters.pop()
            self.is_armed = True
            return "LETTER_DELETED"
        elif self.completed_words:
            self.completed_words.pop()
            return "WORD_DELETED"
        return "BUFFER_EMPTY"

    def clear(self) -> None:
        """Clears both current word and completed word buffers."""
        self.current_word_letters = []
        self.completed_words = []
        self.candidate_char = None
        self.candidate_start_time = None
        self.last_confirmed_char = None
        self.is_armed = True

    @property
    def current_word(self) -> str:
        return "".join(self.current_word_letters)

    @property
    def full_text(self) -> str:
        sentence = " ".join(self.completed_words)
        curr = self.current_word
        return f"{sentence} {curr}".strip() if curr else sentence

    def process_prediction(
        self,
        predicted_char: Optional[str],
        confidence: float = 1.0,
        is_valid: bool = True
    ) -> Dict[str, Any]:
        """Convenience method matching feed_frame."""
        return self.feed_frame(
            predicted_char=predicted_char,
            confidence=confidence,
            is_reliable=is_valid,
            is_fist_closed=False
        )
