"""
Unit tests for ISL Landmark-based English Word Recognition Pipeline.

Covers:
1. Landmark extraction & coordinate normalization (126D canonical vector).
2. Model input validation & error handling.
3. Prediction state machine & temporal stabilization.
4. EnglishWordBuilder (character sequencing, repeated letter release/re-arm, word boundaries).
5. Robust rejection of background/invalid frames.
"""

import unittest
import numpy as np
import time
from unittest.mock import MagicMock

# Import components from src
from src.word_builder import EnglishWordBuilder
from src.preprocessing import normalize_landmarks_126d, validate_landmarks_vector


class TestLandmarkNormalization(unittest.TestCase):
    """Test 126D landmark vector generation, centering, and scale normalization."""

    def test_empty_landmarks_rejection(self):
        """No detected hand should return all zeros or trigger validation failure."""
        empty_vec = np.zeros(126, dtype=np.float32)
        is_valid, msg = validate_landmarks_vector(empty_vec)
        self.assertFalse(is_valid)
        self.assertIn("empty", msg.lower())

    def test_single_hand_normalization_shape(self):
        """Single hand landmarks should produce exactly 126D vector with other hand zero-padded."""
        # Mock 21 3D points for right hand
        mock_landmarks = []
        for i in range(21):
            mock_landmarks.append({'x': 0.5 + i * 0.01, 'y': 0.5 + i * 0.01, 'z': 0.0})

        vec = normalize_landmarks_126d(mock_landmarks, handedness="Right")
        self.assertEqual(vec.shape, (126,))
        self.assertEqual(vec.dtype, np.float32)

        # Right hand should be populated in indices [63:126], Left hand [0:63] should be 0
        self.assertTrue(np.all(vec[:63] == 0.0))
        self.assertFalse(np.all(vec[63:] == 0.0))

    def test_scale_and_translation_invariance(self):
        """Wrist-relative centering ensures wrist landmark (landmark 0) is at origin (0, 0, 0)."""
        mock_landmarks = []
        wrist_x, wrist_y, wrist_z = 0.35, 0.72, -0.05
        for i in range(21):
            mock_landmarks.append({'x': wrist_x + i * 0.01, 'y': wrist_y + i * 0.01, 'z': wrist_z + i * 0.005})

        vec = normalize_landmarks_126d(mock_landmarks, handedness="Left")
        self.assertEqual(vec.shape, (126,))
        
        # Landmark 0 of Left hand should be at [0, 1, 2] and equal to (0, 0, 0)
        np.testing.assert_allclose(vec[0:3], [0.0, 0.0, 0.0], atol=1e-5)


class TestModelInputValidation(unittest.TestCase):
    """Test verification and sanitization of feature vectors before classifier inference."""

    def test_nan_inf_rejection(self):
        vec = np.zeros(126, dtype=np.float32)
        vec[10] = np.nan
        is_valid, msg = validate_landmarks_vector(vec)
        self.assertFalse(is_valid)
        self.assertIn("nan", msg.lower())

        vec[10] = np.inf
        is_valid, msg = validate_landmarks_vector(vec)
        self.assertFalse(is_valid)
        self.assertIn("inf", msg.lower())

    def test_invalid_dimension_rejection(self):
        vec = np.zeros(63, dtype=np.float32)
        is_valid, msg = validate_landmarks_vector(vec)
        self.assertFalse(is_valid)
        self.assertIn("shape", msg.lower())


class TestEnglishWordBuilder(unittest.TestCase):
    """Test English word assembly, release/re-arm transition for repeated letters, and controls."""

    def setUp(self):
        self.builder = EnglishWordBuilder(
            confirm_duration=0.02,
            rearm_duration=0.05,
            space_gesture_duration=0.10
        )

    def test_single_letter_addition(self):
        self.builder.process_prediction('H', 0.95, is_valid=True)
        time.sleep(0.03)
        res = self.builder.process_prediction('H', 0.95, is_valid=True)
        self.assertEqual(res['current_word'], 'H')
        self.assertEqual(res['confirmed_char'], 'H')

    def test_held_pose_does_not_duplicate(self):
        # Initial frame
        self.builder.process_prediction('H', 0.95, is_valid=True)
        time.sleep(0.03)
        # Confirmed
        self.builder.process_prediction('H', 0.95, is_valid=True)
        self.assertEqual(self.builder.current_word, 'H')
        # Continued holding pose should not duplicate
        for _ in range(5):
            time.sleep(0.01)
            res = self.builder.process_prediction('H', 0.95, is_valid=True)
        self.assertEqual(res['current_word'], 'H')

    def test_spelling_hello_with_repeated_l(self):
        # Step 1: 'H'
        self.builder.process_prediction('H', 0.95, is_valid=True)
        time.sleep(0.03)
        self.builder.process_prediction('H', 0.95, is_valid=True)
        self.assertEqual(self.builder.current_word, 'H')

        # Step 2: 'E' (different sign -> immediate re-arm)
        self.builder.process_prediction('E', 0.95, is_valid=True)
        time.sleep(0.03)
        self.builder.process_prediction('E', 0.95, is_valid=True)
        self.assertEqual(self.builder.current_word, 'HE')

        # Step 3: First 'L'
        self.builder.process_prediction('L', 0.95, is_valid=True)
        time.sleep(0.03)
        self.builder.process_prediction('L', 0.95, is_valid=True)
        self.assertEqual(self.builder.current_word, 'HEL')

        # Step 4: Re-arm by holding neutral pose for >= rearm_duration (0.05s)
        self.builder.process_prediction(None, 0.0, is_valid=False)
        time.sleep(0.06)
        self.builder.process_prediction(None, 0.0, is_valid=False)

        # Step 5: Second 'L'
        self.builder.process_prediction('L', 0.95, is_valid=True)
        time.sleep(0.03)
        self.builder.process_prediction('L', 0.95, is_valid=True)
        self.assertEqual(self.builder.current_word, 'HELL')

        # Step 6: 'O'
        self.builder.process_prediction('O', 0.95, is_valid=True)
        time.sleep(0.03)
        self.builder.process_prediction('O', 0.95, is_valid=True)
        self.assertEqual(self.builder.current_word, 'HELLO')

    def test_word_finalization_and_controls(self):
        for char in ['C', 'A', 'T']:
            self.builder.process_prediction(char, 0.95, is_valid=True)
            time.sleep(0.03)
            self.builder.process_prediction(char, 0.95, is_valid=True)
        self.assertEqual(self.builder.current_word, 'CAT')

        # Finalize word
        self.builder.finalize_word()
        self.assertEqual(self.builder.current_word, "")
        self.assertEqual(self.builder.completed_words, ["CAT"])

        # Backspace test on new word
        for char in ['D', 'O', 'G', 'Z']:
            self.builder.process_prediction(char, 0.95, is_valid=True)
            time.sleep(0.03)
            self.builder.process_prediction(char, 0.95, is_valid=True)
        self.assertEqual(self.builder.current_word, 'DOGZ')
        self.builder.backspace()
        self.assertEqual(self.builder.current_word, 'DOG')

        # Clear buffer test
        self.builder.clear()
        self.assertEqual(self.builder.current_word, "")
        self.assertEqual(self.builder.completed_words, [])


if __name__ == '__main__':
    unittest.main()
