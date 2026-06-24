import importlib
import unittest

import numpy as np

from modules.decision_engine import calculate_final_score, generate_reasoning
from modules.emotion_analysis import analyze_image


class ResumeParserImportTest(unittest.TestCase):
    def test_modules_resume_parser_import(self):
        module = importlib.import_module("modules.resume_parser")
        self.assertTrue(hasattr(module, "extract_resume_text"))

    def test_calculate_final_score_accepts_dict_payload(self):
        result = calculate_final_score(
            {
                "resume_score": 80,
                "interview_score": 75,
                "voice_score": 70,
                "behavioral_score": 65,
                "fraud_score": 10,
            }
        )
        self.assertIn("final_score", result)
        self.assertGreaterEqual(result["final_score"], 0)

    def test_generate_reasoning_accepts_dict_payload(self):
        reasoning = generate_reasoning(
            {
                "resume_score": 80,
                "interview_score": 75,
                "voice_score": 70,
                "behavioral_score": 65,
                "fraud_score": 10,
            }
        )
        self.assertIsInstance(reasoning, str)
        self.assertTrue(reasoning)

    def test_analyze_image_accepts_numpy_array(self):
        image_array = np.zeros((32, 32, 3), dtype=np.uint8)
        result = analyze_image(image_array)
        self.assertIn("emotions", result)
        self.assertIn("dominant", result)
        self.assertIn("confidence", result)
        self.assertTrue(result["emotions"])


if __name__ == "__main__":
    unittest.main()
