import unittest
from aihandler.runner import SDRunner


class TestPromptWeightConvert(unittest.TestCase):
    def test_use_case_a(self):
        prompt = "Example (ABC): 1.23 XYZ (DEF) (GHI:2.3)"
        expected_prompt = "Example (ABC)1.1: 1.23 XYZ (DEF)1.1 (GHI)2.3"
        self.assertEqual(SDRunner.convert_prompt_weights(prompt), expected_prompt)

    def test_use_case_b(self):
        prompt = "(a dog:0.5) and a cat"
        expected_prompt = "(a dog)0.5 and a cat"
        self.assertEqual(SDRunner.convert_prompt_weights(prompt), expected_prompt)

    def test_use_case_c(self):
        prompt = "A perfect photo of a woman wearing a respirator wandering through the (toxic wasteland:1.3)"
        expected_prompt = "A perfect photo of a woman wearing a respirator wandering through the (toxic wasteland)1.3"
        self.assertEqual(SDRunner.convert_prompt_weights(prompt), expected_prompt)

    def test_use_case_d(self):
        prompt = "(worst quality:0.8), fantasy, cartoon, halftone print, (cinematic:1.2), verybadimagenegative_v1.3, easynegative, (surreal:0.8), (modernism:0.8), (art deco:0.8), (art nouveau:0.8)"
        expected_prompt = "(worst quality)0.8, fantasy, cartoon, halftone print, (cinematic)1.2, verybadimagenegative_v1.3, easynegative, (surreal)0.8, (modernism)0.8, (art deco)0.8, (art nouveau)0.8"
        self.assertEqual(SDRunner.convert_prompt_weights(prompt), expected_prompt)
