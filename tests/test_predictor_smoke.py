from __future__ import annotations

import unittest
from pathlib import Path

import torch

from inference.predictor import DEFAULT_CLASS_NAMES, PlantDiseasePredictor


class PlantDiseasePredictorSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.checkpoint_path = cls.project_root / "checkpoints" / "best_model.pt"
        cls.sample_dir = cls.project_root / "dataset_redistributed" / "valid" / "images"
        cls.sample_images = sorted(
            path
            for path in cls.sample_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if len(cls.sample_images) < 2:
            raise unittest.SkipTest(
                "Not enough sample images available for smoke testing."
            )

        cls.predictor = PlantDiseasePredictor(checkpoint_path=cls.checkpoint_path)

    def test_model_loaded_and_device_selected(self) -> None:
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(self.predictor.device.type, expected_device)
        self.assertTrue(self.checkpoint_path.exists())

    def test_single_image_prediction(self) -> None:
        result = self.predictor.predict(self.sample_images[0])
        self.assertIn(result.predicted_class, DEFAULT_CLASS_NAMES)
        self.assertIn(result.predicted_class_id, range(len(DEFAULT_CLASS_NAMES)))
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=4)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_batch_prediction(self) -> None:
        results = self.predictor.predict_batch(self.sample_images[:2], batch_size=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(self.predictor.last_failures), 0)
        for result in results:
            self.assertIn(result.predicted_class, DEFAULT_CLASS_NAMES)
            self.assertIn(result.predicted_class_id, range(len(DEFAULT_CLASS_NAMES)))
            self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=4)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
