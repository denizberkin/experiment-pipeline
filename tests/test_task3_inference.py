import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

if torch is None:
    ConditionalUNet = None
    predict_slab = None
else:
    from components.models.conditional_unet import ConditionalUNet
    from scripts.inference_task3_conditional_unet import predict_slab


@unittest.skipUnless(torch is not None, "torch is not installed")
class PredictSlabTests(unittest.TestCase):
    def test_only_populates_submission_range(self):
        model = ConditionalUNet(base_channels=4, max_channels=8, levels=1).eval()
        volume = np.ones((12, 14, 364), dtype=np.float32) * 0.5
        prediction = predict_slab(model, volume, 0, 1, torch.device("cpu"), 4)

        self.assertEqual(prediction.shape, volume.shape)
        self.assertTrue(prediction[:, :, 150:180].any())
        self.assertFalse(prediction[:, :, :150].any())
        self.assertFalse(prediction[:, :, 180:].any())


if __name__ == "__main__":
    unittest.main()
