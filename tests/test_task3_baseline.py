import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is None:
    ConditionalUNet = None
else:
    from components.models.conditional_unet import ConditionalUNet


@unittest.skipUnless(torch is not None, "torch is not installed")
class ConditionalUNetTests(unittest.TestCase):
    def test_uses_both_domains_and_preserves_shape(self):
        model = ConditionalUNet(base_channels=4, max_channels=16, levels=2).eval()
        image = torch.randn(1, 1, 31, 35)

        with torch.no_grad():
            output = model(image, torch.tensor([2]), torch.tensor([1]))
            other_source = model(image, torch.tensor([2]), torch.tensor([0]))

        self.assertEqual(output.shape, image.shape)
        self.assertFalse(torch.equal(output, other_source))


if __name__ == "__main__":
    unittest.main()
