import torch

from components.models.conditional_unet import ConditionalUNet


def test_conditional_unet_uses_both_domains_and_preserves_shape():
    model = ConditionalUNet(base_channels=4, max_channels=16, levels=2).eval()
    image = torch.randn(1, 1, 31, 35)

    with torch.no_grad():
        output = model(image, torch.tensor([2]), torch.tensor([1]))
        other_source = model(image, torch.tensor([2]), torch.tensor([0]))

    assert output.shape == image.shape
    assert not torch.equal(output, other_source)
