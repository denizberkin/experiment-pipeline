import numpy as np
import torch

from components.models.conditional_unet import ConditionalUNet
from scripts.inference_task3_conditional_unet import predict_slab


def test_predict_slab_only_populates_submission_range():
    model = ConditionalUNet(base_channels=4, max_channels=8, levels=1).eval()
    volume = np.ones((12, 14, 364), dtype=np.float32) * 0.5
    prediction = predict_slab(model, volume, 0, 1, torch.device("cpu"), 4)

    assert prediction.shape == volume.shape
    assert prediction[:, :, 150:180].any()
    assert not prediction[:, :, :150].any()
    assert not prediction[:, :, 180:].any()
