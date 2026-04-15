import torch

from sportsintell.models.sportsintell import SportsIntell
from sportsintell.models.baselines import VanillaTransformerPredictor


def test_sportsintell_output_shape():
    model = SportsIntell()
    x = torch.randn(2, 12, 8)
    y = model(x)
    assert y.shape == (2, 4)


def test_vanilla_transformer_output_shape():
    model = VanillaTransformerPredictor()
    x = torch.randn(4, 12, 8)
    y = model(x)
    assert y.shape == (4, 4)
