import torch

from sportsintell.losses.dmal import DirectionalMotionAlignmentLoss


def test_dmal_zero_when_boxes_match():
    loss_fn = DirectionalMotionAlignmentLoss()
    prev = torch.tensor([[10.0, 10.0, 4.0, 4.0]])
    nxt = torch.tensor([[12.0, 10.0, 4.0, 4.0]])
    loss = loss_fn(prev, nxt, nxt)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
