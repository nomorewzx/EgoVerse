import pytest
import torch
from torch import nn

from egomimic.models.denoising_policy import DenoisingPolicy


def test_action_dim_loss_weights_mask_rotation_dims():
    policy = DenoisingPolicy(
        model=nn.Identity(),
        action_horizon=2,
        infer_ac_dims={"ego_view_right_arm": 6},
        action_dim_loss_weights=[1, 1, 1, 0, 0, 0],
        normalize_action_dim_loss_weights=True,
    )
    pred = torch.zeros(2, 3, 6)
    target = torch.ones(2, 3, 6)
    target[..., 3:] = 100.0

    loss = policy.loss_fn(pred, target)

    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_action_dim_loss_weights_require_action_dim_match():
    policy = DenoisingPolicy(
        model=nn.Identity(),
        action_horizon=2,
        infer_ac_dims={"ego_view_right_arm": 6},
        action_dim_loss_weights=[1, 1, 1, 0, 0],
    )
    pred = torch.zeros(2, 3, 6)
    target = torch.zeros(2, 3, 6)

    with pytest.raises(ValueError, match="length must match action dim"):
        policy.loss_fn(pred, target)
