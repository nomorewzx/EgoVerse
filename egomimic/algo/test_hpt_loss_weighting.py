import torch

from egomimic.algo.hpt import HPT
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def test_compute_losses_respects_domain_loss_weights():
    hpt = object.__new__(HPT)
    hpt.device = torch.device("cpu")
    hpt.ot = False
    hpt.domains = ["so100_singlearm", "ego_view_right_arm"]
    hpt.domain_loss_weights = {
        "so100_singlearm": 1.0,
        "ego_view_right_arm": 0.2,
    }

    predictions = {
        "so100_singlearm_loss": torch.tensor(2.0),
        "ego_view_right_arm_loss": torch.tensor(4.0),
    }
    batch = {
        get_embodiment_id("so100_singlearm"): {},
        get_embodiment_id("ego_view_right_arm"): {},
    }

    loss_dict = HPT.compute_losses(hpt, predictions, batch)

    torch.testing.assert_close(loss_dict["so100_singlearm_loss"], torch.tensor(2.0))
    torch.testing.assert_close(loss_dict["ego_view_right_arm_loss"], torch.tensor(4.0))
    torch.testing.assert_close(
        loss_dict["action_loss"],
        torch.tensor((1.0 * 2.0 + 0.2 * 4.0) / 1.2),
    )


def test_compute_losses_defaults_missing_domain_weight_to_one():
    hpt = object.__new__(HPT)
    hpt.device = torch.device("cpu")
    hpt.ot = False
    hpt.domains = ["so100_singlearm", "ego_view_right_arm"]
    hpt.domain_loss_weights = {"so100_singlearm": 0.5}

    predictions = {
        "so100_singlearm_loss": torch.tensor(2.0),
        "ego_view_right_arm_loss": torch.tensor(4.0),
    }
    batch = {
        get_embodiment_id("so100_singlearm"): {},
        get_embodiment_id("ego_view_right_arm"): {},
    }

    loss_dict = HPT.compute_losses(hpt, predictions, batch)

    torch.testing.assert_close(
        loss_dict["action_loss"],
        torch.tensor((0.5 * 2.0 + 1.0 * 4.0) / 1.5),
    )
