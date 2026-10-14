"""Video checkpoint fixtures kept outside production models and strategies."""

from copy import deepcopy


def legacy_checkpoint(strategy):
    """Encode the former combined model/trainer checkpoint layout."""
    state = deepcopy(strategy.state_dict())
    model = state.pop("model")
    state.pop("version")
    state["config"] = {**model["config"], **state["config"]}
    state["model"] = model["module"]
    for key in ("calibration_median", "calibration_scale", "threshold", "distance_mode"):
        state[key] = model[key]
    return state
