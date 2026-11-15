"""Small metric helpers shared by the final paper experiments."""

import numpy as np

from src.benchmark_models import predict


def scores(models, inputs, targets):
    result = {}
    for family, heads in models.items():
        result[family] = {}
        for key, (truth, mask) in targets.items():
            if not mask.any():
                continue
            prediction = (predict(heads[key], inputs[mask], True)[:, 1]
                          if key == "risk" else predict(heads[key], inputs[mask]))
            error = prediction - truth[mask]
            result[family][key] = {
                "n": int(mask.sum()),
                "mae": float(abs(error).mean()),
                "mse": float(np.square(error).mean()),
            }
            if key == "risk":
                result[family][key]["calibration"] = [
                    {"n": int(cell.sum()), "predicted": float(prediction[cell].mean()),
                     "observed": float(truth[mask][cell].mean())}
                    for lower in np.arange(0, 1, .2)
                    if (cell := ((prediction >= lower) &
                                 (prediction < lower + .2 + (1e-9 if lower > .79 else 0)))).any()
                ]
    return result


def targets(values):
    return {key: (np.asarray(value), np.ones(len(value), bool)) for key, value in values.items()}


def summarize(rows):
    if not rows or "profit" not in rows[0]:
        raise ValueError("Expected nonempty SimBank outcome rows")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
