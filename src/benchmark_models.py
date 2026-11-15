"""Supervised model families and replicate seeds for simulator benchmarks."""

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


FAMILIES = ("Regression", "RandomForest", "CatBoost", "XGBoost")


def seeds(master, count):
    return [int(child.generate_state(1)[0] % (2**31 - 1))
            for child in np.random.SeedSequence(master).spawn(count)]


def estimator(family, target, seed, trees, threads=1):
    classification = target in ("risk", "outcome")
    if family == "Regression":
        model = LogisticRegression(max_iter=600) if classification else (
            PoissonRegressor(max_iter=400) if target == "count" else Ridge())
        return make_pipeline(StandardScaler(), model)
    if family == "RandomForest":
        return (RandomForestClassifier if classification else RandomForestRegressor)(
            n_estimators=trees, min_samples_leaf=5, n_jobs=threads, random_state=seed)
    if family == "CatBoost":
        return (CatBoostClassifier if classification else CatBoostRegressor)(
            iterations=trees, depth=4, thread_count=threads, random_seed=seed,
            verbose=False, allow_writing_files=False)
    if family == "XGBoost":
        return (XGBClassifier if classification else XGBRegressor)(
            n_estimators=trees, max_depth=4, learning_rate=.05, n_jobs=threads, random_state=seed)
    raise ValueError(f"Unknown model family: {family}")


def predict(model, x, probability=False):
    method = model.predict_proba if probability else model.predict
    params = {"thread_count": 1} if isinstance(model, (CatBoostClassifier, CatBoostRegressor)) else {}
    return method(x, **params)
