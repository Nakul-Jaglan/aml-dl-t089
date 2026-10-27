"""Explicit baseline families and modest, validation-only candidate grids."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ["lr", "knn", "rf", "svm"]


def candidates(model_name, search="compact"):
    grids = {
        "lr": [{"C": 0.1}, {"C": 1.0}],
        "knn": [{"n_neighbors": 3}, {"n_neighbors": 5}],
        "rf": [{"max_depth": 20}, {"max_depth": None}],
        "svm": [{"C": 1.0}, {"C": 10.0}],
    }
    return grids[model_name] if search == "compact" else [grids[model_name][-1 if model_name == "lr" else 0]]


def make_model(model_name, feature_name, parameters=None, smoke=False):
    parameters = parameters or {}
    scale = StandardScaler() if model_name in {"lr", "svm"} or (model_name == "knn" and feature_name == "hog") else "passthrough"
    if model_name == "lr":
        model = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42, **parameters)
    elif model_name == "knn":
        model = KNeighborsClassifier(weights="distance", algorithm="brute", n_jobs=1, p=2, **parameters)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=30 if smoke else 200, min_samples_leaf=1,
                                       max_features="sqrt", bootstrap=True, random_state=42, n_jobs=1, **parameters)
    elif model_name == "svm":
        model = SVC(kernel="rbf", gamma="scale", probability=False, cache_size=512, **parameters)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return Pipeline([("scale", scale), ("model", model)])

