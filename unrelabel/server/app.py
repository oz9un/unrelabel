from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

from unrelabel.server.state import AppState
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from unrelabel.loaders.dataset_loader import DatasetLoader
from unrelabel.loaders.model_loader import ModelLoader, ModelWrapper

_SKLEARN_SHORTCUTS = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
}

_STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(title="unrelabel", docs_url=None, redoc_url=None)
state = AppState()


def _load_model(model_str: str) -> ModelWrapper:
    if model_str.startswith("sklearn:"):
        name = model_str.split(":", 1)[1]
        if name not in _SKLEARN_SHORTCUTS:
            raise HTTPException(400, f"Unknown sklearn model '{name}'. Choose from: {list(_SKLEARN_SHORTCUTS)}")
        return ModelWrapper(_SKLEARN_SHORTCUTS[name](random_state=42), backend="sklearn")
    elif model_str.startswith("hf:"):
        model_id = model_str.split(":", 1)[1]
        return ModelLoader().load_huggingface(model_id)
    else:
        return ModelLoader().load(model_str)


def _compute_baseline(ds, model: ModelWrapper) -> float:
    clone = model.clone()
    clone.fit(ds.X_train, ds.y_train)
    return float(accuracy_score(ds.y_test, clone.predict(ds.X_test)))


def _dataset_response() -> dict:
    if state.dataset is None:
        raise HTTPException(400, "No dataset loaded.")
    ds = state.dataset
    balance = {}
    for name, idx in zip(ds.class_names, range(len(ds.class_names))):
        balance[name] = int((ds.y_train == idx).sum())
    return {
        "n_train": ds.n_train,
        "n_test": ds.n_test,
        "n_features": ds.n_features,
        "n_classes": ds.n_classes,
        "class_names": ds.class_names,
        "feature_names": ds.feature_names,
        "balance": balance,
        "baseline_accuracy": state.baseline_accuracy,
    }


class DatasetLoadRequest(BaseModel):
    source: str
    name: str | None = None
    dataset_id: str | None = None
    label_col: str = "label"
    test_size: float = 0.2
    seed: int = 42
    model: str = "sklearn:LogisticRegression"


@app.post("/api/dataset/load")
def dataset_load(req: DatasetLoadRequest):
    loader = DatasetLoader()
    try:
        if req.source == "sklearn":
            if not req.name:
                raise HTTPException(400, "name is required for sklearn source")
            ds = loader.load_sklearn(req.name, test_size=req.test_size, seed=req.seed)
        elif req.source == "huggingface":
            if not req.dataset_id:
                raise HTTPException(400, "dataset_id is required for huggingface source")
            ds = loader.load_huggingface(
                req.dataset_id, label_col=req.label_col,
                test_size=req.test_size, seed=req.seed,
            )
        else:
            raise HTTPException(400, f"Unknown source '{req.source}'. Use 'sklearn' or 'huggingface'.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

    model = _load_model(req.model)
    baseline_acc = _compute_baseline(ds, model)

    state.dataset = ds
    state.model = model
    state.baseline_accuracy = baseline_acc
    state.result = None

    return _dataset_response()


@app.post("/api/dataset/upload")
async def dataset_upload(
    file: UploadFile = File(...),
    label_col: str = Form("label"),
    test_size: float = Form(0.2),
    seed: int = Form(42),
    model: str = Form("sklearn:LogisticRegression"),
):
    loader = DatasetLoader()
    suffix = Path(file.filename or "upload.csv").suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if suffix == ".npz":
            ds = loader.load_npz(tmp_path, label_key=label_col, test_size=test_size, seed=seed)
        elif suffix == ".csv":
            ds = loader.load_csv(tmp_path, label_col=label_col, test_size=test_size, seed=seed)
        else:
            raise HTTPException(400, f"Unsupported file type '{suffix}'. Use .csv or .npz.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    m = _load_model(model)
    baseline_acc = _compute_baseline(ds, m)

    state.dataset = ds
    state.model = m
    state.baseline_accuracy = baseline_acc
    state.result = None

    return _dataset_response()


@app.get("/api/dataset/scatter")
def dataset_scatter(
    attack_type: str | None = None,
    poison_rate: float | None = None,
    source_class: int | None = None,
):
    if state.dataset is None:
        raise HTTPException(400, "No dataset loaded. Load a dataset first.")
    ds = state.dataset
    n_features = ds.n_features
    explained_variance = None

    if n_features <= 2:
        coords = ds.X_train[:, :2].tolist()
        strategy = "raw"
    elif n_features <= 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(ds.X_train).tolist()
        strategy = "pca"
        explained_variance = pca.explained_variance_ratio_.tolist()
    else:
        stats = {}
        for name, idx in zip(ds.class_names, range(len(ds.class_names))):
            mask = ds.y_train == idx
            stats[name] = ds.X_train[mask].mean(axis=0).tolist()
        return {
            "coords": [],
            "labels": ds.y_train.tolist(),
            "class_names": ds.class_names,
            "strategy": "stats",
            "highlight_indices": None,
            "stats": stats,
            "explained_variance": None,
        }

    highlight = None
    if attack_type:
        if attack_type in ("label_flipping", "targeted_label") and poison_rate is not None:
            if source_class is not None:
                candidates = np.where(ds.y_train == source_class)[0]
            else:
                candidates = np.arange(len(ds.y_train))
            n_flip = int(len(candidates) * poison_rate)
            rng = np.random.default_rng(42)
            highlight = rng.choice(candidates, size=n_flip, replace=False).tolist()
        elif attack_type == "clean_label" and source_class is not None:
            highlight = np.where(ds.y_train == source_class)[0].tolist()

    return {
        "coords": coords,
        "labels": ds.y_train.tolist(),
        "class_names": ds.class_names,
        "strategy": strategy,
        "highlight_indices": highlight,
        "explained_variance": explained_variance,
    }


@app.get("/")
def index():
    return FileResponse(_STATIC_DIR / "index.html")


class AttackRunRequest(BaseModel):
    attack_type: str
    poison_rates: list[float] | None = None
    source_class: int | None = None
    target_class: int | None = None
    n_neighbors: int = 5
    epsilon: float = 0.25
    seed: int = 42


def _validate_attack_config(req: AttackRunRequest) -> list[str]:
    errors = []
    if req.attack_type not in ("label_flipping", "targeted_label", "clean_label"):
        errors.append(f"Unknown attack_type '{req.attack_type}'.")
    if req.attack_type in ("targeted_label", "clean_label"):
        if req.source_class is None:
            errors.append("source_class is required.")
        if req.target_class is None:
            errors.append("target_class is required.")
        if req.source_class is not None and req.target_class is not None and req.source_class == req.target_class:
            errors.append("Source and target class must differ.")
    if req.attack_type in ("label_flipping", "targeted_label"):
        if not req.poison_rates:
            errors.append("poison_rates is required.")
    return errors


@app.post("/api/attack/validate")
def attack_validate(req: AttackRunRequest):
    errors = _validate_attack_config(req)
    return {"valid": len(errors) == 0, "errors": errors}


@app.post("/api/attack/run")
def attack_run(req: AttackRunRequest):
    if state.dataset is None or state.model is None:
        raise HTTPException(400, "No dataset loaded. Load a dataset first.")

    errors = _validate_attack_config(req)
    if errors:
        raise HTTPException(422, detail=errors)

    try:
        if req.attack_type == "label_flipping":
            from unrelabel.attacks.label_flipping import LabelFlippingAttack
            attack = LabelFlippingAttack(
                poison_rates=req.poison_rates,
                source_class=req.source_class,
                target_class=req.target_class,
                seed=req.seed,
            )
        elif req.attack_type == "targeted_label":
            from unrelabel.attacks.targeted_label import TargetedLabelAttack
            attack = TargetedLabelAttack(
                source_class=req.source_class,
                target_class=req.target_class,
                poison_rates=req.poison_rates,
                seed=req.seed,
            )
        elif req.attack_type == "clean_label":
            from unrelabel.attacks.clean_label import CleanLabelAttack
            attack = CleanLabelAttack(
                source_class=req.source_class,
                target_class=req.target_class,
                n_neighbors=req.n_neighbors,
                epsilon=req.epsilon,
                seed=req.seed,
            )

        result = attack.run(state.dataset, state.model)
        state.result = result
        return result.to_dict()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/report/json")
def report_json():
    if state.result is None:
        raise HTTPException(400, "No attack result available. Run an attack first.")
    from unrelabel.reporting.report import _NumpyEncoder
    return JSONResponse(content=json.loads(json.dumps(state.result.to_dict(), cls=_NumpyEncoder)))


@app.get("/api/report/html")
def report_html():
    if state.result is None:
        raise HTTPException(400, "No attack result available. Run an attack first.")
    from unrelabel.reporting.report import ReportBuilder
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, html_path = ReportBuilder().build(state.result, Path(tmp_dir))
        html_content = html_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
