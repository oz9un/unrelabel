import pickle
import numpy as np
import pytest
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from unrelabel.loaders.model_loader import ModelLoader, ModelWrapper


@pytest.fixture
def sklearn_model_path(tmp_path, monkeypatch):
    # These tests exercise the trusted-load path, so opt in explicitly. The
    # default-off behavior is covered by test_load_pickle_refused_by_default.
    monkeypatch.setenv("UNRELABEL_ALLOW_PICKLE", "1")
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    model = LogisticRegression().fit(X, y)
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path, X, y


def test_load_pickle_refused_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("UNRELABEL_ALLOW_PICKLE", raising=False)
    model = LogisticRegression().fit(*make_classification(n_samples=40, n_features=4, random_state=1))
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    with pytest.raises(RuntimeError, match="disabled by default"):
        ModelLoader().load(str(path))


def test_load_sklearn_pkl(sklearn_model_path):
    path, X, y = sklearn_model_path
    loader = ModelLoader()
    wrapper = loader.load(str(path))
    assert isinstance(wrapper, ModelWrapper)
    preds = wrapper.predict(X)
    assert preds.shape == (100,)


def test_wrapper_predict_proba(sklearn_model_path):
    path, X, y = sklearn_model_path
    wrapper = ModelLoader().load(str(path))
    proba = wrapper.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_wrapper_fit_predict(sklearn_model_path):
    path, X, y = sklearn_model_path
    wrapper = ModelLoader().load(str(path))
    X_new, y_new = make_classification(n_samples=50, n_features=4, random_state=0)
    wrapper.fit(X_new, y_new)
    preds = wrapper.predict(X_new)
    assert len(preds) == 50


def test_unsupported_format_raises(tmp_path):
    bad_file = tmp_path / "model.xyz"
    bad_file.write_text("not a model")
    with pytest.raises(ValueError, match="Unsupported model format"):
        ModelLoader().load(str(bad_file))
