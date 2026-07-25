from __future__ import annotations
import os
import pickle
from pathlib import Path
import numpy as np


def pickle_loading_allowed() -> bool:
    """Whether pickle/torch model deserialization is permitted.

    ``pickle.load`` and ``torch.load`` execute arbitrary code embedded in the
    file, so loading a model you did not create is a remote-code-execution
    vector. It stays off unless the operator opts in for this process (CLI
    ``--allow-pickle`` sets the env var below).
    """
    return os.environ.get("UNRELABEL_ALLOW_PICKLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _refuse_untrusted_deserialization(path: Path, fmt: str) -> None:
    raise RuntimeError(
        f"Refusing to load {fmt} model {str(path)!r}: deserializing it would "
        "execute any code embedded in the file. This is disabled by default. "
        "If you created or trust this file, re-run with --allow-pickle "
        "(or set UNRELABEL_ALLOW_PICKLE=1)."
    )


class ModelWrapper:
    """Uniform interface over sklearn, PyTorch, ONNX, and Keras backends."""

    def __init__(self, model, backend: str):
        self._model = model
        self._backend = backend

    @property
    def backend(self) -> str:
        return self._backend

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ModelWrapper":
        if self._backend == "sklearn":
            self._model.fit(X, y)
        elif self._backend == "pytorch":
            raise NotImplementedError("PyTorch .fit() not supported. Retrain externally.")
        else:
            raise NotImplementedError(f"fit() not implemented for backend '{self._backend}'")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._backend == "sklearn":
            return self._model.predict(X)
        elif self._backend == "pytorch":
            import torch
            self._model.eval()
            with torch.no_grad():
                logits = self._model(torch.tensor(X, dtype=torch.float32))
                return logits.argmax(dim=1).numpy()
        elif self._backend == "onnx":
            input_name = self._model.get_inputs()[0].name
            return np.array(self._model.run(None, {input_name: X.astype(np.float32)})[0])
        else:
            raise NotImplementedError(f"predict() not implemented for backend '{self._backend}'")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._backend == "sklearn":
            if hasattr(self._model, "predict_proba"):
                return self._model.predict_proba(X)
            raise AttributeError("Model does not support predict_proba")
        elif self._backend == "pytorch":
            import torch
            import torch.nn.functional as F
            self._model.eval()
            with torch.no_grad():
                logits = self._model(torch.tensor(X, dtype=torch.float32))
                return F.softmax(logits, dim=1).numpy()
        else:
            raise NotImplementedError(f"predict_proba() not implemented for backend '{self._backend}'")

    def clone(self) -> "ModelWrapper":
        """Return a fresh unfitted copy of the same model architecture."""
        if self._backend == "sklearn":
            from sklearn.base import clone
            return ModelWrapper(clone(self._model), self._backend)
        raise NotImplementedError(f"clone() not implemented for backend '{self._backend}'")


class ModelLoader:
    def load(self, source: str) -> ModelWrapper:
        path = Path(source)
        suffix = path.suffix.lower()

        if suffix == ".pkl":
            return self._load_pickle(path)
        elif suffix in (".pt", ".pth"):
            return self._load_pytorch(path)
        elif suffix in (".h5", ".keras"):
            return self._load_keras(path)
        elif suffix == ".onnx":
            return self._load_onnx(path)
        else:
            raise ValueError(
                f"Unsupported model format '{suffix}'. "
                "Supported: .pkl, .pt, .pth, .h5, .keras, .onnx"
            )

    def load_huggingface(self, model_id: str, task: str = "text-classification") -> ModelWrapper:
        from transformers import pipeline
        pipe = pipeline(task, model=model_id)
        return ModelWrapper(pipe, backend="huggingface")

    def _load_pickle(self, path: Path) -> ModelWrapper:
        if not pickle_loading_allowed():
            _refuse_untrusted_deserialization(path, "pickle")
        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301 - gated behind pickle_loading_allowed()
        return ModelWrapper(model, backend="sklearn")

    def _load_pytorch(self, path: Path) -> ModelWrapper:
        if not pickle_loading_allowed():
            _refuse_untrusted_deserialization(path, "PyTorch")
        import torch
        # weights_only=False is required to rebuild a full nn.Module; it also
        # unpickles arbitrary objects, which is why this path is gated above.
        model = torch.load(path, map_location="cpu", weights_only=False)
        return ModelWrapper(model, backend="pytorch")

    def _load_keras(self, path: Path) -> ModelWrapper:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return ModelWrapper(model, backend="keras")

    def _load_onnx(self, path: Path) -> ModelWrapper:
        import onnxruntime as rt
        session = rt.InferenceSession(str(path))
        return ModelWrapper(session, backend="onnx")
