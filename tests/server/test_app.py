import pytest
from httpx import AsyncClient, ASGITransport

from unrelabel.server.app import app, state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global app state before each test to ensure test isolation."""
    state.reset()
    yield
    state.reset()


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_load_sklearn_dataset(client):
    resp = await client.post("/api/dataset/load", json={
        "source": "sklearn",
        "name": "iris",
        "test_size": 0.2,
        "seed": 42,
        "model": "sklearn:LogisticRegression",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_train"] > 0
    assert data["n_test"] > 0
    assert data["n_features"] == 4
    assert data["n_classes"] == 3
    assert len(data["class_names"]) == 3
    assert "baseline_accuracy" in data
    assert data["baseline_accuracy"] > 0.8


@pytest.mark.asyncio
async def test_load_dataset_invalid_source(client):
    resp = await client.post("/api/dataset/load", json={
        "source": "invalid",
        "name": "nope",
        "model": "sklearn:LogisticRegression",
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_csv(client, tmp_path):
    import pandas as pd
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "f2": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    }).to_csv(csv_path, index=False)

    with open(csv_path, "rb") as f:
        resp = await client.post(
            "/api/dataset/upload",
            files={"file": ("test.csv", f, "text/csv")},
            data={
                "label_col": "label",
                "test_size": "0.2",
                "seed": "42",
                "model": "sklearn:LogisticRegression",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_features"] == 2
    assert data["n_classes"] == 2


@pytest.mark.asyncio
async def test_upload_npz(client, tmp_path):
    import numpy as np
    npz_path = tmp_path / "test.npz"
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                   [2, 1], [4, 3], [6, 5], [8, 7], [10, 9]], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    np.savez(npz_path, X=X, y=y)

    with open(npz_path, "rb") as f:
        resp = await client.post(
            "/api/dataset/upload",
            files={"file": ("test.npz", f, "application/octet-stream")},
            data={
                "label_col": "y",
                "test_size": "0.2",
                "seed": "42",
                "model": "sklearn:LogisticRegression",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_features"] == 2
    assert data["n_classes"] == 2


@pytest.mark.asyncio
async def test_scatter_endpoint_requires_dataset(client):
    resp = await client.get("/api/dataset/scatter")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_scatter_endpoint_after_load(client):
    await client.post("/api/dataset/load", json={
        "source": "sklearn",
        "name": "iris",
        "test_size": 0.2,
        "seed": 42,
        "model": "sklearn:LogisticRegression",
    })
    resp = await client.get("/api/dataset/scatter")
    assert resp.status_code == 200
    data = resp.json()
    assert "coords" in data
    assert "labels" in data
    assert "class_names" in data
    assert data["strategy"] in ("raw", "pca", "stats")


@pytest.mark.asyncio
async def test_scatter_returns_explained_variance(client):
    await client.post("/api/dataset/load", json={
        "source": "sklearn", "name": "iris",
        "test_size": 0.2, "seed": 42,
        "model": "sklearn:LogisticRegression",
    })
    resp = await client.get("/api/dataset/scatter")
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] == "pca"
    assert "explained_variance" in data
    assert len(data["explained_variance"]) == 2
    assert all(0 < v < 1 for v in data["explained_variance"])
