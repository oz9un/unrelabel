# tests/server/test_integration.py
"""End-to-end integration test: load → attack → report."""
import pytest
from httpx import AsyncClient, ASGITransport

from unrelabel.server.app import app, state


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture(autouse=True)
def clean_state():
    state.reset()
    yield
    state.reset()


@pytest.mark.asyncio
async def test_full_flow_label_flipping(client):
    # Step 1: Load dataset
    resp = await client.post("/api/dataset/load", json={
        "source": "sklearn", "name": "iris",
        "test_size": 0.2, "seed": 42,
        "model": "sklearn:LogisticRegression",
    })
    assert resp.status_code == 200
    ds = resp.json()
    assert ds["baseline_accuracy"] > 0.8

    # Step 1b: Scatter
    resp = await client.get("/api/dataset/scatter")
    assert resp.status_code == 200
    assert resp.json()["strategy"] == "pca"

    # Step 2: Run attack
    resp = await client.post("/api/attack/run", json={
        "attack_type": "label_flipping",
        "poison_rates": [0.1, 0.2, 0.3],
        "seed": 42,
    })
    assert resp.status_code == 200
    result = resp.json()
    assert len(result["sweep_results"]) == 3
    assert result["vulnerability_score"] > 0

    # Step 3: Download reports
    resp = await client.get("/api/report/json")
    assert resp.status_code == 200
    assert resp.json()["attack_type"] == "label_flip"

    resp = await client.get("/api/report/html")
    assert resp.status_code == 200
    assert "<!DOCTYPE html>" in resp.text or "<html" in resp.text


@pytest.mark.asyncio
async def test_full_flow_targeted(client):
    await client.post("/api/dataset/load", json={
        "source": "sklearn", "name": "iris",
        "test_size": 0.2, "seed": 42,
        "model": "sklearn:LogisticRegression",
    })
    resp = await client.post("/api/attack/run", json={
        "attack_type": "targeted_label",
        "poison_rates": [0.3],
        "source_class": 0, "target_class": 1,
        "seed": 42,
    })
    assert resp.status_code == 200
    assert resp.json()["attack_type"] == "targeted_label"


@pytest.mark.asyncio
async def test_full_flow_clean_label(client):
    await client.post("/api/dataset/load", json={
        "source": "sklearn", "name": "iris",
        "test_size": 0.2, "seed": 42,
        "model": "sklearn:LogisticRegression",
    })
    resp = await client.post("/api/attack/run", json={
        "attack_type": "clean_label",
        "source_class": 0, "target_class": 1,
        "n_neighbors": 5, "epsilon": 0.25,
        "seed": 42,
    })
    assert resp.status_code == 200
    assert resp.json()["attack_type"] == "clean_label"
