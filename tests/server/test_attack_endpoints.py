import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from unrelabel.server.app import app, state


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest_asyncio.fixture(autouse=True)
async def load_dataset(client):
    state.reset()
    await client.post("/api/dataset/load", json={
        "source": "sklearn", "name": "iris",
        "test_size": 0.2, "seed": 42,
        "model": "sklearn:LogisticRegression",
    })
    yield
    state.reset()


@pytest.mark.asyncio
async def test_attack_run_label_flipping(client):
    resp = await client.post("/api/attack/run", json={
        "attack_type": "label_flipping",
        "poison_rates": [0.1, 0.2],
        "seed": 42,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["attack_type"] == "label_flip"
    assert data["clean_accuracy"] > 0
    assert len(data["sweep_results"]) == 2


@pytest.mark.asyncio
async def test_attack_run_targeted(client):
    resp = await client.post("/api/attack/run", json={
        "attack_type": "targeted_label",
        "poison_rates": [0.3],
        "source_class": 0,
        "target_class": 1,
        "seed": 42,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["attack_type"] == "targeted_label"


@pytest.mark.asyncio
async def test_attack_run_clean_label(client):
    resp = await client.post("/api/attack/run", json={
        "attack_type": "clean_label",
        "source_class": 0,
        "target_class": 1,
        "n_neighbors": 5,
        "epsilon": 0.25,
        "seed": 42,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["attack_type"] == "clean_label"


@pytest.mark.asyncio
async def test_attack_run_no_dataset(client):
    state.reset()
    resp = await client.post("/api/attack/run", json={
        "attack_type": "label_flipping",
        "poison_rates": [0.1],
        "seed": 42,
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_validate_valid(client):
    resp = await client.post("/api/attack/validate", json={
        "attack_type": "label_flipping",
        "poison_rates": [0.1],
        "seed": 42,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["errors"] == []


@pytest.mark.asyncio
async def test_validate_clean_label_missing_source(client):
    resp = await client.post("/api/attack/validate", json={
        "attack_type": "clean_label",
        "target_class": 1,
        "seed": 42,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert any("source_class" in e.lower() for e in data["errors"])


@pytest.mark.asyncio
async def test_validate_same_source_target(client):
    resp = await client.post("/api/attack/validate", json={
        "attack_type": "targeted_label",
        "poison_rates": [0.1],
        "source_class": 0,
        "target_class": 0,
        "seed": 42,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert len(data["errors"]) > 0
