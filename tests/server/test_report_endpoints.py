import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from unrelabel.server.app import app, state


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest_asyncio.fixture(autouse=True)
async def run_attack(client):
    state.reset()
    await client.post("/api/dataset/load", json={
        "source": "sklearn", "name": "iris",
        "test_size": 0.2, "seed": 42,
        "model": "sklearn:LogisticRegression",
    })
    await client.post("/api/attack/run", json={
        "attack_type": "label_flipping",
        "poison_rates": [0.1],
        "seed": 42,
    })
    yield
    state.reset()


@pytest.mark.asyncio
async def test_report_json(client):
    resp = await client.get("/api/report/json")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    data = resp.json()
    assert "attack_type" in data


@pytest.mark.asyncio
async def test_report_html(client):
    resp = await client.get("/api/report/html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_report_json_no_result(client):
    state.result = None
    resp = await client.get("/api/report/json")
    assert resp.status_code == 400
