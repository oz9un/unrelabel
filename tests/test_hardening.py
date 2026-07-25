"""Release-hardening guards: security headers, packaging metadata, and provenance files."""
from pathlib import Path

import pandas as pd
from starlette.testclient import TestClient

from unrelabel.playground import PlaygroundHub, create_app

ROOT = Path(__file__).resolve().parents[1]


def _client():
    return TestClient(create_app(PlaygroundHub(ROOT)))


def test_served_page_sets_a_confining_csp():
    r = _client().get("/")
    csp = r.headers.get("content-security-policy", "")
    assert csp, "no CSP header on the served page"
    assert "default-src 'self'" in csp
    assert "connect-src 'self'" in csp        # every fetch confined to this origin
    assert "frame-ancestors 'none'" in csp
    assert "http://" not in csp and "https://" not in csp  # no external origin allowed
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_security_headers_on_api_responses_too():
    r = _client().get("/api/datasets")
    assert "content-security-policy" in r.headers


def test_provenance_files_exist():
    for name in ("LICENSE", "SECURITY.md", "CHANGELOG.md", "README.md"):
        assert (ROOT / name).is_file(), f"missing {name}"
    assert (ROOT / "docs" / "THREAT_MODEL.md").is_file()


def test_pyproject_has_release_metadata():
    from setuptools.config.pyprojecttoml import read_configuration

    project = read_configuration(ROOT / "pyproject.toml")["project"]
    assert project.get("classifiers"), "no classifiers in pyproject"
    assert any("License :: OSI Approved :: MIT" in c for c in project["classifiers"])
    assert project.get("urls"), "no project URLs"
    assert project.get("authors")
