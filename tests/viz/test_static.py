"""The self-contained frontend ships with the package and is internally linked."""

from importlib import resources


def _static():
    return resources.files("auraflow.viz").joinpath("static")


def test_static_files_present():
    static = _static()
    assert static.joinpath("index.html").is_file()
    assert static.joinpath("app.js").is_file()


def test_index_references_app_js():
    html = _static().joinpath("index.html").read_text(encoding="utf-8")
    assert "app.js" in html
    # three.js is pinned via an import map (no build step, no npm).
    assert "importmap" in html
    assert "three@0.160" in html


def test_app_js_speaks_the_protocol():
    js = _static().joinpath("app.js").read_text(encoding="utf-8")
    # Frontend and Python must agree on the protocol version.
    assert "PROTOCOL_VERSION = 1" in js
    assert "decodeMessage" in js
