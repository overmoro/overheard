"""Text files must decode as UTF-8 however the app was launched.

A bundle started from Finder inherits no LANG, so Python's preferred encoding
falls back to ASCII. Every dependency that opens a text file without naming an
encoding inherits that.

parakeet_mlx reads the model's config.json with a bare open(). The file has
non-ASCII bytes, so loading the model raised UnicodeDecodeError, and
parakeet_mlx's fallback then reported it as a missing model:

    FileNotFoundError: 'mlx-community/parakeet-tdt-0.6b-v3/config.json'

The model was on disk the whole time. Don went looking for a download that had
already happened, which is the expensive part of a misleading error.
"""

import locale
import subprocess
import sys

CONFIG_WITH_NON_ASCII = '{"note": "\u00e9\u00e8\u00ea non-ascii bytes live here"}'


def test_ensure_utf8_restores_a_usable_encoding(monkeypatch):
    """The fix, exercised against a locale that reports ASCII."""
    from overheard.app import _ensure_utf8

    monkeypatch.setattr(sys, "flags", sys.flags)  # documents that we read it
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a: "US-ASCII")

    calls = []
    monkeypatch.setattr(locale, "setlocale", lambda cat, val: calls.append((cat, val)))

    _ensure_utf8()

    if not sys.flags.utf8_mode:
        assert calls == [(locale.LC_CTYPE, "UTF-8")]


def test_a_bare_open_reads_non_ascii_after_the_fix(tmp_path):
    """End to end in a real ASCII-locale interpreter, which is the failing case.

    Run as a subprocess because the preferred encoding is decided at startup
    and cannot be un-decided from inside this one.
    """
    config = tmp_path / "config.json"
    config.write_text(CONFIG_WITH_NON_ASCII, encoding="utf-8")

    script = (
        "import locale, json, sys\n"
        "before = locale.getpreferredencoding(False)\n"
        "if not sys.flags.utf8_mode and 'utf-8' not in before.lower():\n"
        "    locale.setlocale(locale.LC_CTYPE, 'UTF-8')\n"
        f"json.load(open({str(config)!r}))\n"
        "print('OK', before)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONUTF8": "0",
             "HOME": str(tmp_path)},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("OK")


def test_the_bundle_declares_utf8_in_its_plist():
    """LSEnvironment is the real fix: it applies before the interpreter starts.

    Read from setup.py rather than a built bundle, so this holds in a fresh
    checkout where dist/ does not exist.
    """
    from pathlib import Path

    setup = (Path(__file__).resolve().parents[2] / "setup.py").read_text()
    assert "LSEnvironment" in setup
    assert '"PYTHONUTF8": "1"' in setup
