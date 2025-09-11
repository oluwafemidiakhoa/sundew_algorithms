# tests/test_cli_save.py (full)

from __future__ import annotations

import json
import sys
from pathlib import Path

from sundew.cli import main


def test_cli_save_creates_json(tmp_path: Path, monkeypatch) -> None:
    """Ensure `sundew` CLI `--save` writes a valid JSON file."""
    out = tmp_path / "demo_results.json"

    # Simulate: sundew --demo --events 10 --save <path>
    argv = ["sundew", "--demo", "--events", "10", "--save", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    # Run CLI entrypoint
    main()

    # Validate output
    assert out.is_file(), "CLI should create the requested JSON file"
    with out.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    # Be lenient on schema; just check some expected top-level structure.
    # The report typically includes counts / report keys; not strictly enforced here.
    assert data, "Saved JSON should not be empty"
