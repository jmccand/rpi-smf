import json

import pandas as pd

from rpi_smf.journal import append_journal, sha256, write_run


def test_write_run_and_journal(tmp_path):
    inp = tmp_path / "in.csv"; inp.write_text("a,b\n1,2\n")
    run = write_run(tmp_path / "runs", "2026-10-07", {"targets": pd.DataFrame({"w": [1.0]}, index=["A"])},
                    {"lambda": 3.2}, input_files=[inp, tmp_path / "missing.csv"])
    m = json.loads((run / "manifest.json").read_text())
    assert m["lambda"] == 3.2 and m["input_hashes"]["in.csv"] == sha256(inp)
    assert m["input_hashes"]["missing.csv"] is None
    assert (run / "targets.csv").exists()
    append_journal(tmp_path / "runs", {"date": "2026-10-07", "lam": 3.2})
    append_journal(tmp_path / "runs", {"date": "2026-10-14", "lam": 2.0})
    assert len(pd.read_csv(tmp_path / "runs" / "journal.csv")) == 2
