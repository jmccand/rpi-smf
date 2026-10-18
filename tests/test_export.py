import pandas as pd

from rpi_smf.export import export_post_trade


def test_export_post_trade(tmp_path):
    run = tmp_path / "2026-10-05"
    run.mkdir()
    pd.DataFrame({"current": [0.5, 0.5, 0.0, 0.0], "post_trade": [0.6, 0.0, 0.4, 0.0]},
                 index=["B", "A", "C", "Z"]).to_csv(run / "weights.csv")
    out = export_post_trade(run, tmp_path / "out.csv")
    assert out.read_text().splitlines() == [
        "DES,A,STK,SMART/AMEX,,,,,,0.000000",
        "DES,B,STK,SMART/AMEX,,,,,,60.000000",
        "DES,C,STK,SMART/AMEX,,,,,,40.000000",
    ]
