import textwrap

import pytest

from rpi_smf.data.holdings import listed_equities, parse_fidelity_csv

SAMPLE = textwrap.dedent("""\
    Security Name Market Value Assets Ticker CUSIP ISIN SEDOL,,,
    Domestic Equities,,,
    NVIDIA Corp $6,924,826,518 18.737% NVDA 67066G104 US67066G1040 2379504
    Marvell Technology Inc $1,35,516,283 2.802% MRVL 573874104 US5738741041 BNKJSM5
    Amazon.com Inc $940,760,172 2.546% AMZN 023135106 US0231351067 2000019,
    Celestial AI Inc escrow shares $10 0.000% CELES.PP,,,
    Domestic Equities Total: $35,137,49,356 95.1%
    Developed Markets,,,
    1037294.1.0 Fidelity Distributors Company, LLC,,
    ASML Holding NV depository receipt $281,792,062 0.762% ASML USN070592100 B908F01,
    Vast Data Ltd Series A2 $18,831,774 0.051% VDL+A2PP,
    Emerging Markets,,,
    MediaTek Inc $342,461,146 0.927% *MDTK TW0002454006 6372480,
    Bonds,,,
    Tenstorrent Holdings Inc 15% 12/31/2026 $3,622,554 0.010% %THIPP,
    Cash and Other Securities,,,
    Fidelity Cash Central Fund $147 0.000% CF.TAXMM 31635A105 US31635A1051,,,
""")


def test_parse_kinds_and_weights(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text(SAMPLE)
    df = parse_fidelity_csv(p).set_index("ticker")
    assert df.loc["NVDA", "weight"] == pytest.approx(0.18737)  # uses % column, not corrupted market value
    assert df.loc["MRVL", "weight"] == pytest.approx(0.02802)
    assert df.loc["CELES.PP", "kind"] == "private"
    assert df.loc["VDL+A2PP", "kind"] == "private"
    assert df.loc["*MDTK", "kind"] == "private"
    assert df.loc["%THIPP", "kind"] == "bond"
    assert df.loc["CF.TAXMM", "kind"] == "cash"
    assert df.loc["ASML", "section"] == "developed_markets"
    assert "Distributors" not in " ".join(df["name"])


def test_listed_equities_overrides(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text(SAMPLE)
    h = listed_equities(parse_fidelity_csv(p), overrides={"ASML": "ASML.AS"}, exclude=["AMZN"])
    assert set(h.index) == {"NVDA", "MRVL", "ASML.AS"}
    assert h.loc["ASML.AS", "fidelity_ticker"] == "ASML"


def test_real_file_parses():
    from pathlib import Path
    f = Path(__file__).parents[1] / "data/raw/fsptx_holdings_2026-10-07.csv"
    df = parse_fidelity_csv(f)
    listed = df[df.kind == "listed"]
    assert len(listed) > 60
    assert 0.9 < listed.weight.sum() < 1.0
    assert {"NVDA", "AAPL", "MSFT"} <= set(listed.ticker)
