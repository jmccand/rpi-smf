from rpi_smf.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.start_date.isoformat() == "2026-09-28"
    assert cfg.initial_capital == 1_000_000
    assert [f.name for f in cfg.factors][-1] == "volatility"
    assert cfg.factors[-1].expected_sign == -1
    assert cfg.section("optimizer")["single_name_cap"] == 0.08
