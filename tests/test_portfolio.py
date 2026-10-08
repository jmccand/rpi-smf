import pandas as pd
import pytest

from rpi_smf.portfolio import (Portfolio, initial_portfolio, load_history, load_state, nav_series,
                               save_state)


def test_initial_apply_roundtrip(tmp_path):
    px = pd.Series({"A": 10.0, "B": 20.0})
    pf = initial_portfolio(pd.Series({"A": 0.6, "B": 0.4}), px, 1000.0, "2026-09-28")
    assert pf.value(px) == pytest.approx(1000.0)
    assert pf.weights(px)["A"] == pytest.approx(0.6)
    trades = pd.DataFrame({"shares": [-10.0, 5.0]}, index=["A", "B"])    # sell $100 A, buy $100 B
    pf2 = pf.apply(trades, "2026-10-07", 1000.0, total_cost=10.0)
    assert pf2.value(px) == pytest.approx(990.0)                          # cost scaled out of NAV
    with pytest.raises(ValueError):
        pf.apply(pd.DataFrame({"shares": [-1000.0]}, index=["A"]), "2026-10-07", 1000.0, 0.0)
    save_state(tmp_path, pf); save_state(tmp_path, pf2)
    assert load_state(tmp_path).date == pd.Timestamp("2026-10-07")
    h = load_history(tmp_path)
    assert list(h.index) == [pd.Timestamp("2026-09-28"), pd.Timestamp("2026-10-07")]
    prices = pd.DataFrame({"A": [10.0, 11, 12], "B": [20.0, 20, 22]},
                          index=pd.to_datetime(["2026-09-28", "2026-09-29", "2026-10-07"]))
    nav = nav_series(h, prices)
    assert nav.iloc[0] == pytest.approx(1000.0)
    assert nav.iloc[1] == pytest.approx(60 * 11 + 20 * 20)                # old holdings until the rebalance
    assert nav.iloc[2] == pytest.approx(pf2.value(prices.iloc[2]))
