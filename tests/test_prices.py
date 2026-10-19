import pytest
import pandas as pd

from rpi_smf.data.prices import get_prices, get_risk_free


def _fake(tickers):
    idx = pd.bdate_range("2025-01-01", periods=10)
    return pd.DataFrame({t: range(1, 11) for t in tickers if t != "BAD"}, index=idx, dtype=float)


def test_prices_cache_asof_and_missing(tmp_path):
    calls = []

    def dl(ts):
        calls.append(list(ts))
        return _fake(ts)

    df = get_prices(["A", "BAD", "B"], asof=pd.Timestamp("2025-01-07"), cache_dir=tmp_path,
                    downloader=dl)
    assert list(df.columns) == ["A", "B"]
    assert df.index.max() == pd.Timestamp("2025-01-07")      # no look-ahead
    get_prices(["A", "B"], cache_dir=tmp_path, downloader=dl)
    assert calls == [["A", "BAD", "B"]]                        # second call served from cache


def test_risk_free_scaling_and_fallback(tmp_path):
    def reader(url, **kw):
        return pd.DataFrame({"DTB3": ["4.5", "."]},
                            index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    rf = get_risk_free(cache_dir=tmp_path, reader=reader)
    assert rf.iloc[0] == 0.045 and len(rf) == 1                # '.' (holiday) dropped


def test_risk_free_raises_when_fred_unavailable(tmp_path):
    def reader(url, **kw):
        raise OSError("offline")
    with pytest.raises(RuntimeError, match="FRED"):
        get_risk_free(cache_dir=tmp_path, reader=reader)
