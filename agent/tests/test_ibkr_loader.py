"""Offline unit tests for the local TWS / IB Gateway OHLCV loader.

The connector boundary (``get_historical_bars`` / config / readiness probes)
is monkeypatched on the loader's namespace — no network, no ib_async install.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.loaders import ibkr_loader
from src.trading.connectors.ibkr.local import IBKRLocalConfig

pytestmark = pytest.mark.unit


def _bar(date: str, o=1.0, h=2.0, low=0.5, c=1.5, v=100.0) -> dict:
    return {"date": date, "open": o, "high": h, "low": low, "close": c, "volume": v}


class _FakeHistory:
    """Stand-in for the connector's ``get_historical_bars``.

    Records every request; serves canned bars per (bare) symbol. Symbols in
    ``raising`` blow up so the loader's per-symbol containment is exercised.
    """

    def __init__(
        self,
        bars_by_symbol: dict[str, list[dict]] | None = None,
        raising: set[str] | None = None,
        status: str = "ok",
    ) -> None:
        self.calls: list[dict] = []
        self.bars_by_symbol = bars_by_symbol or {}
        self.raising = raising or set()
        self.status = status

    def __call__(
        self,
        symbol: str,
        *,
        config=None,
        duration: str,
        bar_size: str,
        end_date_time: str,
    ) -> dict:
        self.calls.append(
            {
                "symbol": symbol,
                "duration": duration,
                "bar_size": bar_size,
                "end": end_date_time,
            }
        )
        if symbol in self.raising:
            raise RuntimeError(f"gateway hiccup for {symbol}")
        if self.status != "ok":
            return {"status": self.status, "error": "boom"}
        bars = self.bars_by_symbol.get(
            symbol, [_bar("20260518"), _bar("20260519"), _bar("20260520")]
        )
        return {"status": "ok", "bars": bars}


@pytest.fixture()
def ready(monkeypatch: pytest.MonkeyPatch) -> _FakeHistory:
    fake = _FakeHistory()
    monkeypatch.setattr(ibkr_loader, "get_historical_bars", fake)
    monkeypatch.setattr(
        ibkr_loader, "load_config", lambda: IBKRLocalConfig(host="127.0.0.1", port=4002)
    )
    monkeypatch.setattr(ibkr_loader, "ib_async_available", lambda: True)
    monkeypatch.setattr(ibkr_loader, "tcp_port_open", lambda h, p, timeout=0.5: True)
    return fake


def test_fetch_daily_frame_contract(ready: _FakeHistory) -> None:
    result = ibkr_loader.DataLoader().fetch(["AAPL.US"], "2026-05-11", "2026-05-20")

    assert list(result) == ["AAPL.US"]
    frame = result["AAPL.US"]
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.name == "trade_date"
    assert frame.index.tz is None
    assert frame.index.is_monotonic_increasing
    assert all(dtype == "float64" for dtype in frame.dtypes)
    assert frame.attrs["quote_currency"] == "USD"


def test_us_suffix_stripped_request_keyed_by_original(ready: _FakeHistory) -> None:
    result = ibkr_loader.DataLoader().fetch(["AAPL.US"], "2026-05-11", "2026-05-20")

    assert ready.calls[0]["symbol"] == "AAPL"
    assert "AAPL.US" in result


def test_request_window_pass_through(ready: _FakeHistory) -> None:
    ibkr_loader.DataLoader().fetch(["AAPL"], "2026-05-11", "2026-05-20")

    call = ready.calls[0]
    assert call["bar_size"] == "1 day"
    assert call["end"] == "20260520 23:59:59 UTC"
    # 10 calendar days + 7-day holiday pad
    assert call["duration"] == "17 D"


def test_unsupported_interval_rejected(ready: _FakeHistory) -> None:
    result = ibkr_loader.DataLoader().fetch(
        ["AAPL"], "2026-05-11", "2026-05-20", interval="1W"
    )

    assert result == {}
    assert ready.calls == []


def test_window_clipping(ready: _FakeHistory) -> None:
    ready.bars_by_symbol["AAPL"] = [
        _bar("20260515"),  # before the window
        _bar("20260518"),
        _bar("20260520"),  # inclusive end date
        _bar("20260521"),  # after the window
    ]
    result = ibkr_loader.DataLoader().fetch(["AAPL"], "2026-05-18", "2026-05-20")

    assert list(result["AAPL"].index) == [
        pd.Timestamp("2026-05-18"),
        pd.Timestamp("2026-05-20"),
    ]


def test_failed_symbol_omitted_never_raises(ready: _FakeHistory) -> None:
    ready.raising = {"MSFT"}

    result = ibkr_loader.DataLoader().fetch(
        ["MSFT", "AAPL"], "2026-05-11", "2026-05-20"
    )

    assert list(result) == ["AAPL"]


def test_error_status_omits_symbol(ready: _FakeHistory) -> None:
    ready.status = "error"

    result = ibkr_loader.DataLoader().fetch(["AAPL"], "2026-05-11", "2026-05-20")

    assert result == {}


def test_empty_bars_omitted(ready: _FakeHistory) -> None:
    ready.bars_by_symbol["AAPL"] = []

    result = ibkr_loader.DataLoader().fetch(["AAPL"], "2026-05-11", "2026-05-20")

    assert result == {}


def test_intraday_timezone_suffix_parsed_naive(ready: _FakeHistory) -> None:
    ready.bars_by_symbol["AAPL"] = [_bar("20260519 09:30:00 US/Eastern")]

    result = ibkr_loader.DataLoader().fetch(
        ["AAPL"], "2026-05-18", "2026-05-20", interval="1H"
    )

    assert list(result["AAPL"].index) == [pd.Timestamp("2026-05-19 09:30:00")]
    assert result["AAPL"].index.tz is None


def test_is_available_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = ibkr_loader.DataLoader()

    monkeypatch.setattr(ibkr_loader, "load_config", lambda: IBKRLocalConfig(port=4002))
    monkeypatch.setattr(ibkr_loader, "tcp_port_open", lambda h, p, timeout=0.5: True)

    monkeypatch.setattr(ibkr_loader, "ib_async_available", lambda: False)
    assert loader.is_available() is False

    monkeypatch.setattr(ibkr_loader, "ib_async_available", lambda: True)
    monkeypatch.setattr(ibkr_loader, "tcp_port_open", lambda h, p, timeout=0.5: False)
    assert loader.is_available() is False

    monkeypatch.setattr(ibkr_loader, "tcp_port_open", lambda h, p, timeout=0.5: True)
    assert loader.is_available() is True

    def _broken_config() -> None:
        raise ValueError("corrupt json")

    monkeypatch.setattr(ibkr_loader, "load_config", _broken_config)
    assert loader.is_available() is False


def test_registered_in_registry() -> None:
    from backtest.loaders.registry import (
        LOADER_REGISTRY,
        VALID_SOURCES,
        _ensure_registered,
    )

    _ensure_registered()
    assert "ibkr" in VALID_SOURCES
    assert "ibkr" in LOADER_REGISTRY
