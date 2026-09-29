"""IBKR historical OHLCV loader — local TWS / IB Gateway via ib_async.

Feeds backtests and ``get_market_data`` from the user's own Interactive
Brokers session, with that account's market-data subscriptions (e.g. a paid
US-equity bundle). Requires the optional ``ib_async`` package (``pip install
"vibe-trading-ai[ibkr]"``) and a running, logged-in TWS or IB Gateway with
API socket clients enabled — when either is missing ``is_available`` is
``False`` and the ``us_equity`` fallback chain degrades to yahoo/stooq/...

Config is shared with the IBKR trading connector (``~/.vibe-trading/
ibkr-local.json``, managed by ``vibe-trading connector configure
ibkr-paper-local``). The connector module owns the socket pool (thread-local
connections, unique client ids, read-only enforcement, paper/live profile
assertion); this loader is a thin adapter over its ``get_historical_bars``
(india_broker/longbridge precedent for loaders reusing connector plumbing).

IBKR serves unadjusted historical TRADES, so the price caliber is ``raw``.
IBKR caps historical pacing per account (~60 requests / 10 min, shared with
the trading connector on the same gateway), which is why the source sits
behind the zero-setup public endpoints in the default chain — put it first
via ``MARKET_DATA_ORDER_US_EQUITY`` when the local feed should win.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.loaders.base import (
    cached_loader_fetch,
    validate_date_range,
    validate_ohlc,
)
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

# Public connector surface (no underscore names): config + readiness probes +
# the historical read itself. ib_async is imported lazily inside the connector,
# so importing it here stays cheap and never hard-requires the SDK.
from src.trading.connectors.ibkr.local import (  # noqa: E402
    IBKRLocalConfig,
    get_historical_bars,
    ib_async_available,
    load_config,
    tcp_port_open,
)

#: Canonical interval token → IBKR ``barSizeSetting``. ``1m`` (minute) and
#: ``1M`` (month) differ by case; weekly/monthly are never requested anyway
#: (the orchestrator resamples them from dailies, #1479).
_INTERVAL_MAP = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1H": "1 hour",
    "1h": "1 hour",
    "4H": "4 hours",
    "4h": "4 hours",
    "1D": "1 day",
    "1d": "1 day",
}


def _strip_us_suffix(code: str) -> str:
    """``AAPL.US`` → ``AAPL``; other codes pass through uppercased."""
    token = code.strip().upper()
    if token.endswith(".US"):
        token = token[: -len(".US")]
    return token


def _duration_str(start_date: str, end_date: str) -> str:
    """Build an IBKR ``durationStr`` covering ``[start_date, end_date]``.

    IBKR counts duration *backward* from the request's end, so the window
    itself plus a small pad (exchange holidays at the head) is enough;
    over-fetch is discarded by the local clip in :meth:`_bars_to_frame`.
    """
    start = datetime.strptime(start_date[:10], "%Y-%m-%d")
    end = datetime.strptime(end_date[:10], "%Y-%m-%d")
    days = max((end - start).days + 1 + 7, 8)  # +7 holiday pad, min one week
    if days <= 366:
        return f"{days} D"
    years = min(math.ceil(days / 365), 10)
    return f"{years} Y"


def _parse_bar_date(raw: Any) -> pd.Timestamp:
    """Parse an IBKR bar date into a tz-naive timestamp.

    ``formatDate=1`` daily bars arrive as ``"20260529"``; intraday bars may
    carry an exchange-timezone suffix (``"... 09:30:00 US/Eastern"``) or a
    UTC offset, which are stripped/exchanged for their wall-clock time —
    the frame contract is a tz-naive index.
    """
    text = str(raw).strip()
    text = re.sub(r"\s+[A-Za-z][A-Za-z/_]*$", "", text)  # "US/Eastern" suffix
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts)


def _bars_to_frame(
    bars: List[Dict[str, Any]], start_date: str, end_date: str, bar_size: str
) -> Optional[pd.DataFrame]:
    """Map connector bar dicts to the loader frame contract."""
    if not bars:
        return None
    frame = pd.DataFrame(bars)
    if "date" not in frame.columns or "close" not in frame.columns:
        return None
    frame["trade_date"] = frame["date"].map(_parse_bar_date)
    frame = frame.dropna(subset=["trade_date"]).set_index("trade_date").sort_index()
    if bar_size == "1 day":
        frame.index = frame.index.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = float("nan")
    frame = frame[["open", "high", "low", "close", "volume"]].astype(float)
    # The end date is inclusive of its whole day (mt5 loader precedent).
    start = pd.Timestamp(start_date[:10])
    end = pd.Timestamp(end_date[:10]) + pd.Timedelta(days=1)
    frame = frame[(frame.index >= start) & (frame.index < end)]
    frame = frame.dropna()
    frame = validate_ohlc(frame)
    return frame if not frame.empty else None


@register
class DataLoader:
    """US-equity OHLCV from the local TWS / IB Gateway session (opt-in extra)."""

    name = "ibkr"
    markets = {"us_equity"}
    #: A running, logged-in gateway session is the auth surface (mt5/futu
    #: precedent) — no token, but not a public endpoint either.
    requires_auth = True
    volume_units = {"us_equity": "shares"}

    def __init__(self) -> None:  # never raises — registry availability contract
        pass

    def is_available(self) -> bool:
        """SDK importable AND the configured socket accepts connections."""
        if not ib_async_available():
            return False
        try:
            cfg = load_config()
        except ValueError as exc:
            logger.warning("ibkr: invalid local config: %s", exc)
            return False
        return tcp_port_open(cfg.host, cfg.port)

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV frames keyed by the original input codes.

        Per-symbol failures log and skip (never raise) so the orchestrator's
        chain fallback can engage for the missing symbols.
        """
        validate_date_range(start_date, end_date)
        del fields

        bar_size = _INTERVAL_MAP.get(str(interval).strip())
        if bar_size is None:
            # Reject unknown tokens; do not fetch daily bars under the caller's key.
            logger.warning("ibkr unsupported interval %r; rejecting", interval)
            return {}

        cfg = self._config()
        duration = _duration_str(start_date, end_date)
        # IBKR end format; UTC pin keeps the window independent of the TWS
        # display timezone. Duration counts backward from this end.
        end_date_time = f"{end_date[:10].replace('-', '')} 23:59:59 UTC"

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            clean = code.strip()
            if not clean:
                continue
            try:
                frame = cached_loader_fetch(
                    source=self.name,
                    symbol=clean,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    fetch=lambda c=clean: self._fetch_one(
                        c, cfg, duration, bar_size, end_date_time, start_date, end_date
                    ),
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - one symbol never poisons the batch
                logger.warning("ibkr: fetch failed for %s: %s", clean, exc)
                continue
            if frame is not None and not frame.empty:
                result[code] = frame
        return result

    @staticmethod
    def _config() -> IBKRLocalConfig:
        try:
            return load_config()
        except ValueError:
            return IBKRLocalConfig()

    def _fetch_one(
        self,
        code: str,
        cfg: IBKRLocalConfig,
        duration: str,
        bar_size: str,
        end_date_time: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch one symbol from the gateway (``None`` on any failure)."""
        payload = get_historical_bars(
            _strip_us_suffix(code),
            config=cfg,
            duration=duration,
            bar_size=bar_size,
            end_date_time=end_date_time,
        )
        if payload.get("status") != "ok":
            logger.warning(
                "ibkr: %s: %s", code, payload.get("error") or payload.get("status")
            )
            return None
        frame = _bars_to_frame(
            payload.get("bars") or [], start_date, end_date, bar_size
        )
        if frame is not None:
            frame.attrs["quote_currency"] = "USD"
        return frame
