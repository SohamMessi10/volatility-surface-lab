"""Optional live-data fetch: build a :class:`~volsurface.market.MarketData` from Yahoo options.

What it does: for a symbol (default ``SPY``) it reads the listed expiries and option chains via
``yfinance``, cleans the quotes, and returns the same OTM-quote container used by the synthetic
pipeline (``true_price`` and ``iv_true`` are ``nan``, ``heston`` is ``None``).

Forward and discount factor (the Cboe VIX-methodology approach): the discount factor comes
from a risk-free ``rate`` (``df = exp(-rate T)``), and the forward from put–call parity at the
money: among strikes quoted on both sides, take the three where ``|C - P|`` (mid prices) is
smallest and use the median of ``K + (C - P) / df``. No dividend assumption is needed. Near the
money, American early-exercise premia are small, so this is robust for American-style equity
options such as SPY. A least-squares regression of ``C - P`` on ``K`` over a wide strike window
was tried first and rejected: on live SPY chains the in-the-money puts' early-exercise premium
bent the regression to discount factors above 1 (a negative rate). If no strike is quoted on
both sides, the expiry falls back to ``F = S exp(rate T)`` (method ``"fallback"``). Only OTM
options are used for implied vols, which keeps the early-exercise bias out of the smile.

Conventions: ``T`` = calendar days to expiry / 365, days counted from ``now.date()`` to the
expiry date. Quotes need finite values, ``ask >= bid``, ``bid >= min_bid`` (default 0.05) and a
spread of at most ``max_rel_spread`` (default 0.5) of the mid: penny-wide deep-OTM quotes such
as 0.02 x 0.03 carry almost no information about the implied vol. The OTM option per
strike is used (put if ``K < F`` else call). ``n_dropped`` counts OTM-side quotes that were
unusable (bad quote, or no implied vol for the mid). ``noise`` is ``nan`` (unknown; written as
``null`` in JSON) and ``seed`` is None. ``spec.r`` is ``rate``; ``spec.q`` is 0 only as a
placeholder; the per-quote ``forward``/``discount_factor`` are authoritative.

Terms: the Yahoo! Finance API is intended for personal use only; yfinance is not affiliated with
Yahoo. Data fetched with this module is never committed: default output paths live under
``data/``, which is git-ignored.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .implied_vol import implied_vol_array
from .market import MarketData, MarketSpec, Quote

__all__ = ["fetch_market", "fetch_market_with_info"]

_ATM_PARITY_STRIKES = 3  # strikes with the smallest |C - P| used for the forward


def _col(frame: Any, name: str) -> NDArray:
    return np.asarray(frame[name], dtype=float)


def _clean(
    frame: Any, min_bid: float = 0.05, max_rel_spread: float = 0.5
) -> dict[float, tuple[float, float]]:
    """Map strike -> (bid, ask) for usable quotes.

    Usable: finite, ``bid > 0``, ``bid >= min_bid``, ``ask >= bid`` and
    ``(ask - bid) <= max_rel_spread * mid``.
    """
    K, bid, ask = _col(frame, "strike"), _col(frame, "bid"), _col(frame, "ask")
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(K) & np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
        ok &= bid >= min_bid
        ok &= (ask - bid) <= max_rel_spread * 0.5 * (ask + bid)
    return {float(k): (float(b), float(a)) for k, b, a in zip(K[ok], bid[ok], ask[ok], strict=True)}


def _atm_forward(
    calls: dict[float, tuple[float, float]],
    puts: dict[float, tuple[float, float]],
    df: float,
) -> float | None:
    """Forward from put–call parity near the money: median of ``K + (C - P) / df``.

    Uses the strikes quoted on both sides with the smallest ``|C - P|`` (mid prices).
    """
    two_sided = calls.keys() & puts.keys()
    if not two_sided:
        return None
    diff = {k: 0.5 * sum(calls[k]) - 0.5 * sum(puts[k]) for k in two_sided}
    nearest = sorted(two_sided, key=lambda k: abs(diff[k]))[:_ATM_PARITY_STRIKES]
    F = float(np.median([k + diff[k] / df for k in nearest]))
    return F if math.isfinite(F) and F > 0 else None


def _select_expiries(
    listed: Sequence[tuple[str, int]], tenors: Sequence[int] | None, max_expiries: int
) -> list[tuple[str, int]]:
    """Expiries to use: the first ``max_expiries`` by date, or the nearest to each target tenor.

    ``listed`` holds ``(date, days)`` pairs already filtered to the allowed day range. With
    ``tenors`` (days), each target picks the listed expiry closest to it (ties: the earlier);
    duplicates are dropped, so fewer expiries than targets can come back.
    """
    ordered = sorted(listed, key=lambda x: x[1])
    if not tenors:
        return ordered[:max_expiries]
    picked = {min(ordered, key=lambda x: (abs(x[1] - target), x[1])) for target in tenors}
    return sorted(picked, key=lambda x: x[1]) if ordered else []


def _spot(ticker: Any) -> float:
    try:
        s = float(ticker.fast_info["lastPrice"])
    except Exception:
        s = math.nan
    if not (math.isfinite(s) and s > 0):
        s = float(np.asarray(ticker.history(period="5d")["Close"], dtype=float)[-1])
    if not (math.isfinite(s) and s > 0):
        raise ValueError("could not determine a positive spot price")
    return s


def fetch_market_with_info(
    symbol: str = "SPY",
    *,
    max_expiries: int = 8,
    min_days: int = 7,
    max_days: int = 730,
    rate: float = 0.04,
    tenors: Sequence[int] | None = None,
    min_bid: float = 0.05,
    max_rel_spread: float = 0.5,
    retries: int = 3,
    now: datetime | None = None,
    ticker_factory: Callable[[str], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[MarketData, dict]:
    """Fetch quotes; return ``(market, info)`` with ``info[expiry_date]`` diagnostics.

    ``rate`` is the continuously compounded risk-free rate used for discounting.
    ``tenors`` (days, e.g. ``(14, 30, 60, 91, 182, 365)``) selects the listed expiry nearest to
    each target, spreading the slices across the surface; without it the first
    ``max_expiries`` expiries are used (for SPY these are mostly weeklies).
    When Yahoo rate-limits, ``ticker.options`` can come back empty; the request is retried up
    to ``retries`` times with exponential backoff (5, 10, 20 s) before giving up.
    Each info entry has ``T``, ``F``, ``df``, ``method`` (``"parity-atm"``/``"fallback"``) and
    ``n_quotes``. ``ticker_factory(symbol)`` must return a yfinance-like ticker (``.options``,
    ``.option_chain(date)`` with ``.calls``/``.puts``, ``.fast_info``); default is
    ``yfinance.Ticker``.
    """
    if ticker_factory is None:
        try:
            import yfinance
        except ImportError as exc:
            raise ImportError(
                "fetching live data requires yfinance; "
                'install it with pip install "volsurface[fetch]"'
            ) from exc
        ticker_factory = yfinance.Ticker
    now = datetime.now() if now is None else now
    ticker = ticker_factory(symbol)
    spot = _spot(ticker)

    options = tuple(ticker.options)
    for attempt in range(retries):
        if options:
            break
        sleep(5.0 * 2**attempt)
        ticker = ticker_factory(symbol)
        options = tuple(ticker.options)
    if not options:
        raise ValueError(
            f"Yahoo returned no option expiries for {symbol!r} after {retries + 1} attempts "
            "(rate-limited or unknown symbol?); wait a minute and try again"
        )

    listed: list[tuple[str, int]] = []
    for date in options:
        days = (datetime.strptime(str(date), "%Y-%m-%d").date() - now.date()).days
        if min_days <= days <= max_days:
            listed.append((str(date), days))
    chosen = _select_expiries(listed, tenors, max_expiries)

    rows: list[Quote] = []
    info: dict[str, dict] = {}
    dropped = 0
    for date, days in chosen:
        T = days / 365.0
        chain = ticker.option_chain(date)
        if chain is None or chain.calls is None or chain.puts is None:
            info[date] = {"T": T, "F": math.nan, "df": math.nan, "method": "no data", "n_quotes": 0}
            continue
        calls = _clean(chain.calls, min_bid, max_rel_spread)
        puts = _clean(chain.puts, min_bid, max_rel_spread)
        df = math.exp(-rate * T)
        F_atm = _atm_forward(calls, puts, df)
        if F_atm is None:
            F, method = spot * math.exp(rate * T), "fallback"
        else:
            F, method = F_atm, "parity-atm"
        all_strikes = set(_col(chain.calls, "strike")) | set(_col(chain.puts, "strike"))
        strikes = sorted(k for k in all_strikes if math.isfinite(k) and k > 0)
        K_l, bid_l, ask_l, call_l = [], [], [], []
        for k in strikes:
            is_call = k >= F
            q = (calls if is_call else puts).get(float(k))
            if q is None:
                dropped += 1
                continue
            K_l.append(k)
            bid_l.append(q[0])
            ask_l.append(q[1])
            call_l.append(is_call)
        n_before = len(rows)
        if K_l:
            K, bid, ask = np.array(K_l), np.array(bid_l), np.array(ask_l)
            is_call_a = np.array(call_l)
            mid = 0.5 * (bid + ask)
            ivs = [
                implied_vol_array(p, F, K, T, df, is_call_a, errors="nan") for p in (mid, bid, ask)
            ]
            for i in range(K.size):
                if math.isnan(ivs[0][i]):
                    dropped += 1
                    continue
                rows.append(
                    Quote(
                        expiry=T,
                        strike=float(K[i]),
                        forward=F,
                        discount_factor=df,
                        log_moneyness=math.log(K[i] / F),
                        option_type="call" if is_call_a[i] else "put",
                        true_price=math.nan,
                        bid=float(bid[i]),
                        ask=float(ask[i]),
                        mid=float(mid[i]),
                        iv_true=math.nan,
                        iv_bid=float(ivs[1][i]),
                        iv_ask=float(ivs[2][i]),
                        iv_mid=float(ivs[0][i]),
                    )
                )
        n_q = len(rows) - n_before
        info[date] = {"T": T, "F": F, "df": df, "method": method, "n_quotes": n_q}

    if not rows:
        raise ValueError(f"no usable option quotes found for {symbol!r}")
    expiries = tuple(sorted({q.expiry for q in rows}))
    moneyness = tuple(sorted({round(q.strike / spot, 10) for q in rows}))
    spec = MarketSpec(spot=spot, r=rate, q=0.0, expiries=expiries, strike_moneyness=moneyness)
    market = MarketData(spec, None, math.nan, None, dropped, rows)
    return market, info


def fetch_market(
    symbol: str = "SPY",
    *,
    max_expiries: int = 8,
    min_days: int = 7,
    max_days: int = 730,
    rate: float = 0.04,
    tenors: Sequence[int] | None = None,
    min_bid: float = 0.05,
    max_rel_spread: float = 0.5,
    retries: int = 3,
    now: datetime | None = None,
    ticker_factory: Callable[[str], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> MarketData:
    """Like :func:`fetch_market_with_info` but returns only the market."""
    return fetch_market_with_info(
        symbol,
        max_expiries=max_expiries,
        min_days=min_days,
        max_days=max_days,
        rate=rate,
        tenors=tenors,
        min_bid=min_bid,
        max_rel_spread=max_rel_spread,
        retries=retries,
        now=now,
        ticker_factory=ticker_factory,
        sleep=sleep,
    )[0]
