"""Tests for the optional fetch module using a fake ticker (no network, no yfinance)."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from volsurface.black_scholes import bs_price
from volsurface.fetch import fetch_market, fetch_market_with_info
from volsurface.market import load_market

NOW = datetime(2026, 1, 1, 12, 0)
SPOT = 100.0
STRIKES = np.arange(80.0, 120.1, 2.5)
HALF = 0.01
# days -> (r, q) used to build the model forward and discount factor
RATES = {
    3: (0.04, 0.01),
    30: (0.04, 0.01),
    90: (0.045, 0.012),
    400: (0.05, 0.015),
    800: (0.05, 0.0),
}


def model_vol(K, F):
    k = np.log(K / F)
    return 0.2 - 0.1 * k + 0.3 * k**2


def model_fd(days):
    r, q = RATES[days]
    T = days / 365.0
    return SPOT * math.exp((r - q) * T), math.exp(-r * T), T


def make_frames(days, *, mutate=None):
    F, df, T = model_fd(days)
    sig = model_vol(STRIKES, F)
    out = {}
    for name, is_call in (("calls", True), ("puts", False)):
        p = bs_price(F, STRIKES, T, sig, df, is_call)
        out[name] = {
            "strike": list(STRIKES),
            "bid": list(np.maximum(p - HALF, 0.0)),
            "ask": list(p + HALF),
            "lastPrice": list(p),
        }
    if mutate:
        mutate(out)
    return out


def make_ticker(days_list=(3, 30, 90, 400, 800), mutate=None):
    dates = {(NOW + timedelta(days=d)).strftime("%Y-%m-%d"): d for d in days_list}

    def option_chain(date):
        fr = make_frames(dates[date], mutate=mutate.get(dates[date]) if mutate else None)
        return SimpleNamespace(calls=fr["calls"], puts=fr["puts"])

    return SimpleNamespace(
        options=tuple(dates), option_chain=option_chain, fast_info={"lastPrice": SPOT}
    )


def run(**kw):
    t = kw.pop("ticker", None) or make_ticker()
    return fetch_market_with_info("TEST", now=NOW, ticker_factory=lambda s: t, **kw)


def test_expiry_filtering_and_order():
    market, info = run()
    days = sorted(round(v["T"] * 365) for v in info.values())
    assert days == [30, 90, 400]  # 3 < min_days, 800 > max_days
    assert market.spec.expiries == tuple(d / 365 for d in days)
    _, info2 = run(max_expiries=2)
    assert sorted(round(v["T"] * 365) for v in info2.values()) == [30, 90]
    _, info3 = run(min_days=0, max_days=100)
    assert sorted(round(v["T"] * 365) for v in info3.values()) == [3, 30, 90]


def test_tenor_selection_picks_nearest_listed_expiries():
    # Listed (within 7..730 days): 30, 90, 400.
    _, info = run(tenors=(25, 100, 380))
    assert sorted(round(v["T"] * 365) for v in info.values()) == [30, 90, 400]
    # Targets that map to the same expiry are de-duplicated; tenors override max_expiries.
    _, info = run(tenors=(28, 31, 35), max_expiries=1)
    assert [round(v["T"] * 365) for v in info.values()] == [30]
    # Targets outside the allowed range still pick the nearest allowed expiry.
    _, info = run(tenors=(1, 5000))
    assert sorted(round(v["T"] * 365) for v in info.values()) == [30, 400]


def test_atm_parity_recovers_forward():
    # The model's own rate differs from `rate` at 90 and 400 days (4.5%, 5%); near the money
    # that only enters through (F - K) * (df_true / df - 1), so the forward stays accurate.
    _, info = run(rate=0.04)
    errs_F = []
    for d in info.values():
        days = round(d["T"] * 365)
        F, _, T = model_fd(days)
        assert d["method"] == "parity-atm"
        assert d["df"] == pytest.approx(math.exp(-0.04 * T), rel=1e-15)
        errs_F.append(abs(d["F"] / F - 1))
    assert max(errs_F) < 2e-4


def test_forward_robust_to_american_early_exercise_premium():
    # Live SPY options are American: in-the-money puts trade above their European value.
    # Add such a premium (growing with depth) to the fake puts. A C - P regression on K over a
    # wide window then implies a discount factor above 1 (seen on live SPY chains); the ATM
    # parity estimate must still recover the forward.
    def mutate(fr):
        F, _, _ = model_fd(90)
        for i, k in enumerate(STRIKES):
            if k > F:
                premium = 0.05 * (k - F)  # 0.5 on a 10-point-ITM put (upper bound ~K r T = 1.2)
                fr["puts"]["bid"][i] += premium
                fr["puts"]["ask"][i] += premium

    _, info = run(ticker=make_ticker((90,), {90: mutate}))
    (d,) = info.values()
    F, _, _ = model_fd(90)
    assert d["method"] == "parity-atm"
    assert abs(d["F"] / F - 1) < 5e-4
    # The rejected approach on the same quotes: slope of C - P against K over +-10%.
    t = make_ticker((90,), {90: mutate})
    ch = t.option_chain(_date(90))
    K = np.array(ch.calls["strike"])
    c_mid = 0.5 * (np.array(ch.calls["bid"]) + np.array(ch.calls["ask"]))
    p_mid = 0.5 * (np.array(ch.puts["bid"]) + np.array(ch.puts["ask"]))
    sel = np.abs(K / SPOT - 1) <= 0.10
    slope = np.polyfit(K[sel], (c_mid - p_mid)[sel], 1)[0]
    _, df_true, _ = model_fd(90)
    assert -slope > df_true + 0.01  # biased upward by the premium...
    assert -slope > 1.0  # ...to an implied df above 1, i.e. a negative interest rate


def test_iv_mid_matches_model_and_otm_selection():
    market, info = run()
    assert market.heston is None
    assert math.isnan(market.noise)
    assert market.seed is None
    for q in market.rows:
        days = round(q.expiry * 365)
        F, _, _ = model_fd(days)
        assert q.option_type == ("put" if q.strike < q.forward else "call")
        assert q.forward == pytest.approx(info[_date(days)]["F"])
        assert math.isnan(q.true_price)
        assert math.isnan(q.iv_true)
        assert q.log_moneyness == pytest.approx(math.log(q.strike / q.forward))
        assert q.iv_bid < q.iv_mid < q.iv_ask
        # mids are model prices at the true F, df; implied F/df errors perturb the vol slightly
        assert abs(q.iv_mid - model_vol(q.strike, F)) < 5e-3
    assert market.spec.spot == SPOT
    assert list(market.spec.strike_moneyness) == sorted(set(market.spec.strike_moneyness))


def _date(days):
    return (NOW + timedelta(days=days)).strftime("%Y-%m-%d")


def test_bad_quotes_dropped():
    def mutate(fr):
        # strike 105 call: bid 0; strike 90 put: crossed; strike 85 put: nan
        i105, i90, i85 = (list(STRIKES).index(k) for k in (105.0, 90.0, 85.0))
        fr["calls"]["bid"][i105] = 0.0
        fr["puts"]["bid"][i90], fr["puts"]["ask"][i90] = 5.0, 4.0
        fr["puts"]["ask"][i85] = float("nan")

    # Disable the min-bid / spread filters so only the bad-quote rules are exercised.
    no_filters = {"min_bid": 0.0, "max_rel_spread": math.inf}
    clean, _ = run(**no_filters)
    market, _ = run(ticker=make_ticker((30,), {30: mutate}), **no_filters)
    strikes = {q.strike for q in market.rows}
    assert not strikes & {105.0, 90.0, 85.0}
    assert market.n_dropped >= 3
    assert strikes <= {q.strike for q in clean.rows}
    assert len(strikes) >= 8


def test_min_bid_and_spread_filters():
    def mutate(fr):
        i100, i110 = (list(STRIKES).index(k) for k in (100.0, 110.0))
        fr["calls"]["bid"][i110] = 0.03  # positive but below min_bid
        fr["calls"]["ask"][i110] = 0.04
        mid = 0.5 * (fr["calls"]["bid"][i100] + fr["calls"]["ask"][i100])
        fr["calls"]["bid"][i100] = 0.6 * mid  # spread 0.8 * mid > 0.5 * mid
        fr["calls"]["ask"][i100] = 1.4 * mid

    market, _ = run(ticker=make_ticker((30,), {30: mutate}))
    strikes = {q.strike for q in market.rows if q.option_type == "call"}
    assert 110.0 not in strikes
    assert 100.0 not in strikes
    loose, _ = run(ticker=make_ticker((30,), {30: mutate}), min_bid=0.0, max_rel_spread=1.0)
    assert {100.0, 110.0} <= {q.strike for q in loose.rows}
    for q in market.rows:
        assert q.bid >= 0.05
        assert q.ask - q.bid <= 0.5 * q.mid + 1e-12


def test_fallback_when_no_strike_is_quoted_on_both_sides():
    def mutate(fr):
        for i in range(len(STRIKES)):
            fr["puts"]["bid"][i] = 0.0

    market, info = run(ticker=make_ticker((30,), {30: mutate}), rate=0.03)
    (d,) = info.values()
    T = 30 / 365
    assert d["method"] == "fallback"
    assert d["F"] == pytest.approx(SPOT * math.exp(0.03 * T))
    assert d["df"] == pytest.approx(math.exp(-0.03 * T))
    assert d["n_quotes"] == len(market.rows) > 0


def test_no_quotes_raises():
    t = make_ticker((3,))
    with pytest.raises(ValueError, match="no usable"):
        run(ticker=t)


def test_fetch_market_returns_market():
    m = fetch_market("TEST", now=NOW, ticker_factory=lambda s: make_ticker())
    assert len(m.rows) > 0


def test_import_error_without_yfinance(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match=r"volsurface\[fetch\]"):
        fetch_market("SPY")


def test_json_roundtrip(tmp_path):
    market, _ = run()
    path = tmp_path / "m.json"
    market.to_json(path)
    back = load_market(path)
    assert back.spec == market.spec
    assert len(back.rows) == len(market.rows)
    assert back.rows[0].strike == market.rows[0].strike
    assert back.rows[0].iv_mid == pytest.approx(market.rows[0].iv_mid)
    assert math.isnan(back.rows[0].true_price)
    assert math.isnan(back.noise)
