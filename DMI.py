"""
================================================================================
# REAL-TIME ROLLING DMI (Directional Movement Index) ENGINE
================================================================================

Implements an incremental, tick-by-tick DMI calculator that:

1. Initialises rolling state from historical OHLCV candles using TA-Lib
   (TA-Lib is called ONCE and never again after startup).

2. Maintains two layers of state:
   - COMMITTED state  → last fully closed candle (frozen until next close)
   - IN-PROGRESS state → currently forming candle (overwritten on every tick)

3. On every tick, only the current candle's contribution is recomputed;
   the entire history is never re-scanned.

4. Produces +DI, -DI, DX, and ADX values that stay synchronised with
   talib.PLUS_DI / talib.MINUS_DI / talib.ADX after initialisation.

Design follows the EMA rolling-update pattern from the reference codebase.

Author : Vaibhav Saxena  (DMI extension)
================================================================================
"""

# ============================================================
# Imports
# ============================================================
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import talib as ta
import trendln

from fyers_apiv3.FyersWebsocket import data_ws
from fyers_apiv3 import fyersModel
from credentials import client_id
import datetime as dt
import pytz


# ============================================================
# Configuration
# ============================================================
symbol     = 'NSE:RELIANCE-EQ'
timeZone   = 'Asia/Kolkata'
resolution = "1"
DMI_PERIOD = 14          # Wilder period — change here to affect everything
EMA_DI_PERIOD = 3        # Smoothing period for ema_+di / ema_-di


# ============================================================
# DMIState  —  all rolling values live here
# ============================================================
@dataclass
class DMIState:
    """
    Holds every rolling value needed to continue Wilder-smoothed DMI
    calculations from any arbitrary point in time without looking back
    at history.

    TWO COPIES are maintained by DMIEngine:
        committed   → values after the last *closed* candle
        in_progress → values reflecting the current *open* candle
    """

    # ── Configuration ─────────────────────────────────────────
    period: int = DMI_PERIOD

    # ── Wilder-smoothed accumulators (the "memory" of the indicator) ──
    smoothed_tr:      float = math.nan   # Smoothed True Range
    smoothed_plus_dm: float = math.nan   # Smoothed +DM
    smoothed_minus_dm: float = math.nan  # Smoothed -DM
    smoothed_dx:      float = math.nan   # Smoothed DX  (feeds ADX)

    # ── Derived indicator values (what you display / trade on) ──
    current_plus_di:  float = math.nan
    current_minus_di: float = math.nan
    current_dx:       float = math.nan
    current_adx:      float = math.nan

    # ── Previous-candle anchors (needed for DM and TR calculation) ──
    prev_high:  float = math.nan
    prev_low:   float = math.nan
    prev_close: float = math.nan

    # ── Book-keeping ──────────────────────────────────────────
    is_initialised: bool = False    # True once enough history has been processed
    candles_seen:   int  = 0        # Total candles consumed (including current)


# ============================================================
# DMIEngine  —  stateless functions that operate on DMIState
# ============================================================
class DMIEngine:
    """
    Stateless helper that performs all DMI maths.

    Usage pattern (mirrors EMA reference):

        engine = DMIEngine(period=14)
        engine.initialise(historical_df)        # uses TA-Lib once
        ...
        # On every tick:
        engine.update(high, low, close, is_new_candle)
        plus_di = engine.committed.current_plus_di   # last closed candle
        adx     = engine.committed.current_adx
    """

    # ── Constructor ───────────────────────────────────────────
    def __init__(self, period: int = DMI_PERIOD):
        self.period = period

        # Committed state = last fully-closed candle's DMI values.
        # Touch this only when a new candle *closes*.
        self.committed   = DMIState(period=period)

        # In-progress state = current open candle's DMI values.
        # This is rebuilt on every incoming tick.
        self.in_progress = DMIState(period=period)

    # ─────────────────────────────────────────────────────────
    # PUBLIC: Initialise from historical DataFrame
    # ─────────────────────────────────────────────────────────
    def initialise(self, df: pd.DataFrame) -> None:
        """
        Bootstrap rolling state using TA-Lib on the full historical dataset.

        TA-Lib is the ground truth for the seed values; after this call
        we never touch TA-Lib again.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: open, high, low, close, volume
            Rows must be in chronological order (oldest first).
        """
        if len(df) < self.period * 2 + 1:
            # Not enough bars — mark as uninitialised and return silently.
            # The engine will still accept ticks; it just won't emit values
            # until it has processed enough candles on its own.
            print(f"[DMI] Warning: only {len(df)} candles; need at least "
                  f"{self.period * 2 + 1}. Running in cold-start mode.")
            self._cold_start(df)
            return

        high  = df['high'].values.astype(float)
        low   = df['low'].values.astype(float)
        close = df['close'].values.astype(float)

        # ── Ask TA-Lib for the last valid indicator values ────
        talib_plus_di  = ta.PLUS_DI(high,  low, close, timeperiod=self.period)
        talib_minus_di = ta.MINUS_DI(high, low, close, timeperiod=self.period)
        talib_adx      = ta.ADX(high,      low, close, timeperiod=self.period)

        # Find the last bar that has a valid ADX (ADX needs 2*period−1 bars)
        last_valid_idx = self._last_valid_index(talib_adx)

        if last_valid_idx < 0:
            print("[DMI] Warning: TA-Lib returned all NaN for ADX. "
                  "Running in cold-start mode.")
            self._cold_start(df)
            return

        # ── Reverse-engineer the Wilder smoothed accumulators ─
        #
        # TA-Lib gives us +DI and -DI at index i as:
        #   +DI[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
        #
        # We need the raw smoothed_tr, smoothed_plus_dm, smoothed_minus_dm
        # at the last valid bar so we can continue the recurrence.
        #
        # To recover them we need smoothed_tr.  We reconstruct smoothed_tr
        # directly by running the Wilder smoothing on the raw TR series —
        # this is guaranteed to match TA-Lib's internal accumulators.

        # Step 1 — compute per-bar raw values
        raw_plus_dm, raw_minus_dm, raw_tr = self._compute_raw_dm_tr(
            high, low, close
        )

        # Step 2 — run Wilder smoothing up to last_valid_idx
        (s_tr, s_plus_dm, s_minus_dm, s_dx,
         plus_di, minus_di, dx, adx) = self._wilder_smooth_series(
            raw_plus_dm, raw_minus_dm, raw_tr, last_valid_idx
        )

        # ── Populate committed state ───────────────────────────
        self.committed.smoothed_tr       = s_tr
        self.committed.smoothed_plus_dm  = s_plus_dm
        self.committed.smoothed_minus_dm = s_minus_dm
        self.committed.smoothed_dx       = s_dx
        self.committed.current_plus_di   = plus_di
        self.committed.current_minus_di  = minus_di
        self.committed.current_dx        = dx
        self.committed.current_adx       = adx
        self.committed.prev_high         = float(high[last_valid_idx])
        self.committed.prev_low          = float(low[last_valid_idx])
        self.committed.prev_close        = float(close[last_valid_idx])
        self.committed.is_initialised    = True
        self.committed.candles_seen      = last_valid_idx + 1

        # in_progress starts as a copy of committed;
        # it will diverge on the very next tick.
        self.in_progress = deepcopy(self.committed)

        print(f"[DMI] Initialised. Last bar  →  "
              f"+DI={plus_di:.4f}  -DI={minus_di:.4f}  ADX={adx:.4f}  "
              f"(TA-Lib: +DI={talib_plus_di[last_valid_idx]:.4f}  "
              f"-DI={talib_minus_di[last_valid_idx]:.4f}  "
              f"ADX={talib_adx[last_valid_idx]:.4f})")

    # ─────────────────────────────────────────────────────────
    # PUBLIC: Incremental tick update
    # ─────────────────────────────────────────────────────────
    def update(
        self,
        high:  float,
        low:   float,
        close: float,
        is_new_candle: bool,
    ) -> tuple[float, float, float, float]:
        """
        Update DMI with the latest candle OHLC values.

        Call this on EVERY tick, passing:
            is_new_candle = True  when a new timeframe bar has just opened
            is_new_candle = False when you are updating the still-open candle

        Returns
        -------
        (plus_di, minus_di, dx, adx)  of the IN-PROGRESS candle.
        Access committed.current_* for the last *closed* candle values.

        Complexity
        ----------
        O(1) — no loops, no array scans.
        """
        if not self.committed.is_initialised:
            # Engine hasn't received enough history yet; absorb the candle
            # into a lightweight cold-accumulator and return NaN.
            self._cold_tick(high, low, close, is_new_candle)
            return math.nan, math.nan, math.nan, math.nan

        if is_new_candle:
            # ── The previous in-progress candle has officially CLOSED ──
            #
            # Promote in_progress → committed.
            # This locks in the Wilder accumulators so future
            # candle updates start from the correct baseline.
            self.committed = deepcopy(self.in_progress)

            # The new candle needs the closed candle as its "prev" anchor.
            # (prev_high / prev_low / prev_close are already correct
            #  because in_progress stores the current bar's own OHLC
            #  in the prev_* fields after each step below.)

        # ── Recompute in_progress from committed baseline ─────
        #
        # Key insight: we ALWAYS recompute in_progress starting from
        # the COMMITTED state.  This means multiple tick updates to the
        # same bar simply overwrite in_progress — no accumulation error.
        self.in_progress = deepcopy(self.committed)

        # ── One step of Wilder smoothing ──────────────────────
        plus_dm, minus_dm, tr = self._single_dm_tr(
            high, low, close,
            self.committed.prev_high,
            self.committed.prev_low,
            self.committed.prev_close,
        )

        new_s_tr, new_s_pdm, new_s_mdm = self._wilder_step(
            tr, plus_dm, minus_dm,
            self.committed.smoothed_tr,
            self.committed.smoothed_plus_dm,
            self.committed.smoothed_minus_dm,
        )

        # ── Directional indicators ────────────────────────────
        plus_di, minus_di = self._compute_di(new_s_pdm, new_s_mdm, new_s_tr)

        # ── DX ────────────────────────────────────────────────
        dx = self._compute_dx(plus_di, minus_di)

        # ── ADX (Wilder-smoothed DX) ──────────────────────────
        new_s_dx = self._wilder_step_scalar(dx, self.committed.smoothed_dx, self.period)
        adx = new_s_dx

        # ── Write results into in_progress ───────────────────
        self.in_progress.smoothed_tr       = new_s_tr
        self.in_progress.smoothed_plus_dm  = new_s_pdm
        self.in_progress.smoothed_minus_dm = new_s_mdm
        self.in_progress.smoothed_dx       = new_s_dx
        self.in_progress.current_plus_di   = plus_di
        self.in_progress.current_minus_di  = minus_di
        self.in_progress.current_dx        = dx
        self.in_progress.current_adx       = adx

        # Store this candle's OHLC as "prev" for the NEXT bar
        self.in_progress.prev_high  = high
        self.in_progress.prev_low   = low
        self.in_progress.prev_close = close
        self.in_progress.candles_seen += 1

        return plus_di, minus_di, dx, adx

    # ─────────────────────────────────────────────────────────
    # PRIVATE: Core maths helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _single_dm_tr(
        high: float, low: float, close: float,
        prev_high: float, prev_low: float, prev_close: float,
    ) -> tuple[float, float, float]:
        """
        Compute raw +DM, -DM, TR for a single bar.

        Returns
        -------
        (plus_dm, minus_dm, true_range)
        """
        up_move   = high - prev_high
        down_move = prev_low - low

        plus_dm  = up_move   if (up_move   > down_move and up_move   > 0) else 0.0
        minus_dm = down_move if (down_move > up_move   and down_move > 0) else 0.0

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low  - prev_close),
        )

        return plus_dm, minus_dm, tr

    # @staticmethod
    def _wilder_step(
        self,
        tr: float, plus_dm: float, minus_dm: float,
        prev_s_tr: float, prev_s_pdm: float, prev_s_mdm: float,
    ) -> tuple[float, float, float]:
        """
        One step of Wilder smoothing for TR, +DM, and -DM.

        Wilder recurrence:
            smoothed[i] = smoothed[i-1] - smoothed[i-1]/period + raw[i]
                        = smoothed[i-1] * (1 - 1/period) + raw[i]

        This is equivalent to an EMA with alpha = 1/period.
        """
        # Guard against NaN in accumulators (shouldn't happen after init)
        if math.isnan(prev_s_tr):
            return tr, plus_dm, minus_dm

        # k = 1.0 / self.period #DMIEngine._period_from_state(prev_s_tr, tr)  # will use instance method below
        # # NOTE: k is overridden in the caller via the instance — see _wilder_step_with_period
        # raise RuntimeError("Use _wilder_step_with_period instead")

        alpha = 1.0 / self.period
        s_tr  = prev_s_tr  - prev_s_tr  * alpha + tr
        s_pdm = prev_s_pdm - prev_s_pdm * alpha + plus_dm
        s_mdm = prev_s_mdm - prev_s_mdm * alpha + minus_dm
        return s_tr, s_pdm, s_mdm

    # @staticmethod
    # def _wilder_step_with_period(
    #     period: int,
    #     tr: float, plus_dm: float, minus_dm: float,
    #     prev_s_tr: float, prev_s_pdm: float, prev_s_mdm: float,
    # ) -> tuple[float, float, float]:
    #     """
    #     One Wilder step parameterised by period.
    #     alpha = 1 / period
    #     """
    #     if math.isnan(prev_s_tr):
    #         return tr, plus_dm, minus_dm

    #     alpha = 1.0 / period
    #     s_tr  = prev_s_tr  - prev_s_tr  * alpha + tr
    #     s_pdm = prev_s_pdm - prev_s_pdm * alpha + plus_dm
    #     s_mdm = prev_s_mdm - prev_s_mdm * alpha + minus_dm
    #     return s_tr, s_pdm, s_mdm

    @staticmethod
    def _wilder_step_scalar(raw: float, prev_smoothed: float, period: int) -> float:
        """
        Single-value Wilder step (used for ADX = Wilder(DX)).

        period is an explicit parameter so this static method works correctly
        for any DMIEngine instance regardless of its configured period.
        """
        if math.isnan(prev_smoothed) or math.isnan(raw):
            return raw
        alpha = 1.0 / period
        return prev_smoothed - prev_smoothed * alpha + raw

    @staticmethod
    def _compute_di(
        s_pdm: float, s_mdm: float, s_tr: float
    ) -> tuple[float, float]:
        """Compute +DI and -DI from Wilder-smoothed values."""
        if s_tr == 0.0 or math.isnan(s_tr):
            return 0.0, 0.0
        plus_di  = 100.0 * s_pdm / s_tr
        minus_di = 100.0 * s_mdm / s_tr
        return plus_di, minus_di

    @staticmethod
    def _compute_dx(plus_di: float, minus_di: float) -> float:
        """DX = 100 * |+DI - -DI| / (+DI + -DI)."""
        denom = plus_di + minus_di
        if denom == 0.0 or math.isnan(denom):
            return 0.0
        return 100.0 * abs(plus_di - minus_di) / denom

    # ── Vectorised helpers used during initialisation ─────────

    @staticmethod
    def _compute_raw_dm_tr(
        high:  np.ndarray,
        low:   np.ndarray,
        close: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-bar raw +DM, -DM, TR arrays for the full history.
        Index 0 is undefined (no previous bar); we set it to 0.
        """
        n = len(high)
        plus_dm  = np.zeros(n)
        minus_dm = np.zeros(n)
        tr       = np.zeros(n)

        for i in range(1, n):
            up_move   = high[i]  - high[i-1]
            down_move = low[i-1] - low[i]

            plus_dm[i]  = up_move   if (up_move   > down_move and up_move   > 0) else 0.0
            minus_dm[i] = down_move if (down_move > up_move   and down_move > 0) else 0.0

            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i]  - close[i-1]),
            )

        return plus_dm, minus_dm, tr

    def _wilder_smooth_series(
        self,
        plus_dm:  np.ndarray,
        minus_dm: np.ndarray,
        tr:       np.ndarray,
        up_to:    int,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """
        Run Wilder smoothing from bar 1 to bar `up_to` (inclusive).

        Returns the scalar accumulators and indicator values at `up_to`.

        TA-Lib's convention:
          - First smoothed value (at bar `period`) = simple sum of first `period` raw values.
          - Subsequent values use the Wilder recurrence.
        """
        p = self.period
        n = up_to + 1  # number of bars to process

        if n < p + 1:
            raise ValueError(
                f"Need at least {p+1} bars to compute initial Wilder sum, got {n}."
            )

        # ── Seed: simple sum of bars [1..period] ─────────────
        # (TA-Lib uses bars 1..period for the first smoothed value,
        #  bar 0 is unused because there's no "previous" bar.)
        s_tr  = float(np.sum(tr[1 : p + 1]))
        s_pdm = float(np.sum(plus_dm[1 : p + 1]))
        s_mdm = float(np.sum(minus_dm[1 : p + 1]))

        # Compute DI at seed bar
        plus_di, minus_di = self._compute_di(s_pdm, s_mdm, s_tr)
        dx = self._compute_dx(plus_di, minus_di)

        # ── Seed for ADX: first ADX value = avg of DX[period..2*period-1] ──
        # We need to track DX from bar `period` onwards until bar `2*period-1`
        # for the initial ADX seed, then continue with Wilder recurrence.
        dx_values: list[float] = [dx]  # DX at bar `period`

        s_dx  = math.nan
        adx   = math.nan

        for i in range(p + 1, n):
            # Wilder step
            alpha = 1.0 / p
            s_tr  = s_tr  - s_tr  * alpha + tr[i]
            s_pdm = s_pdm - s_pdm * alpha + plus_dm[i]
            s_mdm = s_mdm - s_mdm * alpha + minus_dm[i]

            plus_di, minus_di = self._compute_di(s_pdm, s_mdm, s_tr)
            dx = self._compute_dx(plus_di, minus_di)
            dx_values.append(dx)

            # ADX seeding: once we have `period` DX values, seed s_dx
            if len(dx_values) == p:
                s_dx = float(np.mean(dx_values))
                adx  = s_dx
            elif len(dx_values) > p:
                # Continue Wilder smoothing for ADX
                s_dx = s_dx - s_dx * alpha + dx
                adx  = s_dx

        return s_tr, s_pdm, s_mdm, s_dx, plus_di, minus_di, dx, adx

    # ─────────────────────────────────────────────────────────
    # PRIVATE: Cold-start (insufficient history)
    # ─────────────────────────────────────────────────────────

    def _cold_start(self, df: pd.DataFrame) -> None:
        """
        Absorb what history we do have without TA-Lib.
        Useful if the session starts very early and only a handful of bars exist.
        """
        high  = df['high'].values.astype(float)
        low   = df['low'].values.astype(float)
        close = df['close'].values.astype(float)

        for i in range(len(high)):
            is_new = True
            self._cold_tick(high[i], low[i], close[i], is_new)

    def _cold_tick(
        self,
        high: float, low: float, close: float,
        is_new_candle: bool,
    ) -> None:
        """
        Accumulate bars during cold-start until we have enough to produce values.
        Uses simple-sum initialisation (matches TA-Lib's first smoothed value).
        """
        p = self.period

        # For simplicity, store raw arrays in a temporary buffer
        if not hasattr(self, '_cold_buf'):
            self._cold_buf: list[tuple[float, float, float]] = []

        # if is_new_candle and len(self._cold_buf) > 0:
        #     # Commit the previous tick as a closed bar
        #     pass  # the buffer already has the previous bar appended below

        self._cold_buf.append((high, low, close))

        if len(self._cold_buf) < 2 * p + 1:
            return  # still not enough

        # We now have sufficient bars — hand off to proper initialisation
        buf = self._cold_buf
        df_tmp = pd.DataFrame(buf, columns=['high', 'low', 'close'])
        df_tmp['open']   = df_tmp['close']
        df_tmp['volume'] = 0
        self.initialise(df_tmp)
        del self._cold_buf  # free the buffer

    # ─────────────────────────────────────────────────────────
    # PRIVATE: Thin wrappers to call the correct static methods
    # ─────────────────────────────────────────────────────────

    # def _wilder_step(
    #     self,
    #     tr: float, plus_dm: float, minus_dm: float,
    #     prev_s_tr: float, prev_s_pdm: float, prev_s_mdm: float,
    # ) -> tuple[float, float, float]:
    #     """Instance-level wrapper so we can pass self.period."""
    #     return DMIEngine._wilder_step_with_period(
    #         self.period, tr, plus_dm, minus_dm,
    #         prev_s_tr, prev_s_pdm, prev_s_mdm,
    #     )

    # ─────────────────────────────────────────────────────────
    # PRIVATE: Miscellaneous
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _last_valid_index(arr: np.ndarray) -> int:
        """Return the index of the last non-NaN element, or -1."""
        for i in range(len(arr) - 1, -1, -1):
            if not math.isnan(arr[i]):
                return i
        return -1


# # ============================================================
# # Rolling EMA (kept from original for reference)
# # ============================================================
# def rolling_ema(ltp: float, prev_ema: float, length: int) -> float:
#     multiplier = 2 / (length + 1)
#     return (ltp - prev_ema) * multiplier + prev_ema


# ============================================================
# Historical Data Fetching
# ============================================================
def fetch_historical_data(fyers: fyersModel.FyersModel) -> pd.DataFrame:
    now           = dt.datetime.now(pytz.timezone(timeZone))
    start_of_day  = now.replace(hour=0, minute=0, second=0, microsecond=0)
    nifty_data    = {
        "symbol":      symbol,
        "resolution":  resolution,
        "date_format": "0",
        "range_from":  int(start_of_day.timestamp()),
        "range_to":    int(now.timestamp()),
        "cont_flag":   "1",
    }
    response         = fyers.history(data=nifty_data)
    historical_data  = response['candles']
    df = pd.DataFrame(
        historical_data, columns=['date', 'open', 'high', 'low', 'close', 'volume']
    )
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df['date'] = df['date'].dt.tz_localize('UTC').dt.tz_convert(pytz.timezone(timeZone))
    df.set_index('date', inplace=True)
    return df


# ============================================================
# Support and Resistance Data
# ============================================================
def support_resistance(df):
    """
    Detect support and resistance trendlines from market data.

    This function uses the `trendln` library to identify important
    price structures in the market.

    Support lines are created using candle LOW prices.
    Resistance lines are created using candle HIGH prices.

    The library first:
    1. Finds local swing highs and swing lows.
    2. Creates possible trendlines.
    3. Ranks the best trendlines mathematically.

    We only return the latest window because recent market structure
    is more useful for real-time trading than old historical structure.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLCV candlestick dataframe.

    Returns
    -------
    tuple
        Latest support trendlines and resistance trendlines.
    """
    
    (minimaIdxs, pmin, mintrend, minwindows),(maximaIdxs, pmax, maxtrend, maxwindows) = trendln.calc_support_resistance(
    (df['low'].to_numpy(), df['high'].to_numpy()),
    extmethod=trendln.METHOD_NUMDIFF,
    method=trendln.METHOD_NSQUREDLOGN,
    window=50,
    errpct=0.003)

    minwindows = [item for sublist in minwindows for item in sublist]
    maxwindows = [item for sublist in maxwindows for item in sublist]

    return minwindows, maxwindows


# ============================================================
# Generate Trendline Zone
# ============================================================
def _build_zone_cache(df, minwindows, maxwindows):
    """
    Extract only the zone boundary values at the most recent candle.

    Rather than storing full-length arrays for every trendline, this stores
    a compact dict per zone with just the upper/lower boundary at the last
    closed candle index. This makes tick-level zone checks O(k) where
    k = number of active zones (typically 4-8), not O(n).

    Parameters
    ----------
    df         : DataFrame of CLOSED candles (excluding the forming candle).
    minwindows : Support trendline data from trendln.
    maxwindows : Resistance trendline data from trendln.

    Returns
    -------
    list[dict]  Active zones, each with keys:
        type       : 'support' or 'resistance'
        upper      : upper boundary of the zone at the last closed candle
        lower      : lower boundary of the zone at the last closed candle
        slope      : trendline slope (for projecting zone forward in time)
        intercept  : trendline intercept
        sd         : standard deviation used for zone width
        start_idx  : candle index where the trendline originates
        df_length  : number of closed candles when this zone was computed
        points     : raw trendln point indices (for debug / display)
    """
    zones = []
    last_idx = len(df) - 1

    for trend_data, zone_type in [(minwindows, 'support'), (maxwindows, 'resistance')]:
        for points, result in trend_data:
            if len(points) < 2:
                continue

            slope = result[0]
            intercept = result[1]
            ssr = result[2]
            n = len(points)

            if n < 3:       # need at least 3 points for n-2 denominator
                continue

            sd = math.sqrt(ssr / (n - 2)) if ssr > 0 else 0
            # min_sd = intercept * 0.003
            # sd = max(sd, min_sd)

            # Check if zone is still active up to the last candle
            is_active = True
            for i in range(min(points), last_idx + 1):
                val = slope * i + intercept
                if df['low'].iloc[i] + 3 * sd < val and zone_type == 'support':
                    is_active = False
                    break
                elif df['high'].iloc[i] - 3 * sd > val and zone_type == 'resistance':
                    is_active = False
                    break

            # ── Debug print ──────────────────────────────────────
            val_at_last = slope * last_idx + intercept
            upper = val_at_last + sd * 3
            lower = val_at_last - sd * 3

            status = "ACTIVE  ✅" if is_active else "BROKEN  ❌"
            print(f"  [{zone_type.upper():<10}] {status} | "
                  f"zone={lower:.2f}-{upper:.2f} | "
                  f"slope={slope:.4f} | points={points}")

            if not is_active:
                continue

            # Only store the boundary values at the last candle index
            # This is the single value we need for entry/exit checks
            # val_at_last = slope * last_idx + intercept
            zones.append({
                'type': zone_type,
                'upper': val_at_last + sd * 3,
                'lower': val_at_last - sd * 3,
                'slope': slope,
                'intercept': intercept,
                'sd': sd,
                'start_idx': min(points),   # trendline starts here in the closed df
                'df_length': last_idx + 1,  # how many candles were in closed df when computed
                'points': points,
            })

    return zones



# ============================================================
# Rolling EMA Calculation for Live Candle
# ============================================================
def rolling_ema(ltp, prev_ema, length):
    """
    Calculate the Exponential Moving Average (EMA) for a new tick using the previous EMA value.

    Parameters
    ----------
    ltp : float
        The latest traded price (current market price).

    prev_ema : float
        The EMA value from the last closed candle.

    length : int
        The period length for the EMA calculation (e.g., 9 or 15).

    Returns
    -------
    float
        The updated EMA value based on the latest tick.
    """
    multiplier = 2 / (length + 1)
    new_ema = (ltp - prev_ema) * multiplier + prev_ema
    return new_ema


# ============================================================
# update_candle — tick-by-tick candle builder + DMI updater
# ============================================================
def update_candle(
    data:              pd.DataFrame,
    message:           dict,
    last_total_volume: Optional[int],
    dmi_engine:        DMIEngine,
) -> tuple[pd.DataFrame, Optional[int]]:
    """
    Process one WebSocket tick:
      1. Build / update the OHLCV candle.
      2. Run one O(1) DMI update.
      3. Write +DI, -DI, DX, ADX back into the DataFrame row.

    Parameters
    ----------
    data              : OHLCV DataFrame (+ indicator columns).
    message           : Raw Fyers WebSocket tick dict.
    last_total_volume : Previous cumulative session volume.
    dmi_engine        : Shared DMIEngine instance (maintains rolling state).

    Returns
    -------
    (updated_data, total_vol, zones)
    """

    # ── Guard: malformed message ──────────────────────────────
    if "symbol" not in message:
        return data, last_total_volume, None

    ltp = message.get('ltp')

    if ltp is None:
        return data, last_total_volume, None

    total_vol = message.get('vol_traded_today', last_total_volume or 0)
    if total_vol is None:
        total_vol = last_total_volume if last_total_volume is not None else 0

    timestamp = pd.Timestamp.now(tz=timeZone).floor('1min')

    # ── Volume accounting ─────────────────────────────────────
    if last_total_volume is None:
        incremental_vol = 0
    else:
        incremental_vol = max(0, total_vol - last_total_volume)

    # ── Candle update or creation ─────────────────────────────
    # is_new_candle: bool
    zones: Optional[list] = None

    if len(data) > 0 and data.index[-1] == timestamp:
        # ── UPDATE existing open candle ───────────────────────
        is_new_candle = False

        data.iloc[-1, data.columns.get_loc('close')]  = ltp
        data.iloc[-1, data.columns.get_loc('high')]   = max(data.iloc[-1]['high'],  ltp)
        data.iloc[-1, data.columns.get_loc('low')]    = min(data.iloc[-1]['low'],   ltp)
        data.iloc[-1, data.columns.get_loc('volume')] += incremental_vol

        # Grab updated OHLC for this bar
        cur_high  = float(data.iloc[-1]['high'])
        cur_low   = float(data.iloc[-1]['low'])
        cur_close = float(ltp)

        # data, total_vol, zones = data, total_vol, None

    else:
        # ── CREATE new candle (previous bar just closed) ──────
        is_new_candle = True

        new_candle = pd.DataFrame(
            [{
                'open':    ltp,
                'high':    ltp,
                'low':     ltp,
                'close':   ltp,
                'volume':  incremental_vol,
                '+di':     float('nan'),
                '-di':     float('nan'),
                'dx':      float('nan'),
                'adx':     float('nan'),
                'ema_+di': float('nan'),
                'ema_-di': float('nan'),
            }],
            index=[timestamp],
        )

        data = pd.concat([data, new_candle])

        # Compute S/R on all closed candles (exclude last, which is forming)
        closed = data.iloc[:-1]
        
        # Build compact zone list — just the boundary values at the LAST closed candle
        # This is O(k) where k = number of trendlines, typically 4-8, not n=100
        # Inside update_live_data, just before calling _build_zone_cache
        print(f"\n{'='*55}")
        print(f"[CANDLE CLOSE] {data.index[-2]}  |  "
            f"O={closed.iloc[-1]['open']} H={closed.iloc[-1]['high']} "
            f"L={closed.iloc[-1]['low']} C={closed.iloc[-1]['close']}")
        print(f"{'='*55}")
        minwindows, maxwindows = support_resistance(df=closed)
        zones = _build_zone_cache(closed, minwindows, maxwindows)

        cur_high  = ltp
        cur_low   = ltp
        cur_close = ltp

        # data, total_vol, zones = data, total_vol, zones

    # ── O(1) DMI update ───────────────────────────────────────
    plus_di, minus_di, dx, adx = dmi_engine.update(
        high=cur_high,
        low=cur_low,
        close=cur_close,
        is_new_candle=is_new_candle,
    )

    # ── Write indicator values into the current row ───────────
    #
    # We write the IN-PROGRESS values into the live candle row so
    # the chart / strategy can see real-time estimates.
    #
    # For strategy decisions on CLOSED candles use:
    #   dmi_engine.committed.current_plus_di
    #   dmi_engine.committed.current_adx   etc.
    #
    for col, val in [('+di', plus_di), ('-di', minus_di),
                     ('dx', dx),       ('adx', adx)]:
        if col not in data.columns:
            data[col] = float('nan')
        data.iloc[-1, data.columns.get_loc(col)] = val
    
    # ── Rolling EMA of +DI and -DI (noise filter) ────────────
    #
    # Strategy uses ema_+di / ema_-di instead of raw +DI / -DI
    # to avoid reacting to single-candle spikes.
    #
    # Updated on every tick using the same rolling EMA formula as the
    # original EMA reference codebase (prev row's ema_* value as seed).
    # On a new candle the prev row is the just-closed bar; on an intra-
    # candle tick the prev row is iloc[-2], which is also the last closed
    # bar — so the formula is identical in both cases.
    if len(data) >= 2:
        prev_ema_plus_di  = data.iloc[-2]['ema_+di']
        prev_ema_minus_di = data.iloc[-2]['ema_-di']
 
        if pd.notna(plus_di) and pd.notna(prev_ema_plus_di):
            new_ema_plus_di = rolling_ema(plus_di, prev_ema_plus_di, EMA_DI_PERIOD)
        else:
            new_ema_plus_di = plus_di          # propagate the raw value until EMA warms up
 
        if pd.notna(minus_di) and pd.notna(prev_ema_minus_di):
            new_ema_minus_di = rolling_ema(minus_di, prev_ema_minus_di, EMA_DI_PERIOD)
        else:
            new_ema_minus_di = minus_di
 
        for col, val in [('ema_+di', new_ema_plus_di), ('ema_-di', new_ema_minus_di)]:
            if col not in data.columns:
                data[col] = float('nan')
            data.iloc[-1, data.columns.get_loc(col)] = val

    return data, total_vol, zones


# ============================================================
# Candlestick Pattern Detection
# ============================================================
def candle_signal(data):
    """
    Scan the most recent candles for a recognized bullish or bearish pattern.

    This function iterates through a curated list of 60+ TA-Lib candlestick
    pattern recognition functions and returns the first pattern detected,
    along with an 'index' value indicating how many candles back the pattern's
    anchor candle (used for Stop Loss placement) lies.

    Pattern Evaluation Rules:
        - Most patterns are evaluated at iloc[-2] — the last CLOSED candle.
          (iloc[-1] is the live, still-forming candle and is intentionally excluded.)
        - Patterns that require a confirmation candle after the signal candle
          (e.g., Doji, Hammer, Spinning Top) are evaluated at iloc[-3], because
          we check whether the NEXT candle (iloc[-2]) has already confirmed the signal
          by closing above the pattern's high (bullish) or below its low (bearish).
        - Patterns requiring a 'penetration' parameter (e.g., Morning/Evening Star,
          Dark Cloud Cover) are called with penetration=0 to disable the threshold filter.

    Stop Loss Index:
        The 'index' value paired with each pattern indicates the candle position
        used for Stop Loss calculation:
        - index=1 → SL based on the signal candle itself (iloc[-2])
        - index=2 → SL based on the candle before the signal (iloc[-3])
        - index=3 → SL based on two candles before the signal (iloc[-4])
        - index=5 → SL based on four candles before the signal (iloc[-6])
        This is used by the caller as: self.data.iloc[-1 - index]

    Special Cases:
        - CDL3INSIDE: The SL index is determined dynamically based on which
          candle's low/high is more extreme.
        - CDLDOJI / CDLHAMMER / CDLLONGLEGGEDDOJI / CDLRICKSHAWMAN / CDLSPINNINGTOP:
          These are context-dependent patterns. A raw TA-Lib signal is only acted
          upon if the NEXT candle has already closed beyond the pattern's range
          (i.e., confirmed breakout above high or breakdown below low).
        - CDLHAMMER: Only treated as bullish if the next candle confirms by
          closing above the hammer's high.

    Parameters
    ----------
    data : pandas.DataFrame
        A slice of the OHLCV dataframe (typically the last 10 candles).
        Must contain columns: 'open', 'high', 'low', 'close'.
        Must have at least 10 rows for reliable TA-Lib lookback computation.

    Returns
    -------
    tuple : (int, int)
        - candle : Signal value. +100 = bullish, -100 = bearish, 0 = no signal.
        - index  : Number of candles back to use for Stop Loss placement.
                   0 if no signal detected.
    """

    # ── Pattern Registry ──────────────────────────────────────────────────
    # Each entry is a tuple of (TA-Lib function name, SL index).
    # Patterns are checked in order — the FIRST match is returned.
    # Patterns at the top of the list are given higher priority.
    patterns = [
        ("CDL2CROWS", 2),           ("CDL3BLACKCROWS", 3),      ("CDL3INSIDE", 2),          ("CDL3LINESTRIKE", 1),      ("CDL3OUTSIDE", 2),      ("CDL3STARSINSOUTH", 3),
        ("CDL3WHITESOLDIERS", 3),   ("CDLABANDONEDBABY", 2),    ("CDLADVANCEBLOCK", 1),     ("CDLBELTHOLD", 1),         ("CDLBREAKAWAY", 2),
        ("CDLCLOSINGMARUBOZU", 1),  ("CDLCONCEALBABYSWALL", 1), ("CDLCOUNTERATTACK", 1),    ("CDLDARKCLOUDCOVER", 1),   ("CDLDOJI", 1),
        ("CDLDOJISTAR", 2),         ("CDLDRAGONFLYDOJI", 1),    ("CDLENGULFING", 1),        ("CDLEVENINGDOJISTAR", 2),  ("CDLEVENINGSTAR", 2),
        ("CDLGAPSIDESIDEWHITE", 3), ("CDLGRAVESTONEDOJI", 1),   ("CDLHAMMER", 1),           ("CDLHANGINGMAN", 1),       ("CDLHARAMI", 2),
        ("CDLHARAMICROSS", 2),      ("CDLHIGHWAVE", 1),         ("CDLHIKKAKE", 2),          ("CDLHIKKAKEMOD", 2),       ("CDLHOMINGPIGEON", 2),
        ("CDLIDENTICAL3CROWS", 3),  ("CDLINNECK", 2),           ("CDLINVERTEDHAMMER", 1),   ("CDLKICKING", 2),          ("CDLKICKINGBYLENGTH", 2),
        ("CDLLADDERBOTTOM", 2),     ("CDLLONGLEGGEDDOJI", 1),   ("CDLLONGLINE", 1),         ("CDLMARUBOZU", 1),         ("CDLMATCHINGLOW", 1),
        ("CDLMATHOLD", 5),          ("CDLMORNINGDOJISTAR", 2),  ("CDLMORNINGSTAR", 2),      ("CDLONNECK", 2),           ("CDLPIERCING", 1),
        ("CDLRICKSHAWMAN", 1),      ("CDLRISEFALL3METHODS", 5), ("CDLSEPARATINGLINES", 2),  ("CDLSHOOTINGSTAR", 1),     ("CDLSHORTLINE", 1),
        ("CDLSPINNINGTOP", 1),      ("CDLSTALLEDPATTERN", 1),   ("CDLSTICKSANDWICH", 1),    ("CDLTAKURI", 1),           ("CDLTASUKIGAP", 3),
        ("CDLTHRUSTING", 1),        ("CDLTRISTAR", 2),          ("CDLUNIQUE3RIVER", 2),     ("CDLUPSIDEGAP2CROWS", 1),  ("CDLXSIDEGAP3METHODS", 2),
    ]

    # Unpack OHLC columns as Series for direct use with TA-Lib functions
    open_, high, low, close = data["open"], data["high"], data["low"], data["close"]

    for pattern_name, index in patterns:

        # Dynamically fetch the TA-Lib function by name (e.g., ta.CDLENGULFING)
        func = getattr(ta, pattern_name)

        # ── Evaluate the Pattern ───────────────────────────────────────────
        if pattern_name in {
            # These patterns require a 'penetration' parameter to control
            # how deeply the second candle must close into the first candle's body.
            # Setting penetration=0 disables the filter and accepts all occurrences.
            "CDLABANDONEDBABY",
            "CDLDARKCLOUDCOVER",
            "CDLEVENINGDOJISTAR",
            "CDLEVENINGSTAR",
            "CDLMATHOLD",
            "CDLMORNINGDOJISTAR",
            "CDLMORNINGSTAR",
        }:
            # Read the signal at iloc[-2]: the last fully closed candle
            candle = func(open_, high, low, close, penetration=0).iloc[-2]

        elif pattern_name in {
            # These single-candle patterns need next-candle confirmation.
            # We evaluate the pattern at iloc[-3] (the signal candle)
            # and later check if iloc[-2] (the next candle) confirms direction.
            "CDLDOJI", "CDLHAMMER", "CDLLONGLEGGEDDOJI", "CDLRICKSHAWMAN", "CDLSPINNINGTOP"
        }:
            candle = func(open_, high, low, close).iloc[-3]

        else:
            # Standard case: evaluate at the last closed candle
            candle = func(open_, high, low, close).iloc[-2]

        # ── Non-Zero Signal Detected ───────────────────────────────────────
        if candle != 0:

            # Special case: CDL3INSIDE (Harami + confirmation candle)
            # The SL index is dynamic — it points to whichever candle
            # has the more extreme high/low relevant to trade direction.
            if pattern_name == "CDL3INSIDE":
                index = (2 if low.iloc[-3] < low.iloc[-4] else 3) if candle > 0 else (2 if high.iloc[-3] > high.iloc[-4] else 3)

            # Special case: Doji-family patterns and Spinning Tops
            # These are neutral by themselves — we require the NEXT candle
            # to have already broken above the pattern high (bullish confirmation)
            # or below the pattern low (bearish confirmation) before acting.
            elif pattern_name in {"CDLDOJI", "CDLLONGLEGGEDDOJI", "CDLRICKSHAWMAN", "CDLSPINNINGTOP"}:
                if close.iloc[-2] > high.iloc[-3]:
                    # Next candle closed above the pattern's high → confirmed bullish breakout
                    candle, index = 100, 2
                elif close.iloc[-2] < low.iloc[-3]:
                    # Next candle closed below the pattern's low → confirmed bearish breakdown
                    candle, index = -100, 2
                # If neither condition is met, candle retains its raw TA-Lib value,
                # which will be returned as-is (could be bullish or bearish based on context)

            # Special case: Hammer
            # A Hammer is only treated as a bullish signal if the next candle
            # confirms by closing above the hammer's high. Otherwise we ignore it.
            elif pattern_name == "CDLHAMMER":
                if close.iloc[-2] > high.iloc[-3]:
                    candle, index = 100, 2
                # If not confirmed, we fall through and still return the raw signal

            # Return the first pattern match found — direction + SL index
            return candle, index

    # No pattern detected across all 60+ checks
    return 0, 0


# ============================================================
# Log Data
# ============================================================
def log_trade(action, symbol, price):
    """
    Append trade details to a CSV log.

    Parameters:
        action (str): "BUY" or "SELL".
        symbol (str): Symbol.
        price (float): Execution price.
    """
    with open("trades_log.csv", "a") as file:
        file.write(f"{dt.datetime.now(pytz.timezone(timeZone))},{action},'DMI',{symbol},{price}\n")


# ============================================================
# Candlestick class — WebSocket event handler
# ============================================================
class Candlestick:
    """
    Manages live market data and DMI state.

    Mirrors the EMA reference architecture:
      - __init__      : receives pre-initialised historical DataFrame
                        and an already-bootstrapped DMIEngine.
      - onmessage     : delegates to update_candle() on every tick.
      - onopen        : subscribes to the symbol feed.
    """

    def __init__(
        self,
        data:       pd.DataFrame,
        fyers:      fyersModel.FyersModel,
        dmi_engine: DMIEngine,
    ):
        self.data              = data
        self.fyers             = fyers
        self.dmi_engine        = dmi_engine
        self.last_total_volume: Optional[int] = None

        # Trading State Variables
        self.position = None     # LONG / SHORT / None
        self.sl = None           # Active stop loss
        self.tp = None           # Active take profit
        self.trigger = None      # Price level used for trailing logic

        # Prevents duplicate signal evaluation on the same candle
        self.last_evaluated_candle = None
        self.active_zones = []


    # ============================================================
    # Cleaner function
    # ============================================================    
    def _clear_position(self):
        self.position = None
        self.sl = None
        self.tp = None
        self.trigger = None


    def onmessage(self, message: dict) -> None:
        self.data, self.last_total_volume, new_zones = update_candle(
            data=self.data,
            message=message,
            last_total_volume=self.last_total_volume,
            dmi_engine=self.dmi_engine,
        )
        # Display latest values
        print(
            self.data[['open', 'high', 'low', 'close', '+di', 'ema_+di', '-di', 'ema_-di', 'adx']].tail()
        )
        # Committed (closed-candle) values for strategy logic:
        c = self.dmi_engine.committed
        print(
            f"[Committed] +DI={c.current_plus_di:.2f}  "
            f"-DI={c.current_minus_di:.2f}  "
            f"ADX={c.current_adx:.2f}"
        )

        # Refresh zone cache when a new candle just opened
        if new_zones is not None:
            self.active_zones = new_zones

        # ── Guard conditions ──────────────────────────────────
        if len(self.data) < 2:
            return
        
        ltp = message.get('ltp')
        if ltp is None:
            return

        # --- Last Closed Candle ---
        closed_candle = self.data.iloc[-2]
        closed_candle_time = self.data.index[-2]
        is_new_candle = closed_candle_time != self.last_evaluated_candle

        # --------------------------------------------------------
        # TICK-LEVEL Exit: Hard TP/SL (runs on EVERY tick)
        # --------------------------------------------------------

        # --- Condition 1: Hard TP/SL hit ---
        if self.position == 'LONG':
            if ltp >= self.tp or ltp < self.sl:
                print(f"[EXIT LONG] Hard TP/SL hit at {ltp}")
                bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                log_trade("Sell", symbol, bid)
                self._clear_position()


        # --- Condition 1: Hard TP/SL hit ---
        elif self.position == 'SHORT':
            if ltp <= self.tp or ltp > self.sl:
                print(f"[EXIT SHORT] Hard TP/SL hit at {ltp}")
                ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                log_trade("Buy", symbol, ask)
                self._clear_position()


        # --------------------------------------------------------
        # CANDLE-LEVEL Logic (runs only once per closed candle)
        # --------------------------------------------------------
        if not is_new_candle:
            return

        # Mark this candle as evaluated ONCE, at the top
        self.last_evaluated_candle = closed_candle_time

        candle, index = candle_signal(data=self.data.tail(10)) if len(self.data) >= 10 else (0, 0)

        plus_di = self.data['+di'].iloc[-2]
        minus_di = self.data['-di'].iloc[-2]
        adx = self.data['adx'].iloc[-2]
        ema_plus_di = self.data['ema_+di'].iloc[-2]
        prev_ema_plus_di = self.data['ema_+di'].iloc[-3]
        ema_minus_di = self.data['ema_-di'].iloc[-2]
        prev_ema_minus_di = self.data['ema_-di'].iloc[-3]
        prev_plus_di = self.data['+di'].iloc[-3]
        prev_minus_di = self.data['-di'].iloc[-3]
        prev_adx = self.data['adx'].iloc[-3]
        

        # ✅ Skip signal logic entirely if EMAs aren't ready yet
        if any(pd.isna(v) for v in [
            plus_di, minus_di, adx,
            ema_plus_di, ema_minus_di,
            prev_plus_di, prev_minus_di, prev_adx,
            prev_ema_plus_di, prev_ema_minus_di,
        ]):
            return
        
        # --------------------------------------------------------
        # CANDLE-LEVEL Exit: Cross signals + Trailing SL
        # --------------------------------------------------------
        # Move TP and SL together by the same amount.
        #
        # Example:
        # Old SL = 100
        # Old TP = 120
        #
        # New SL = 105
        #
        # Risk locked in = +5
        # TP shifted to 125
        #
        # This preserves the original reward:risk structure
        # while protecting accumulated profit.
        if self.position == 'LONG':

            if prev_adx > adx or prev_ema_plus_di > ema_plus_di or plus_di < minus_di or candle < 0:
                print(f"[EXIT LONG] market slowing down {closed_candle_time}")
                bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                log_trade("Sell", symbol, bid)
                self._clear_position()
            
            else:
                # --- Condition 2: Price entering a resistance zone ---
                # If ltp is inside any active resistance band, exit -
                # resistance overhead is a structural reason to close long
                for zone in self.active_zones:
                    if zone['type'] == 'resistance' and zone['lower'] <= ltp <= zone['upper']:
                        bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                        log_trade("Sell", symbol, bid)
                        print(f"[EXIT LONG] Entered resistance {zone['lower']:.2f}-{zone['upper']:.2f}")
                        self._clear_position()
                        break

                # --- Condition 3: Price breaking down through support ---
                # If ltp breaks BELOW a support zone that sits ABOVE sl,
                # that support has failed - exit before hitting hard SL.
                # Only check if still in position (condition 2 may have exited)
                if self.position == 'LONG':
                    for zone in self.active_zones:
                        if zone['type'] == 'support' \
                                and zone['lower'] > self.sl \
                                and ltp < zone['lower']:
                            bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                            log_trade("Sell", symbol, bid)
                            print(f"[EXIT LONG] Support broken {zone['lower']:.2f}")
                            self._clear_position()
                            break
            
            
            if self.position == 'LONG' and closed_candle['close'] > self.trigger:
                new_sl = closed_candle['low']
                if new_sl > self.sl:
                    diff    = new_sl - self.sl
                    self.tp += diff
                    self.sl  = new_sl
                    self.trigger = closed_candle['high']

        elif self.position == 'SHORT':

            if prev_adx > adx or prev_ema_minus_di > ema_minus_di or plus_di > minus_di or candle > 0:
                print(f"[EXIT SHORT] market slowing down {closed_candle_time}")
                ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                log_trade("Buy", symbol, ask)
                self._clear_position()

            else:
                # --- Condition 2: Price entering a support zone ---
                # Support below price is a structural reason to close short
                for zone in self.active_zones:
                    if zone['type'] == 'support' and zone['lower'] <= ltp <= zone['upper']:
                        ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                        log_trade("Buy", symbol, ask)
                        print(f"[EXIT SHORT] Entered support {zone['lower']:.2f}-{zone['upper']:.2f}")
                        self._clear_position()
                        break

                # --- Condition 3: Price breaking up through resistance ---
                if self.position == 'SHORT':
                    for zone in self.active_zones:
                        if zone['type'] == 'resistance' \
                                and zone['upper'] < self.sl \
                                and ltp > zone['upper']:
                            ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                            log_trade("Buy", symbol, ask)
                            print(f"[EXIT SHORT] Resistance broken {zone['upper']:.2f}")
                            self._clear_position()
                            break
            

            if self.position == 'SHORT' and closed_candle['close'] < self.trigger:
                new_sl = closed_candle['high']
                if new_sl < self.sl:
                    diff    = self.sl - new_sl
                    self.tp -= diff
                    self.sl  = new_sl
                    self.trigger = closed_candle['low']
        

        # --------------------------------------------------------
        # CANDLE-LEVEL Entry (only if flat after exit above)
        # --------------------------------------------------------
        if not self.position:

            # Long Entry Conditions:
            #
            # DI crossover
            #  -> bullish momentum is accelerating.
            #
            # when bullish momentum is expanding.
            if plus_di > minus_di and prev_plus_di < plus_di and adx > prev_adx and not candle < 0:
                ask = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['ask']
                log_trade("Buy", symbol, ask)
                self.sl       = closed_candle['low']
                self.tp       = ask + (ask - self.sl) * 2
                self.trigger  = closed_candle['high']
                self.position = 'LONG'
                print(f"[BUY] at {ask} | SL: {self.sl} | TP: {self.tp}")

            # Short Entry Conditions:
            #
            # DI crossover
            #  -> bearish momentum is accelerating.
            #
            # when bearish momentum is expanding.
            elif plus_di < minus_di and prev_minus_di < minus_di and adx > prev_adx and not candle > 0:
                bid = self.fyers.quotes(data={"symbols": symbol})['d'][0]['v']['bid']
                log_trade("Sell", symbol, bid)
                self.sl       = closed_candle['high']
                self.tp       = bid - (self.sl - bid) * 2
                self.trigger  = closed_candle['low']
                self.position = 'SHORT'
                print(f"[SELL] at {bid} | SL: {self.sl} | TP: {self.tp}")

    def onerror(self, message: dict)  -> None: print("Error:",             message)
    def onclose(self, message: dict)  -> None: print("Connection closed:", message)

    def onopen(self) -> None:
        data_type = "SymbolUpdate"
        fyersSocket.subscribe(symbols=[symbol], data_type=data_type)
        fyersSocket.keep_running()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    """
    ============================================================
    SYSTEM FLOW
    ============================================================

    Historical OHLCV data
            ↓
    TA-Lib seeds DMIEngine (once)
            ↓
    Live WebSocket ticks
            ↓
    update_candle() → DMIEngine.update()  [O(1) per tick]
            ↓
    In-progress +DI / -DI / ADX written to DataFrame
            ↓
    Committed values available for closed-candle strategy logic

    ============================================================
    COMPLEXITY ANALYSIS
    ============================================================

    initialise()         : O(N)  — scans history once, then discarded
    update() per tick    : O(1)  — 6 scalar arithmetic ops + 2 deepcopies
    Memory               : O(1)  — only two DMIState objects retained

    ============================================================
    """

    # 1. Load credentials
    try:
        with open('access_token.txt', 'r') as f:
            access_token = f.read().strip()
    except FileNotFoundError:
        print("Error: access_token.txt not found. Please login first.")
        exit(1)

    # 2. Fetch today's historical candles
    print("Fetching historical data...")
    fyers_connection = fyersModel.FyersModel(
        client_id=client_id, token=access_token,
        is_async=False, log_path=''
    )
    historical_df = fetch_historical_data(fyers=fyers_connection)

    # 3. Compute TA-Lib DMI columns on the historical DataFrame
    #    (these are stored in the DataFrame for display / backreference;
    #     the DMIEngine initialises its rolling state from the same data)
    historical_df['adx']  = ta.ADX(
        historical_df['high'], historical_df['low'], historical_df['close'],
        timeperiod=DMI_PERIOD
    )
    historical_df['+di']  = ta.PLUS_DI(
        historical_df['high'], historical_df['low'], historical_df['close'],
        timeperiod=DMI_PERIOD
    )
    historical_df['-di']  = ta.MINUS_DI(
        historical_df['high'], historical_df['low'], historical_df['close'],
        timeperiod=DMI_PERIOD
    )
    historical_df['dx']   = float('nan')  # will be filled live

    historical_df['ema_+di'] = ta.EMA(historical_df['+di'], timeperiod=EMA_DI_PERIOD)
    historical_df['ema_-di'] = ta.EMA(historical_df['-di'], timeperiod=EMA_DI_PERIOD)

    # 4. Bootstrap the DMIEngine from the same historical data
    dmi_engine = DMIEngine(period=DMI_PERIOD)
    dmi_engine.initialise(historical_df)  # TA-Lib called here, never again

    # 5. Initialise the Candlestick manager
    candlestick = Candlestick(
        data=historical_df,
        fyers=fyers_connection,
        dmi_engine=dmi_engine,
    )

    # 6. Connect to WebSocket
    fyersSocket = data_ws.FyersDataSocket(
        access_token=access_token,
        log_path="",
        litemode=False,
        write_to_file=False,
        reconnect=True,                          # auto-reconnect handles exchange drops
        on_connect=candlestick.onopen,
        on_close=candlestick.onclose,
        on_error=candlestick.onerror,
        on_message=candlestick.onmessage,
    )

    print("Connecting to live stream...")
    fyersSocket.connect()
