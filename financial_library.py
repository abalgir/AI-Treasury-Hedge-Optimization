# financial_library.py
# --------------------------------------------------------------------------------------
# PURPOSE (for CFO readers, non-technical):
# --------------------------------------------------------------------------------------
# What is this:
#   A self-contained “risk & optimization toolkit” used by the orchestrator and agent.
#   It loads data, calibrates risk inputs, proposes/normalizes hedge menus (universes),
#   optimizes hedge notionals, enforces guardrails, and runs stress scenarios.
#
# What it does:
#   - Defines policy constraints (VaR cap, duration-gap band, FX reduction target, budget).
#   - Loads portfolio/static data and writes/extends a hedge universe CSV.
#   - Calibrates US rate volatility and an average correlation measure.
#   - Computes baseline risk metrics (ΔEVE, VaR, duration gap, FX impact) with options
#     for parallel or curve (steepening/flattening) shifts.
#   - Optimizes hedge notionals given a candidate universe (menu of hedges).
#   - Proposes a hedge universe via AI (multiple providers) with fallbacks (cached/heuristic).
#   - Simulates a library of scenarios on base vs. hedged books.
#
# Why:
#   This isolates the “math & mechanics” so the orchestrator can focus on the business
#   workflow. It also makes the white paper’s claims reproducible and auditable.
#
# Data used:
#   Inputs:
#     - treasury_portfolio.csv (positions with notional, duration, currency, etc.)
#     - prices.csv (static rate refs; kept for parity)
#     - Market history via Yahoo Finance for ^TNX (10Y UST) and EURUSD=X
#     - Environment API keys (x.ai/OpenAI/Anthropic) if AI proposal is enabled
#     - last_good_universe.json (cached “good” AI universe)
#   Outputs:
#     - hedge_universe.csv (created/extended with proposed candidates)
# --------------------------------------------------------------------------------------

import os
import re
import json
import random
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# GLOBAL HOOK (wired by orchestrator/agent during closed-loop AI refinement)
# --------------------------------------------------------------------------------------
# What is this:
#   An optional callback that scores a candidate hedge universe during AI iterations.
#
# What it does:
#   Holds a function pointer set by the orchestrator; if present, it is used to
#   evaluate/score an AI-proposed universe so the AI can refine toward feasibility.
#
# Why:
#   Enables “governed AI”: the AI proposes, a bank-owned evaluator scores; AI adjusts.
#
# Data used:
#   Receives a DataFrame of candidate hedges; returns a dict of booleans/scores.
_UNIVERSE_EVALUATOR = None

# ======================================================================================
# CONSTRAINTS & ADAPTERS
# ======================================================================================

@dataclass(frozen=True)
class Constraints:
    """
    What is this:
        Centralized policy/guardrail settings used across optimization and checks.

    What it does:
        Stores limits as simple numbers:
          - var_cap_pct: Hedged daily VaR (95%) must be ≤ baseline × (1 + var_cap_pct).
          - gap_neg_floor / gap_abs_cap: Allowed duration-gap band [floor, +cap] in years.
          - fx_reduction: Required % reduction of FX P&L under +10% FX shock.
          - hedge_budget_pct: Max sum of hedge notionals vs. total assets.

    Why:
        One place to change high-level policy. Keeps math consistent and auditable.

    Data used:
        N/A (constants set at construction).
    """
    var_cap_pct: float = 0.10
    gap_neg_floor: float = -0.20
    gap_abs_cap: float = 0.80
    fx_reduction: float = 0.35
    hedge_budget_pct: float = 0.50  # universe upper-bound sum vs assets

DEFAULT_CONSTRAINTS = Constraints()

def align_selection(universe_df: pd.DataFrame, selection_df: pd.DataFrame) -> np.ndarray:
    """
    What is this:
        Utility to align a selection table back into a numeric vector in universe order.

    What it does:
        - Matches rows by (instrument_type + maturity_date) keys.
        - Returns a numpy vector of notionals aligned to universe_df rows.

    Why:
        Optimizers/repair routines operate on vectors; printers store tables.
        This bridges the two representations without ambiguity.

    Data used:
        Input:
          - universe_df: candidate hedge rows with instrument_type, maturity_date
          - selection_df: chosen hedges with selected_notional
        Output:
          - np.ndarray of notionals (float) in universe row order
    """
    vec = np.zeros(len(universe_df), dtype=float)
    if selection_df is None or selection_df.empty or universe_df is None or universe_df.empty:
        return vec
    ukey = (
        universe_df['instrument_type'].astype(str).str.lower().str.strip()
        + '|' + pd.to_datetime(universe_df['maturity_date']).dt.strftime('%Y-%m-%d')
    )
    k2i = {k: i for i, k in enumerate(ukey)}
    for _, r in selection_df.iterrows():
        k = (str(r['instrument_type']).lower().strip()
             + '|' + pd.to_datetime(r['maturity_date']).strftime('%Y-%m-%d'))
        i = k2i.get(k)
        if i is not None:
            vec[i] = float(r['selected_notional'])
    return vec

# ======================================================================================
# I/O HELPERS
# ======================================================================================

def load_portfolio(file_path: str = 'treasury_portfolio.csv') -> pd.DataFrame:
    """
    What is this:
        Portfolio loader for the balance sheet.

    What it does:
        Reads CSV into a DataFrame and parses maturity_date.

    Why:
        Creates the in-memory representation we operate on.

    Data used:
        Input: treasury_portfolio.csv
        Output: DataFrame with at least ['notional_amount','duration','currency','maturity_date', ...]
    """
    df = pd.read_csv(file_path)
    df['maturity_date'] = pd.to_datetime(df['maturity_date'])
    print(f"Portfolio loaded from {file_path}")
    return df

def load_static_rates(file_path: str = 'prices.csv') -> pd.DataFrame:
    """
    What is this:
        Loader for static rate references (kept for parity; not strictly required later).

    What it does:
        Reads CSV and returns as DataFrame.

    Why:
        Many treasury stacks keep reference price files; we keep the interface uniform.

    Data used:
        Input: prices.csv
        Output: DataFrame of reference values (unused downstream in this file).
    """
    df = pd.read_csv(file_path)
    print(f"Static rates loaded from {file_path}")
    return df

# financial_library.py  (replace the whole create_hedge_universe function)

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def create_hedge_universe(file_path: str = 'hedge_universe.csv', mode: str = 'minimal') -> pd.DataFrame:
    """
    What is this:
        Builder for a static (non-AI) hedge catalog used to benchmark the optimizer.
        It now supports two modes:
          - 'minimal'  : legacy-style 3-line menu (IRS, UST future, one FX option).
          - 'expanded' : 8-line menu mirroring AI families (IRS, OIS, futures, two FX
                         puts, FX fwd, and a cross-currency swap).

    What it does:
        - Creates an in-memory table of candidate hedges with guardrails (max notionals,
          maturities, currencies, FX deltas where relevant).
        - Saves the table to CSV so the orchestrator can use it for the “traditional”
          optimization path.

    Why:
        - “Minimal” reflects how many teams hedge today (small static list).
        - “Expanded” removes any claim that AI wins because the baseline lacked instruments.
          It gives the same families as AI, but keeps it static (no closed-loop feedback).

    Data used:
        Output only:
          - CSV file at `file_path` for transparency/audit.
    """
    now = datetime.now

    if mode.lower() == 'minimal':
        data = [
            {
                'instrument_type': 'Interest Rate Swap (Pay Fixed)',
                'max_notional': 20_000_000,
                'maturity_date': now() + timedelta(days=365 * 3),
                'interest_rate': 4.0,
                'duration': 2.8,
                'currency': 'USD',
                'volatility': 10.0,
                'correlation_to_us10y': 0.9,
                'strike': np.nan,
                'fx_delta': np.nan
            },
            {
                'instrument_type': 'US Treasury Future (Short)',
                'max_notional': 15_000_000,
                'maturity_date': now() + timedelta(days=365 * 2),
                'interest_rate': 3.8,
                'duration': 1.9,
                'currency': 'USD',
                'volatility': 8.0,
                'correlation_to_us10y': 0.95,
                'strike': np.nan,
                'fx_delta': np.nan
            },
            {
                'instrument_type': 'FX Option (USD/EUR Call)',
                'max_notional': 10_000_000,
                'maturity_date': now() + timedelta(days=365 * 1),
                'interest_rate': 0.1,
                'duration': 1.0,
                'currency': 'USD/EUR',
                'volatility': 15.0,
                'correlation_to_us10y': -0.6,
                'strike': 1.10,
                'fx_delta': 0.40
            }
        ]

    elif mode.lower() == 'expanded':
        # Mirrors the breadth of the AI universe shown in your recent run.
        data = [
            # IRS (5y, 10y): core duration management
            {
                'instrument_type': 'USD IRS pay-fixed',
                'max_notional': 35_000_000,
                'maturity_date': now() + timedelta(days=365 * 5),
                'interest_rate': 4.0,
                'duration': 4.5,
                'currency': 'USD',
                'volatility': 10.0,
                'correlation_to_us10y': 0.90,
                'strike': np.nan,
                'fx_delta': np.nan
            },
            {
                'instrument_type': 'USD IRS pay-fixed',
                'max_notional': 25_000_000,
                'maturity_date': now() + timedelta(days=365 * 10),
                'interest_rate': 4.1,
                'duration': 8.5,
                'currency': 'USD',
                'volatility': 10.5,
                'correlation_to_us10y': 0.90,
                'strike': np.nan,
                'fx_delta': np.nan
            },

            # OIS (7y): short-to-intermediate duration, lower beta than IRS
            {
                'instrument_type': 'USD OIS pay-fixed',
                'max_notional': 28_000_000,
                'maturity_date': now() + timedelta(days=365 * 7),
                'interest_rate': 3.8,
                'duration': 4.0,
                'currency': 'USD',
                'volatility': 8.0,
                'correlation_to_us10y': 0.70,
                'strike': np.nan,
                'fx_delta': np.nan
            },

            # Futures (10y): liquid convexity & efficient DV01
            {
                'instrument_type': 'UST futures short',
                'max_notional': 20_000_000,
                'maturity_date': now() + timedelta(days=365 * 2),
                'interest_rate': 3.8,
                'duration': 6.5,  # effective risk of the CTD exposure
                'currency': 'USD',
                'volatility': 6.5,
                'correlation_to_us10y': 0.95,
                'strike': np.nan,
                'fx_delta': np.nan
            },

            # FX options (two maturities) to *hit policy target* on FX shock reduction
            {
                'instrument_type': 'FX USD/EUR put option',
                'max_notional': 15_000_000,
                'maturity_date': now() + timedelta(days=365 * 1),
                'interest_rate': 0.1,
                'duration': 0.8,
                'currency': 'USD/EUR',
                'volatility': 15.0,
                'correlation_to_us10y': -0.60,
                'strike': 1.05,
                'fx_delta': 0.35
            },
            {
                'instrument_type': 'FX USD/EUR put option',
                'max_notional': 18_000_000,
                'maturity_date': now() + timedelta(days=365 * 2),
                'interest_rate': 0.1,
                'duration': 1.2,
                'currency': 'USD/EUR',
                'volatility': 15.0,
                'correlation_to_us10y': -0.60,
                'strike': 1.05,
                'fx_delta': 0.40
            },

            # FX forward (1y): linear hedge for translation risk without optionality cost
            {
                'instrument_type': 'FX USD/EUR forward',
                'max_notional': 20_000_000,
                'maturity_date': now() + timedelta(days=365 * 1),
                'interest_rate': 0.1,
                'duration': 0.5,
                'currency': 'USD/EUR',
                'volatility': 12.0,
                'correlation_to_us10y': -0.50,
                'strike': np.nan,
                'fx_delta': 0.20  # linear exposure proxy
            },

            # Cross-currency swap (5y): longer-dated FX sensitivity with rate overlay
            {
                'instrument_type': 'Cross-currency swap USD/EUR receive USD pay EUR',
                'max_notional': 25_000_000,
                'maturity_date': now() + timedelta(days=365 * 5),
                'interest_rate': 0.5,
                'duration': 4.0,
                'currency': 'USD/EUR',
                'volatility': 12.0,
                'correlation_to_us10y': 0.40,
                'strike': np.nan,
                'fx_delta': 0.25
            },
        ]

    else:
        raise ValueError("create_hedge_universe(mode=...) must be 'minimal' or 'expanded'.")

    df = pd.DataFrame(data)
    df['maturity_date'] = pd.to_datetime(df['maturity_date'])
    # Save human-friendly, then restore datetime on readback
    df_out = df.copy()
    df_out['maturity_date'] = df_out['maturity_date'].dt.strftime('%Y-%m-%d')
    df_out.to_csv(file_path, index=False)
    print(f"Hedge universe created and saved to {file_path} (mode='{mode}')")
    return df


# ======================================================================================
# MARKET DATA & CALIBRATION
# ======================================================================================

def get_historical_data(tickers: list = ['^TNX', 'EURUSD=X'], start_date: str = None,
                        end_date: str = None) -> pd.DataFrame:
    """
    What is this:
        Downloader for recent market history (10Y UST yield and EUR/USD).

    What it does:
        Uses Yahoo Finance to pull daily closing prices for the requested window.
        Defaults to the last ~12 months if dates are not provided.

    Why:
        Realized volatility (and correlations) should be based on recent data.

    Data used:
        Output: DataFrame of Close prices (columns per ticker). Empty DataFrame on failure.
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Close']
        print(f"Historical data fetched for {tickers} from {start_date} to {end_date}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}. Using empty DataFrame.")
        return pd.DataFrame()

def calibrate_vol_and_corr(historical_data: pd.DataFrame, portfolio_df: pd.DataFrame) -> tuple:
    """
    What is this:
        Simple calibration of (1) daily US 10Y volatility in basis points and
        (2) an average absolute correlation to 10Y (proxy for portfolio rate sensitivity).

    What it does:
        - If ^TNX is present, compute daily change std dev × 100 to get bp vol.
        - Otherwise fall back to 8bp.
        - For correlation, read 'correlation_to_us10y' from the portfolio if present,
          take absolute values and average; else use 0.75.

    Why:
        These two numbers drive VaR scaling and rough risk aggregation for the book.

    Data used:
        Input: historical_data (^TNX), portfolio_df (optional column)
        Output: (us10y_daily_vol_bp: float, mean_abs_corr: float)
    """
    us10y_daily_vol_bp = 8.0
    mean_abs_corr = portfolio_df.get('correlation_to_us10y', pd.Series([0.75])).abs().mean()
    if not historical_data.empty and '^TNX' in historical_data.columns:
        daily_changes = (historical_data['^TNX'] - historical_data['^TNX'].shift(1)).dropna() * 100
        if not daily_changes.empty:
            us10y_daily_vol_bp = float(daily_changes.std())
            print(f"Calibrated US10Y daily volatility: {us10y_daily_vol_bp:.2f} bp")
    print(f"Mean absolute correlation to US10Y: {mean_abs_corr:.2f}")
    return us10y_daily_vol_bp, float(mean_abs_corr)

# ======================================================================================
# RISK ENGINE
# ======================================================================================

def compute_baseline_risks(portfolio_df: pd.DataFrame,
                           us10y_daily_vol_bp: float = 8.0,
                           mean_abs_corr: float = 0.75,
                           rate_shift_bp: float = 0,
                           vol_shift_pct: float = 0,
                           fx_shift_pct: float = 0,
                           curve_mode: str = None) -> dict:
    """
    What is this:
        Core risk calculator for ΔEVE, VaR(95% daily), duration gap, and FX shock impact.

    What it does:
        - Supports two yield-curve modes:
            * parallel (default): same bp shift across all durations
            * steepening/flattening: different bp shifts by bucket (short/mid/long)
        - Computes:
            * ΔEVE under the prescribed rate shift (by duration × notional × shift)
            * Duration gap (weighted asset duration − weighted liability duration)
            * VaR(95% daily) ≈ 1.65 × (10Y bp vol × mean |corr| × DV01 proxy)
            * FX P&L under a % FX move (using 'fx_delta' sensitivities on FX rows)

    Why:
        Produces the apples-to-apples metrics used in head-to-head comparisons.

    Data used:
        Input:
          - portfolio_df: positions (assets positive, liabilities negative notionals)
          - us10y_daily_vol_bp, mean_abs_corr: calibrated risk drivers
          - rate_shift_bp: bp shift size (can be negative)
          - vol_shift_pct: relative increase in vol (for VaR stress)
          - fx_shift_pct: % FX move for FX impact
          - curve_mode: None/'steepening'/'flattening' (curve shape stress)
        Output:
          - dict of headline risk metrics
    """
    df = portfolio_df.copy()

    # ----- Rate shift logic (parallel vs. curve) -----
    if curve_mode in ("steepening", "flattening"):
        # Tag buckets by duration
        short_mask = df['duration'] <= 2.0
        long_mask = df['duration'] >= 5.0

        if curve_mode == "steepening":
            df.loc[short_mask, 'shift_bp'] = rate_shift_bp  # e.g., short rates up
            df.loc[long_mask, 'shift_bp'] = 0
        elif curve_mode == "flattening":
            df.loc[short_mask, 'shift_bp'] = 0
            df.loc[long_mask, 'shift_bp'] = rate_shift_bp   # e.g., long rates up

        # Middle bucket gets a half shift for realism
        mid_mask = ~(short_mask | long_mask)
        df.loc[mid_mask, 'shift_bp'] = rate_shift_bp * 0.5

        # Bucketed ΔEVE
        delta_y = df['shift_bp'] / 10000.0
        delta_eve_assets = -(df.loc[df['notional_amount'] > 0, 'notional_amount']
                             * df.loc[df['notional_amount'] > 0, 'duration']
                             * delta_y[df['notional_amount'] > 0]).sum()
        delta_eve_liabilities = (abs(df.loc[df['notional_amount'] < 0, 'notional_amount'])
                                 * df.loc[df['notional_amount'] < 0, 'duration']
                                 * delta_y[df['notional_amount'] < 0]).sum()
        total_delta_eve = float(delta_eve_assets + delta_eve_liabilities)
    else:
        # Parallel shift
        delta_y = rate_shift_bp / 10000.0
        assets = df[df['notional_amount'] > 0]
        liabilities = df[df['notional_amount'] < 0]
        delta_eve_assets = -(assets['notional_amount'] * assets['duration'] * delta_y).sum() if not assets.empty else 0.0
        delta_eve_liabilities = (abs(liabilities['notional_amount']) * liabilities['duration'] * delta_y).sum() if not liabilities.empty else 0.0
        total_delta_eve = float(delta_eve_assets + delta_eve_liabilities)

    # ----- Duration gap, VaR, and FX impact (independent of curve_mode) -----
    assets = df[df['notional_amount'] > 0]
    liabilities = df[df['notional_amount'] < 0]
    total_assets = float(assets['notional_amount'].sum()) if not assets.empty else 0.0
    total_liabilities = float(abs(liabilities['notional_amount'].sum())) if not liabilities.empty else 0.0
    wad = (assets['notional_amount'] * assets['duration']).sum() / total_assets if total_assets > 0 else 0.0
    wld = (abs(liabilities['notional_amount']) * liabilities['duration']).sum() / total_liabilities if total_liabilities > 0 else 0.0
    duration_gap = float(wad - wld)

    # DV01 proxy and VaR
    dv01 = float((abs(df['notional_amount']) * df['duration']).sum() * 0.0001)
    portfolio_daily_vol_bp = max(0.0, us10y_daily_vol_bp) * max(0.0, mean_abs_corr) * (1 + vol_shift_pct / 100.0)
    var_95_daily = 1.65 * portfolio_daily_vol_bp * dv01

    # FX shock impact (uses instrument-level 'fx_delta' on USD/EUR rows)
    fx_df = df[df['currency'] == 'USD/EUR']
    if not fx_df.empty:
        fx_sens = pd.to_numeric(fx_df.get('fx_delta', 0.5), errors='coerce').fillna(0.5).clip(0.0, 1.0)
        fx_impact = float((fx_df['notional_amount'] * (fx_shift_pct / 100.0) * fx_sens).sum())
    else:
        fx_impact = 0.0

    return {
        'total_assets': total_assets,
        'total_liabilities': total_liabilities,
        'weighted_asset_duration': float(wad),
        'weighted_liab_duration': float(wld),
        'duration_gap': duration_gap,
        'delta_eve': total_delta_eve,
        'var_95_daily_rate': float(var_95_daily),
        'fx_impact': fx_impact,
        'rate_shift_bp': rate_shift_bp,
        'curve_mode': curve_mode or "parallel"
    }

# ======================================================================================
# OPTIMIZER
# ======================================================================================

def optimize_hedge_basket(
    portfolio_df: pd.DataFrame,
    hedge_universe_df: pd.DataFrame,
    us10y_daily_vol_bp: float,
    mean_abs_corr: float,
    *,
    gap_abs_cap: float = 0.80,
    gap_neg_floor: float = -0.20,
    var_cap_pct: float = 0.10,
    fx_improve_target: float = 0.35,
    penalty_weights: Optional[dict] = None,
) -> pd.DataFrame:
    """
    What is this:
        Continuous optimizer that selects hedge notionals within instrument caps.

    What it does:
        - Builds an objective that prefers:
            * Smaller absolute duration gap (toward +0.80y target)
            * Lower ΔEVE @ +100bp
            * Lower VaR
        - Adds quadratic penalties if constraints are breached:
            * Duration-gap band, VaR cap, FX reduction target, and total budget cap.
        - Solves with L-BFGS-B under [0, max_notional] bounds per instrument.

    Why:
        Provides a reproducible, fast optimization step before post-processing/repair.

    Data used:
        Input:
          - portfolio_df, hedge_universe_df (with 'max_notional' per instrument)
          - us10y_daily_vol_bp, mean_abs_corr (for risk metrics)
          - gap_abs_cap, gap_neg_floor, var_cap_pct, fx_improve_target, penalty_weights
        Output:
          - selection_df with chosen instruments and 'selected_notional'. Empty if none.
    """
    if penalty_weights is None:
        penalty_weights = {'gap_abs': 5.0, 'gap_neg': 5.0, 'var': 3.0, 'fx': 5.0, 'total': 4.0}

    n_hedges = len(hedge_universe_df)
    if n_hedges == 0:
        print("Optimization warning: empty hedge universe.")
        return pd.DataFrame()

    max_notionals = hedge_universe_df['max_notional'].values.astype(float)
    max_total_hedge = DEFAULT_CONSTRAINTS.hedge_budget_pct * float(portfolio_df[portfolio_df['notional_amount'] > 0]['notional_amount'].sum())

    base_no = compute_baseline_risks(portfolio_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=0)
    base_var = base_no['var_95_daily_rate']
    base_fx10 = compute_baseline_risks(portfolio_df, us10y_daily_vol_bp, mean_abs_corr, fx_shift_pct=10)['fx_impact']
    fx_guard_active = abs(base_fx10) > 1e-6

    base_cols = list(hedge_universe_df.columns)

    def build_hedged_df(x: np.ndarray) -> pd.DataFrame:
        picks = []
        for i, notional in enumerate(x):
            if notional > 1e-9:
                r = hedge_universe_df.iloc[i].to_dict()
                r['notional_amount'] = -float(notional)
                picks.append(r)
        if picks:
            hedges_df = pd.DataFrame(picks)[base_cols + ['notional_amount']]
            return pd.concat([portfolio_df, hedges_df], ignore_index=True)
        return portfolio_df

    cache: Dict[Tuple[float, ...], Dict[str, dict]] = {}

    def eval_metrics(x: np.ndarray) -> Dict[str, dict]:
        key = tuple(np.round(x, 2))
        hit = cache.get(key)
        if hit is not None:
            return hit
        hedged_df = build_hedged_df(x)
        out = {
            'no': compute_baseline_risks(hedged_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=0),
            'p100': compute_baseline_risks(hedged_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=100),
            'fx10': compute_baseline_risks(hedged_df, us10y_daily_vol_bp, mean_abs_corr, fx_shift_pct=10),
        }
        cache[key] = out
        return out

    def objective(x: np.ndarray) -> float:
        m_no = eval_metrics(x)['no']
        m_p100 = eval_metrics(x)['p100']
        m_fx = eval_metrics(x)['fx10']

        gap_target = 0.80
        gap = abs(m_no['duration_gap'])
        deve = abs(m_p100['delta_eve']) / 1e6
        varv = m_no['var_95_daily_rate'] / 1e6

        # Primary trade-offs
        obj = 0.60 * max(0.0, gap - gap_target) + 0.30 * deve + 0.10 * varv

        # Constraint penalties
        viol_gap_abs = max(0.0, abs(m_no['duration_gap']) - gap_abs_cap)
        obj += penalty_weights['gap_abs'] * (viol_gap_abs ** 2)

        viol_gap_neg = max(0.0, -m_no['duration_gap'] - abs(gap_neg_floor))
        obj += penalty_weights['gap_neg'] * (viol_gap_neg ** 2)

        viol_var = max(0.0, m_no['var_95_daily_rate'] - base_var * (1.0 + var_cap_pct)) / max(1e-9, base_var)
        obj += penalty_weights['var'] * (viol_var ** 2)

        if fx_guard_active:
            target_fx_abs = abs(base_fx10) * (1.0 - fx_improve_target)
            viol_fx = max(0.0, abs(m_fx['fx_impact']) - target_fx_abs) / max(1e-9, abs(base_fx10))
            obj += penalty_weights['fx'] * (viol_fx ** 2)

        viol_total = max(0.0, float(np.sum(x)) - max_total_hedge) / max(1.0, max_total_hedge)
        obj += penalty_weights['total'] * (viol_total ** 2)

        return float(obj)

    # Initialization and bounds
    x0 = np.minimum(max_notionals * 0.1, max_total_hedge / max(1, n_hedges))
    bounds = [(0.0, float(b)) for b in max_notionals]

    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'disp': False, 'maxiter': 500, 'ftol': 1e-9})

    if not res.success:
        print(f"Optimization warning: {res.message}")

    x_star = np.clip(res.x, 0.0, max_notionals)
    if np.all(x_star < 1e-6):
        print("Optimization finished but selected no hedges.")
        return pd.DataFrame()

    out = hedge_universe_df.copy()
    out['selected_notional'] = x_star
    out = out[out['selected_notional'] > 1e-9][
        ['instrument_type', 'selected_notional', 'maturity_date', 'interest_rate', 'duration', 'currency', 'strike']
    ]
    print("Optimal hedge basket selected.")
    return out

# ======================================================================================
# SCENARIO ENGINE
# ======================================================================================

def simulate_scenarios(portfolio_df, hedged_df, us10y_daily_vol_bp, mean_abs_corr) -> pd.DataFrame:
    """
    What is this:
        Batch scenario runner that compares Base vs. Hedged across pre-defined stresses.

    What it does:
        Runs parallel hikes/cuts, vol shocks, FX shocks, and curve shape shocks,
        then returns a tidy table of metrics (duration gap, ΔEVE, VaR, FX impact).

    Why:
        Produces the head-to-head evidence needed for the white paper.

    Data used:
        Input: portfolio_df, hedged_df, risk calibration
        Output: DataFrame (one row per scenario with base/hedged columns)
    """
    scenarios = [
        # Canonical set
        {'name': 'Base (No Shift)', 'rate_shift_bp': 0, 'vol_shift_pct': 0, 'fx_shift_pct': 0},
        {'name': 'Rate +100bps', 'rate_shift_bp': 100, 'vol_shift_pct': 0, 'fx_shift_pct': 0},
        {'name': 'Rate +200bps', 'rate_shift_bp': 200, 'vol_shift_pct': 0, 'fx_shift_pct': 0},
        {'name': 'Vol +20%', 'rate_shift_bp': 0, 'vol_shift_pct': 20, 'fx_shift_pct': 0},
        {'name': 'FX +10%', 'rate_shift_bp': 0, 'vol_shift_pct': 0, 'fx_shift_pct': 10},
        {'name': 'Yield Curve Steepening (+150bp short end)', 'rate_shift_bp': 150, 'curve_mode': 'steepening'},
        {'name': 'Yield Curve Flattening (+150bp long end)', 'rate_shift_bp': 150, 'curve_mode': 'flattening'},

        # Additional stresses used in examples/results
        {'name': 'Rate -100bps', 'rate_shift_bp': -100, 'vol_shift_pct': 0, 'fx_shift_pct': 0},
        {'name': 'Yield Curve Steepening (+150bp short end)', 'rate_shift_bp': 150, 'vol_shift_pct': 0, 'fx_shift_pct': 0},
        {'name': 'Yield Curve Flattening (+150bp long end)', 'rate_shift_bp': 0, 'vol_shift_pct': 0, 'fx_shift_pct': 0, 'custom': 'flatten'},
        {'name': 'Rate +150bps & FX +15%', 'rate_shift_bp': 150, 'vol_shift_pct': 0, 'fx_shift_pct': 15},
        {'name': 'Liquidity Crunch (Vol +40%, FX +10%)', 'rate_shift_bp': 0, 'vol_shift_pct': 40, 'fx_shift_pct': 10},
        {'name': 'Flight to Quality', 'rate_shift_bp': -75, 'vol_shift_pct': 25, 'fx_shift_pct': 5},
    ]

    rows = []
    for sc in scenarios:
        base_results = compute_baseline_risks(
            portfolio_df, us10y_daily_vol_bp, mean_abs_corr,
            rate_shift_bp=sc.get('rate_shift_bp', 0),
            vol_shift_pct=sc.get('vol_shift_pct', 0),
            fx_shift_pct=sc.get('fx_shift_pct', 0),
            curve_mode=sc.get('curve_mode')
        )

        hedged_results = compute_baseline_risks(
            hedged_df, us10y_daily_vol_bp, mean_abs_corr,
            rate_shift_bp=sc.get('rate_shift_bp', 0),
            vol_shift_pct=sc.get('vol_shift_pct', 0),
            fx_shift_pct=sc.get('fx_shift_pct', 0)
        )
        rows.append({
            'Scenario': sc['name'],
            'Duration Gap (Base)': base_results['duration_gap'],
            'Duration Gap (Hedged)': hedged_results['duration_gap'],
            'ΔEVE (Base)': base_results['delta_eve'],
            'ΔEVE (Hedged)': hedged_results['delta_eve'],
            'VaR (Base)': base_results['var_95_daily_rate'],
            'VaR (Hedged)': hedged_results['var_95_daily_rate'],
            'FX Impact (Base)': base_results['fx_impact'],
            'FX Impact (Hedged)': hedged_results['fx_impact']
        })
    print("Scenario simulation completed (extended set).")
    return pd.DataFrame(rows)

# ======================================================================================
# AI UNIVERSE BUILD HELPERS
# ======================================================================================

def _clean_notional_range_to_tuple(s: str) -> tuple:
    """
    What is this:
        Parser that turns strings like "$5,000,000-15,000,000" into (low, high).

    What it does:
        Strips symbols/commas, handles single numbers as (0, n), and returns ints.

    Why:
        Normalizes heterogeneous AI outputs into a numeric cap we can enforce.

    Data used:
        Input: string or number-like
        Output: (low:int, high:int)
    """
    if s is None:
        return (0, 0)
    raw = str(s).replace('$', '').replace(',', '').strip()
    if '-' not in raw:
        try:
            val = int(float(re.sub(r'[^0-9.\-]', '', raw)))
            return (0, val)
        except:
            return (0, 0)
    a, b = raw.split('-', 1)
    try:
        return (int(float(re.sub(r'[^0-9.\-]', '', a))),
                int(float(re.sub(r'[^0-9.\-]', '', b))))
    except:
        return (0, 0)

def _grok_prompt_text(total_assets: float, duration_gap: float, delta_eve_100bps: float) -> str:
    """
    What is this:
        The instruction we send to the AI model to propose a hedge universe.

    What it does:
        Requests 5–9 hedge candidates with maturity, notional caps, and (for FX options)
        an expected delta, under explicit governance targets.

    Why:
        Clear prompts produce structured, auditable AI outputs.

    Data used:
        Input: portfolio totals and gaps to contextualize the ask
        Output: A string prompt (used by multiple providers)
    """
    return (
        "You are a senior bank treasury/ALM structurer.\n"
        "Task: PROPOSE a hedge UNIVERSE as a JSON ARRAY (5–9 items) that a downstream optimizer will select from.\n"
        "Return ONLY a JSON array (no prose). Each element must be an object with keys:\n"
        "{\"instrument_type\": str, \"maturity\": int (years), \"notional_range\": \"low-high\" dollars, "
        "\"strike\": number|null, \"expected_fx_delta\": number in [0,1] (for FX options)}\n"
        "Targets the optimizer must be able to HIT (post-optimization vs baseline):\n"
        "1) |Duration gap| ≤ 0.80y and gap not negative by more than 0.20y.\n"
        "2) ΔEVE under +100bp improves ≥ 45% vs baseline (avoid very large POSITIVE ΔEVE).\n"
        "3) VaR (daily, 95%) ≤ baseline × 1.10.\n"
        "4) FX +10% P&L reduction ≥ 35%.\n\n"
        "Context (USD bank book):\n"
        f"- Total assets: ${total_assets:,.0f}\n"
        f"- Current duration gap (years): {duration_gap:.2f}\n"
        f"- ΔEVE @ +100 bp: ${delta_eve_100bps:,.0f}\n\n"
        "Allowed families: USD IRS (pay-fixed), UST futures (short), USD OIS, FX (USD/EUR) options/forwards, "
        "cross-currency swaps if helpful. You may add other relevant families, but keep keys as specified.\n"
        "Bounds: Sum of 'notional_range' UPPER limits should be ≤ 40% of assets.\n"
        "Return strictly a JSON array with double-quoted keys/strings—no comments, no explanation.\n"
    )

def _balanced_json_array_slice(text: str) -> str:
    """
    What is this:
        Robust extractor that finds the first complete JSON array in an LLM reply.

    What it does:
        Handles code fences, nested brackets, and quoted strings safely.

    Why:
        AI can decorate outputs; we need the literal JSON array only.

    Data used:
        Input: raw text
        Output: string containing exactly one JSON array
    """
    s = str(text)
    fence = re.search(r"```(?:json)?(.*?)```", s, re.S | re.I)
    if fence:
        s = fence.group(1)
    start = s.find('[')
    if start == -1:
        raise ValueError("No '[' found in reply.")
    i, depth, in_str, esc = start, 0, False, False
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '[': depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return s[start:i+1].strip()
        i += 1
    raise ValueError("Unbalanced brackets.")

def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """
    What is this:
        Parser that converts the array slice into Python objects, repairing quotes if needed.

    What it does:
        Attempts json.loads; if it fails due to single quotes, repairs and retries.

    Why:
        Resilient to minor formatting issues while remaining strict about structure.

    Data used:
        Input: text
        Output: list of dicts
    """
    payload = _balanced_json_array_slice(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        repaired = re.sub(r"(?<!\\)'", '"', payload)
        return json.loads(repaired)

def _normalize_grok_universe_items(items: list) -> list:
    """
    What is this:
        Normalizer for diverse AI items (keys, units, ranges) into a consistent schema.

    What it does:
        - Harmonizes key names (instrument_type, maturity, notional_range, strike, expected_fx_delta).
        - Coerces maturities to integer years and notionals to "low-high" dollars.
        - Infers FX deltas for FX options if absent.
        - Accepts string shorthand items and expands them to structured dicts.

    Why:
        Downstream code assumes consistent schema; this shields it from model variance.

    Data used:
        Input: list (mixed dicts/strings)
        Output: list of clean dicts with required keys
    """
    def _as_int_years(v):
        if isinstance(v, (int, float)) and np.isfinite(v):
            return int(v)
        s = str(v); m = re.search(r'(\d+)\s*y', s, re.I)
        return int(m.group(1)) if m else 5

    def _clean_money(s):
        s = str(s).lower().replace('$','').replace(',','').strip()
        s = s.replace('mm','000000').replace('m','000000').replace('bn','000000000')
        try: return int(float(re.sub(r'[^0-9.\-]','',s)))
        except: return 0

    def _coerce_range(v):
        if v is None: return "0-0"
        s = str(v)
        m = re.search(r'([0-9\.,]+)\s*[–-]\s*([0-9\.,]+)\s*([mb]n|mm|m)?', s, re.I)
        if m:
            lo = _clean_money(m.group(1)+(m.group(3) or ''))
            hi = _clean_money(m.group(2)+(m.group(3) or ''))
            if lo and hi and hi>=lo: return f"{lo}-{hi}"
        n = _clean_money(s)
        return f"0-{n}" if n>0 else "0-0"

    def _maybe_strike(s):
        m = re.search(r'@?\s*([0-9]+\.[0-9]+)', str(s))
        try: return float(m.group(1)) if m else None
        except: return None

    out = []
    for item in items:
        if isinstance(item, dict):
            d = dict(item)
            alt = {
                'type':'instrument_type','name':'instrument_type','instrument':'instrument_type','instrumentType':'instrument_type',
                'tenor':'maturity','maturity_years':'maturity','maturityY':'maturity',
                'upper_bound':'notional_range','max_notional':'notional_range','notional':'notional_range','cap':'notional_range',
                'strike_price':'strike','expected_fx_delta':'expected_fx_delta'
            }
            for k, v in alt.items():
                if k in d and v not in d:
                    d[v] = d[k]
            inst = str(d.get('instrument_type','') or d.get('name','') or d.get('type','')).strip()
            maturity = _as_int_years(d.get('maturity', d.get('tenor', d.get('maturity_years','5y'))))
            nr = _coerce_range(d.get('notional_range', d.get('max_notional', d.get('notional','0-0'))))
            strike = d.get('strike', None)
            if strike in ("null","",None):
                strike = None
            elif isinstance(strike,str):
                try: strike = float(strike.replace('@','').strip())
                except: strike = _maybe_strike(strike)
            inst_l = inst.lower()
            fx_delta = d.get('expected_fx_delta', None)
            if fx_delta is None:
                fx_delta = 0.4 if ('option' in inst_l and ('fx' in inst_l or 'usd/eur' in inst_l)) else 0.0
            try: fx_delta = float(fx_delta)
            except: fx_delta = 0.4 if ('option' in inst_l and ('fx' in inst_l or 'usd/eur' in inst_l)) else 0.0
            fx_delta = float(np.clip(fx_delta, 0.0, 1.0))
            out.append({
                'instrument_type': inst or 'Unknown',
                'maturity': int(maturity),
                'notional_range': nr,
                'strike': strike,
                'expected_fx_delta': fx_delta,
            })
        elif isinstance(item, str):
            s = item.strip()
            inst = re.sub(r'\(\s*\d+\s*y\s*\)', '', s).strip()
            mat = _as_int_years(s)
            strike = _maybe_strike(s)
            is_fx_opt = 'option' in s.lower() and ('fx' in s.lower() or 'usd/eur' in s.lower())
            nr = "8000000-15000000" if is_fx_opt else "8000000-16000000"
            out.append({
                'instrument_type': inst,
                'maturity': mat,
                'notional_range': nr,
                'strike': strike,
                'expected_fx_delta': 0.4 if is_fx_opt else 0.0,
            })
    return out

def _build_universe_df_from_grok(proposed_hedges: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    What is this:
        Converter from normalized AI items to the numeric universe DataFrame.

    What it does:
        - Maps instrument families to rate/FX attributes (duration, vol, correlation).
        - Computes maturity_date from (today + maturity years).
        - Carries max_notional (upper end of notional_range) as the cap.

    Why:
        The optimizer needs a concrete table (one row per candidate with numeric attrs).

    Data used:
        Input: list of normalized dicts
        Output: DataFrame with columns used across the toolkit
    """
    rows = []
    for hedge in proposed_hedges:
        n_low, n_high = _clean_notional_range_to_tuple(hedge.get('notional_range','0-0'))
        maturity_years = int(hedge.get('maturity', 1))
        itype_raw = str(hedge.get('instrument_type','Unknown')).strip()
        itype_l = itype_raw.lower()
        fx_delta = float(hedge.get('expected_fx_delta', 0)) if 'expected_fx_delta' in hedge else np.nan

        if ('cross-currency' in itype_l) or ('cross currency' in itype_l) or ('ccs' in itype_l):
            ir, dur, cur, vol, corr, strike = 0.5, max(0.4, maturity_years*0.8), 'USD/EUR', 12.0, 0.4, np.nan
        elif ('ois' in itype_l):
            ir, dur, cur, vol, corr, strike = 0.5, max(0.3, maturity_years*0.5), 'USD', 8.0, 0.7, np.nan
        elif ('future' in itype_l) or ('ust' in itype_l):
            ir, dur, cur, vol, corr, strike = 3.8, max(0.4, maturity_years*0.7), 'USD', 6.5, 0.95, np.nan
        elif 'option' in itype_l and ('fx' in itype_l or 'usd/eur' in itype_l):
            ir, dur, cur, vol, corr, strike = 0.1, max(0.3, maturity_years*0.6), 'USD/EUR', 15.0, -0.6, hedge.get('strike', np.nan)
        elif 'forward' in itype_l and ('fx' in itype_l or 'usd/eur' in itype_l):
            ir, dur, cur, vol, corr, strike = 0.1, max(0.2, maturity_years*0.5), 'USD/EUR', 12.0, -0.5, np.nan
        elif ('swap' in itype_l) or ('irs' in itype_l):
            ir, dur, cur, vol, corr, strike = 4.0, max(0.5, maturity_years*0.9), 'USD', 10.0, 0.9, np.nan
        else:
            ir, dur, cur, vol, corr, strike = 0.5, max(0.2, maturity_years*0.5), 'USD', 10.0, 0.5, hedge.get('strike', np.nan)

        rows.append({
            'instrument_type': itype_raw,
            'max_notional': int(n_high),
            'maturity_date': (datetime.now() + timedelta(days=365*maturity_years)).strftime('%Y-%m-%d'),
            'interest_rate': ir,
            'duration': float(dur),
            'currency': cur,
            'volatility': float(vol),
            'correlation_to_us10y': float(corr),
            'strike': (np.nan if (hedge.get('strike', None) in [None, "null"]) else strike),
            'fx_delta': fx_delta,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['maturity_date'] = pd.to_datetime(df['maturity_date'])
    return df

# ======================================================================================
# LLM PROVIDERS (AI calls)
# ======================================================================================

def _xai_call(messages: List[Dict[str, str]], model: str, connect_to: float, read_to: float) -> str:
    """
    What is this:
        Low-level client for x.ai (Grok) chat completions.

    What it does:
        Sends messages to the selected model and returns text content.

    Why:
        Isolates provider specifics; orchestration code stays clean.

    Data used:
        Input: messages, model, timeouts (requires XAI_API_KEY env var)
        Output: provider text response
    """
    url = "https://api.x.ai/v1/chat/completions"
    key = os.getenv("XAI_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 2000, "temperature": 0.25}
    resp = requests.post(url, headers=headers, json=payload, timeout=(connect_to, read_to))
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")

def _openai_call(messages: List[Dict[str, str]], model: str = None) -> str:
    """
    What is this:
        Low-level client for OpenAI chat completions.

    What it does:
        Sends messages to the selected model and returns text content.

    Why:
        Provider abstraction.

    Data used:
        Input: messages, model (requires OPENAI_API_KEY)
        Output: provider text response
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer " + key, "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1600}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", ""

)

def _anthropic_call(messages: List[Dict[str, str]], model: str = None, system: Optional[str] = None) -> str:
    """
    What is this:
        Low-level client for Anthropic chat completions.

    What it does:
        Sends messages (with optional system prompt) and returns concatenated text.

    Why:
        Provider abstraction.

    Data used:
        Input: messages, optional system, model (requires ANTHROPIC_API_KEY)
        Output: provider text response
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    if model is None:
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    # Anthropic separates 'system' and 'messages'
    sys = system or ""
    user_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    payload = {"model": model, "max_tokens": 1600, "temperature": 0.2, "messages": user_msgs}
    if sys:
        payload["system"] = sys
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # content is a list of blocks
    blocks = data.get("content", [])
    text = ""
    for b in blocks:
        if b.get("type") == "text":
            text += b.get("text", "")
    return text

# ======================================================================================
# HEURISTIC LAST-RESORT GENERATOR (DYNAMIC, NOT FIXED MENU)
# ======================================================================================

def _heuristic_universe_generator(total_assets: float, duration_gap: float, delta_eve_100bps: float) -> List[Dict[str, Any]]:
    """
    What is this:
        A programmatic universe generator used only if all AI providers fail and no
        cached universe exists. It still adapts to inputs and randomness (not fixed).

    What it does:
        - Allocates a ~40% asset “budget” into 5–9 instruments with randomization.
        - Chooses instrument families/maturities sensibly based on duration_gap.
        - Adds FX items with realistic deltas and optional strikes.

    Why:
        Ensures we ALWAYS have a workable universe so the orchestrator never stalls.

    Data used:
        Input: total_assets, duration_gap, delta_eve_100bps (for mild steering)
        Output: list of instrument dicts (same schema the AI would return)
    """
    rng = random.Random(42 + int(abs(duration_gap)*1000))
    items: List[Dict[str, Any]] = []
    k = rng.randint(5, 9)

    # Budget cap ~ 40% assets split across k with randomness
    cap_total = 0.40 * total_assets
    raw = np.array([rng.uniform(0.5, 1.5) for _ in range(k)])
    weights = raw / raw.sum()
    uppers = (weights * cap_total).astype(int)

    # Duration strategy
    long_bias = min(10, max(2, int( (duration_gap if duration_gap>0 else 2.0) * 3 )))
    rate_mats = [2,3,5,7,10,12]
    fx_mats = [1,2,3]
    fx_deltas = [0.30,0.35,0.40,0.45,0.50]

    families = [
        "USD Pay-Fixed IRS",
        "2Y UST Future (Short)",
        "5Y UST Future (Short)",
        "10Y UST Future (Short)",
        "USD OIS (Receive-Floating)",
        "USD/EUR Call Option",
        "USD/EUR Put Option",
        "USD/EUR Forward",
        "USD/EUR Cross-Currency Swap"
    ]
    rng.shuffle(families)

    for i in range(k):
        fam = families[i % len(families)]
        if "USD/EUR" in fam:
            mat = rng.choice(fx_mats)
            strike = round(rng.uniform(0.95, 1.15), 2) if "Option" in fam else None
            fx_delta = rng.choice(fx_deltas) if "Option" in fam else 0.20
        elif "OIS" in fam:
            mat = rng.choice([2,3,5])
            strike, fx_delta = None, 0.0
        elif "Future" in fam:
            mat = rng.choice([2,5,10])
            strike, fx_delta = None, 0.0
        else:  # IRS or other USD rates
            mat = rng.choice(rate_mats if duration_gap > 0 else [2,3,5])
            strike, fx_delta = None, 0.0

        upper = max(5_000_000, int(uppers[i]))
        lower = int(upper * rng.uniform(0.3, 0.6))
        items.append({
            "instrument_type": fam,
            "maturity": int(mat),
            "notional_range": f"{lower}-{upper}",
            "strike": strike,
            "expected_fx_delta": fx_delta
        })

    return items

# ======================================================================================
# PROPOSE VIA AI (NEVER RETURNS EMPTY)
# ======================================================================================

def propose_hedge_basket(portfolio_metrics: dict) -> list:
    """
    What is this:
        High-level entry to obtain a hedge universe for the given portfolio metrics.

    What it does:
        Calls the iterative AI refine function with a sensible default number of tries.

    Why:
        Single-call interface for the orchestrator/agent.

    Data used:
        Input: dict with total_assets, duration_gap, delta_eve_100bps
        Output: list of proposed instruments (normalized dicts)
    """
    return propose_and_refine_hedge_universe(portfolio_metrics, baseline_metrics=None, max_iters=4)

def propose_and_refine_hedge_universe(portfolio_metrics: dict,
                                      baseline_metrics: Optional[dict] = None,
                                      max_iters: int = 4) -> list:
    """
    What is this:
        Multi-provider, closed-loop proposal that ALWAYS returns a universe.

    What it does:
        Attempt order:
          1) x.ai (Grok) with iterative refinement using _UNIVERSE_EVALUATOR feedback
          2) OpenAI
          3) Anthropic
          4) Load last_good_universe.json if present
          5) Heuristic synthesis
        Normalizes and caps universe (≤ 50% assets), persists CSV and cache file.

    Why:
        Resilience and auditability. The AI can fail or be unreachable; we still deliver.

    Data used:
        Input: portfolio metrics (assets, gap, ΔEVE@+100bp)
        Output: list of instrument dicts (schema defined in normalizer)
    """
    total_assets = float(portfolio_metrics['total_assets'])
    duration_gap = float(portfolio_metrics['duration_gap'])
    delta_eve_100bps = float(portfolio_metrics['delta_eve_100bps'])

    # --- common prompt & messages
    system_msg = {"role": "system",
                  "content": "You are a treasury/ALM expert. Respond with ONLY valid JSON arrays for hedge universes."}
    user_prompt = _grok_prompt_text(total_assets, duration_gap, delta_eve_100bps)

    def _try_refine_via_provider(provider: str) -> Optional[list]:
        """What is this: A provider-specific refinement loop (kept internal)."""
        best_json = None
        best_score = -1e18
        context: List[Dict[str, str]] = []

        def _score_df(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
            global _UNIVERSE_EVALUATOR
            if '_UNIVERSE_EVALUATOR' not in globals() or _UNIVERSE_EVALUATOR is None:
                return 0.0, {'feedback': 'Evaluator not wired', 'duration_gap_ok': True, 'var_ok': True, 'fx_ok': True, 'overshoot_neg_gap': False}
            ev = _UNIVERSE_EVALUATOR(df)
            return float(ev.get('score', 0.0)), ev

        for it in range(max_iters):
            msgs = [system_msg] + context + [{"role": "user", "content": user_prompt}]
            text = ""
            try:
                if provider == "xai":
                    model = os.getenv("GROK_MODEL", "grok-code-fast-1")
                    text = _xai_call(msgs, model=model,
                                     connect_to=float(os.getenv("GROK_CONNECT_TIMEOUT","8")),
                                     read_to=float(os.getenv("GROK_READ_TIMEOUT","30")))
                elif provider == "openai":
                    text = _openai_call(msgs)
                elif provider == "anthropic":
                    text = _anthropic_call(msgs, system=system_msg["content"])
                else:
                    return None
            except Exception as e:
                logger.warning(f"[propose:{provider}] call failed on iter {it+1}: {e}")
                # tighten guardrails and retry
                context.append({"role":"system","content":"Return ONLY a JSON array. No prose, no comments."})
                continue

            if not str(text).strip():
                context.append({"role":"system","content":"Your reply was empty. Return ONLY a JSON array now."})
                continue

            try:
                raw = _extract_json_array(text)
                normalized = _normalize_grok_universe_items(raw)
                if not normalized:
                    raise ValueError("normalized array is empty")
            except Exception as e:
                logger.warning(f"[propose:{provider}] parse repair needed: {e}")
                context.append({"role":"system",
                                "content":"Return ONLY a JSON array with objects having keys instrument_type, maturity, "
                                          "notional_range ('low-high'), strike (or null), expected_fx_delta (0..1 for FX options)."})
                continue

            df = _build_universe_df_from_grok(normalized)
            # enforce universe notional cap ≤ 50% assets
            cap = DEFAULT_CONSTRAINTS.hedge_budget_pct * total_assets
            s = float(pd.to_numeric(df['max_notional'], errors='coerce').fillna(0).sum())
            if s > cap and s > 0:
                df = df.copy()
                df['max_notional'] = pd.to_numeric(df['max_notional'], errors='coerce').fillna(0.0) * (cap / s)

            score, ev = _score_df(df)
            if score > best_score:
                best_score, best_json = score, normalized

            ok_gap = bool(ev.get('duration_gap_ok', True))
            ok_var = bool(ev.get('var_ok', True))
            ok_fx = bool(ev.get('fx_ok', True))
            overshoot_neg = bool(ev.get('overshoot_neg_gap', False))
            fb = str(ev.get('feedback',''))
            print(f"[{provider}] iter {it+1}: score={score:.1f} ok_gap={ok_gap} ok_var={ok_var} ok_fx={ok_fx} overshoot_neg={overshoot_neg}")
            if fb: print(f"[{provider}] feedback:\n{fb}")

            if ok_gap and ok_var and ok_fx and not overshoot_neg:
                return normalized

            context.append({"role":"user",
                            "content": "Revise the universe per these measured outcomes:\n"
                                       f"{fb}\n"
                                       "Increase FX option headroom (delta ~0.35–0.45) if needed; trim rates 15–30% if VaR high. "
                                       "Sum of max notionals ≤ 40% of assets. Return ONLY a JSON array."})
        return best_json  # return best attempt even if not perfect

    # 1) Grok
    result = None
    if os.getenv("XAI_API_KEY"):
        result = _try_refine_via_provider("xai")
    # 2) OpenAI
    if (result is None or not result) and os.getenv("OPENAI_API_KEY"):
        result = _try_refine_via_provider("openai")
    # 3) Anthropic
    if (result is None or not result) and os.getenv("ANTHROPIC_API_KEY"):
        result = _try_refine_via_provider("anthropic")

    # 4) Persisted last_good
    if (result is None or not result):
        try:
            with open("last_good_universe.json","r") as f:
                cached = json.load(f)
            if isinstance(cached, list) and cached:
                print("[propose] Using persisted last_good_universe.json")
                result = cached
        except Exception:
            result = None

    # 5) Heuristic synthesis (last resort, still dynamic — not a fixed basket)
    if (result is None or not result):
        print("[propose] All LLMs unavailable. Synthesizing a heuristic universe.")
        result = _heuristic_universe_generator(total_assets, duration_gap, delta_eve_100bps)

    # Persist result as the new last_good for resilience
    try:
        with open("last_good_universe.json","w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to persist last_good_universe.json: {e}")

    # Append/merge to CSV for inspection
    try:
        current = pd.read_csv('hedge_universe.csv')
        current['maturity_date'] = pd.to_datetime(current.get('maturity_date', pd.Series([], dtype='datetime64[ns]')), errors='coerce')
    except Exception:
        current = pd.DataFrame(columns=[
            'instrument_type','max_notional','maturity_date','interest_rate','duration','currency',
            'volatility','correlation_to_us10y','strike','fx_delta'
        ])
    new_df = _build_universe_df_from_grok(result)
    new_df['maturity_date_str'] = new_df['maturity_date'].dt.strftime('%Y-%m-%d')
    if not current.empty:
        current['maturity_date_str'] = current['maturity_date'].dt.strftime('%Y-%m-%d')
    for _, r in new_df.iterrows():
        duplicate = False if current.empty else (
            (current['instrument_type'] == r['instrument_type']) &
            (current['maturity_date_str'] == r['maturity_date_str'])
        ).any()
    # If not duplicate, append for transparency
        if not duplicate:
            add_row = r.drop(labels=['maturity_date_str']).to_dict()
            current = pd.concat([current, pd.DataFrame([add_row])], ignore_index=True)
    if 'maturity_date_str' in current.columns:
        current = current.drop(columns=['maturity_date_str'])
    current.to_csv('hedge_universe.csv', index=False)

    print("Proposed hedge basket appended to hedge_universe.csv")
    return result

# ======================================================================================
# BUILD HEDGED BOOK & POST-PROCESSOR
# ======================================================================================

def _build_hedged_df(portfolio_df: pd.DataFrame,
                     universe_df: pd.DataFrame,
                     notionals_vec: np.ndarray) -> pd.DataFrame:
    """
    What is this:
        Deterministic constructor that appends hedge rows onto the portfolio.

    What it does:
        - For each positive notional in 'notionals_vec', append the corresponding
          universe row with 'notional_amount' negative (hedge).
        - Returns the combined DataFrame with aligned columns.

    Why:
        Single point of truth for how hedges are injected into the book.

    Data used:
        Input: portfolio_df, universe_df, notionals_vec
        Output: combined DataFrame (portfolio + hedges)
    """
    x = np.asarray(notionals_vec, dtype=float)
    if len(x) != len(universe_df):
        raise ValueError("notionals_vec length must match universe_df length")
    picks = []
    for i, notional in enumerate(x):
        if notional > 1e-9:
            r = universe_df.iloc[i].to_dict()
            r['notional_amount'] = -float(notional)
            picks.append(r)
    if not picks:
        return portfolio_df
    hedges_df = pd.DataFrame(picks)
    if 'maturity_date' in hedges_df.columns and hedges_df['maturity_date'].dtype == object:
        try:
            hedges_df['maturity_date'] = pd.to_datetime(hedges_df['maturity_date'])
        except Exception:
            pass
    all_cols = list({*portfolio_df.columns, *hedges_df.columns})
    portfolio_aligned = portfolio_df.reindex(columns=all_cols)
    hedges_aligned = hedges_df.reindex(columns=all_cols)
    return pd.concat([portfolio_aligned, hedges_aligned], ignore_index=True)

def postprocess_hedge_selection(
    portfolio_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    us10y_daily_vol_bp: float,
    mean_abs_corr: float,
    *,
    gap_abs_cap: float = 0.80,
    gap_neg_floor: float = -0.20,
    var_cap_pct: float = 0.10,
    fx_improve_target: float = 0.35,
    max_iter: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    What is this:
        Post-optimizer “tightening” loop that enforces constraints precisely.

    What it does:
        Iteratively:
          - Caps VaR via bisection on USD rate hedges.
          - Tops up FX options to achieve the FX +10% reduction target.
          - Nudges duration gap back into band if off (trim long tenors / scale).
          - Uses remaining VaR headroom to approach +0.80y target gap.
        Stops when no meaningful change is detected or after max_iter.

    Why:
        The optimizer balances many terms; this step guarantees compliance and
        converts “good” into “governed”.

    Data used:
        Input: portfolio_df, universe_df, initial selection_df, risk inputs, limits
        Output: repaired selection_df (same schema), ready for scenarios/printing
    """
    if selection_df.empty:
        return selection_df.copy()

    base_no = compute_baseline_risks(portfolio_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=0)
    base_var = base_no['var_95_daily_rate']
    base_fx10 = compute_baseline_risks(portfolio_df, us10y_daily_vol_bp, mean_abs_corr, fx_shift_pct=10)['fx_impact']
    fx_guard_active = abs(base_fx10) > 1e-6
    var_cap = base_var * (1.0 + var_cap_pct)

    def selection_to_vector(sel: pd.DataFrame) -> np.ndarray:
        vec = np.zeros(len(universe_df), dtype=float)
        if sel.empty:
            return vec
        uni_key = (universe_df['instrument_type'].astype(str).str.lower().str.strip()
                   + '|' + pd.to_datetime(universe_df['maturity_date']).dt.strftime('%Y-%m-%d'))
        key_to_idx = {k: i for i, k in enumerate(uni_key)}
        for _, row in sel.iterrows():
            k = (str(row['instrument_type']).lower().strip()
                 + '|' + pd.to_datetime(row['maturity_date']).strftime('%Y-%m-%d'))
            i = key_to_idx.get(k)
            if i is not None:
                vec[i] = float(row['selected_notional'])
        return vec

    def vector_to_selection(vec: np.ndarray) -> pd.DataFrame:
        out = universe_df.copy()
        out['selected_notional'] = vec
        out = out[out['selected_notional'] > 1e-9][
            ['instrument_type','selected_notional','maturity_date','interest_rate','duration','currency','strike']
        ]
        return out

    x = selection_to_vector(selection_df)
    max_notionals = universe_df['max_notional'].values.astype(float)

    is_rate = (universe_df['currency'].astype(str).str.upper() == 'USD')
    is_fx_option = universe_df['instrument_type'].astype(str).str.contains('Option', case=False, na=False) & \
                   (universe_df['currency'].astype(str).str.upper() == 'USD/EUR')

    def eval_metrics(xvec: np.ndarray) -> dict:
        hedged_df = _build_hedged_df(portfolio_df, universe_df, xvec)
        return {
            'no': compute_baseline_risks(hedged_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=0),
            'p100': compute_baseline_risks(hedged_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=100),
            'fx10': compute_baseline_risks(hedged_df, us10y_daily_vol_bp, mean_abs_corr, fx_shift_pct=10)
        }

    def apply_bisection_on_rates_for_var(xvec: np.ndarray) -> np.ndarray:
        if not is_rate.any():
            return xvec
        m = eval_metrics(xvec)
        if m['no']['var_95_daily_rate'] <= var_cap:
            return xvec
        low, high = 0.5, 1.0
        for _ in range(18):
            mid = 0.5 * (low + high)
            x_try = xvec.copy()
            x_try[is_rate] *= mid
            if eval_metrics(x_try)['no']['var_95_daily_rate'] <= var_cap:
                high = mid
            else:
                low = mid
        x_new = xvec.copy()
        x_new[is_rate] *= high
        return x_new

    def topup_fx_option_for_target(xvec: np.ndarray) -> np.ndarray:
        if not fx_guard_active or not is_fx_option.any():
            return xvec
        met = eval_metrics(xvec)
        fx_abs_now = abs(met['fx10']['fx_impact'])
        target_abs = abs(base_fx10) * (1.0 - fx_improve_target)
        if fx_abs_now <= target_abs:
            return xvec
        x_new = xvec.copy()
        idxs = np.where(is_fx_option)[0]
        for i in idxs:
            if max_notionals[i] <= 0:
                continue
            cur = x_new[i]
            lo, hi = cur, max_notionals[i]
            if hi <= lo + 1e-6:
                continue
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                x_try = x_new.copy()
                x_try[i] = mid
                fx_abs_try = abs(eval_metrics(x_try)['fx10']['fx_impact'])
                if fx_abs_try <= target_abs:
                    hi = mid
                else:
                    lo = mid
            x_new[i] = hi
        x_new = apply_bisection_on_rates_for_var(x_new)
        return x_new

    def nudge_gap_band(xvec: np.ndarray) -> np.ndarray:
        m = eval_metrics(xvec)
        gap_now = m['no']['duration_gap']
        x_new = xvec.copy()
        if gap_now < gap_neg_floor:
            long_rate = is_rate & (universe_df['duration'] >= universe_df['duration'].quantile(0.5))
            x_new[long_rate] *= 0.95
        elif abs(gap_now) > gap_abs_cap:
            x_new[is_rate] *= 0.95
        return x_new

    for it in range(max_iter):
        changed = False
        met = eval_metrics(x)
        if met['no']['var_95_daily_rate'] > var_cap + 1.0:
            x2 = apply_bisection_on_rates_for_var(x)
            if not np.allclose(x2, x, rtol=0, atol=1e-6):
                x = x2; changed = True
                if verbose: print(f"[post] Iter {it}: scaled rates for VaR cap.")
        if fx_guard_active:
            fx_abs = abs(met['fx10']['fx_impact'])
            target_fx_abs = abs(base_fx10) * (1.0 - fx_improve_target)
            if fx_abs > target_fx_abs + 1.0:
                x2 = topup_fx_option_for_target(x)
                if not np.allclose(x2, x, rtol=0, atol=1e-6):
                    x = x2; changed = True
                    if verbose: print(f"[post] Iter {it}: topped-up FX option and re-capped VaR.")
        m_after = eval_metrics(x)
        gap_now = m_after['no']['duration_gap']
        if (gap_now < gap_neg_floor - 1e-3) or (abs(gap_now) > gap_abs_cap + 1e-3):
            x2 = nudge_gap_band(x)
            if not np.allclose(x2, x, rtol=0, atol=1e-6):
                x = x2; changed = True
                if verbose: print(f"[post] Iter {it}: nudged gap toward band.")
        if not changed:
            break

    # Use remaining VaR headroom to approach +0.80y target, carefully
    gap_target = 0.80
    var_tol = 250.0

    def current_metrics(xvec):
        m_no = eval_metrics(xvec)['no']
        return m_no['duration_gap'], m_no['var_95_daily_rate']

    gap_now, var_now = current_metrics(x)
    if (var_now < var_cap - var_tol) and (abs(gap_now) > gap_target) and is_rate.any():
        low, high = 1.0, min(1.25, (var_cap / max(var_now, 1.0)))
        best = x.copy()
        for _ in range(18):
            mid = 0.5 * (low + high)
            x_try = x.copy()
            x_try[is_rate] = np.minimum(x_try[is_rate] * mid, max_notionals[is_rate])
            g_try, v_try = current_metrics(x_try)
            if v_try <= var_cap + 1.0:
                best = x_try
                if abs(g_try) <= gap_target + 1e-3:
                    high = mid
                else:
                    low = mid
            else:
                high = mid
        x = best

    x = np.minimum(x, max_notionals)
    return vector_to_selection(x)
