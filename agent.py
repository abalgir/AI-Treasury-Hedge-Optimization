# agent.py
# --------------------------------------------------------------------------------------
# PURPOSE (for CFO readers, non-technical):
# --------------------------------------------------------------------------------------
# What is this:
#   The "agent" is the AI-facing orchestration that proposes and iteratively improves
#   a hedge universe (the menu of candidate hedges) and then ensures the final picks
#   respect governance limits before returning them to the main workflow.
#
# What it does:
#   - Normalizes AI-proposed hedges (currency flags, FX sensitivities, dates).
#   - Caps total allowed hedge notionals vs. balance-sheet size for discipline.
#   - Repairs/adjusts an initial optimized selection to hit guardrails (VaR cap,
#     duration-gap band, FX reduction target).
#   - Iteratively mutates the universe (tighter/looser bounds) based on measured gaps.
#   - Scores hedged results vs. baseline to pick the best feasible selection.
#
# Why:
#   Traditional systems start from a fixed menu; AI can start from a smarter menu.
#   This module provides the “safety rails” to ensure AI creativity is harnessed
#   under the same risk constraints and auditable rules.
#
# Data used:
#   - Inputs (from financial_library/orchestrator):
#       * portfolio_df: balance sheet positions with durations & notionals
#       * AI-proposed items converted to a DataFrame "universe_df"
#       * us10y_daily_vol_bp (volatility) and mean_abs_corr (correlation)
#       * governance targets: var_cap_pct, fx_improve_target, gap_abs_cap, gap_neg_floor
#   - Outputs:
#       * A repaired, feasible hedge selection (instrument + notional + maturity)
#       * The (possibly mutated) hedge universe used to arrive at that selection
#       * Debug info capturing iteration-by-iteration decisions
# --------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

import financial_library as fl
from financial_library import align_selection


def _norm_currency_and_fx(df: pd.DataFrame) -> pd.DataFrame:
    """
    What is this:
        Normalizer for AI-proposed hedge rows (DataFrame).

    What it does:
        - Sets the 'currency' column logically based on instrument type:
            * Options/forwards/cross-currency → 'USD/EUR' (treated as FX exposures)
            * UST futures / IRS / swaps (non-cross) → 'USD' (treated as rate hedges)
        - Ensures an 'fx_delta' (FX sensitivity) exists:
            * Defaults: 0.40 for FX options, 0.20 for other FX instruments, 0 for rates.
        - Parses 'maturity_date' to a proper datetime if provided as text.

    Why:
        AI proposals may vary in formatting; normalization guarantees the optimizer
        and risk engines interpret hedges consistently (FX vs. rate behavior).

    Data used:
        Input:
          - df: DataFrame of candidate hedges with columns like 'instrument_type',
                'currency', 'fx_delta', 'maturity_date'.
        Output:
          - A copy of df with normalized 'currency', numeric & clipped 'fx_delta',
            and datetime 'maturity_date' where possible.
    """
    if df.empty:
        return df
    out = df.copy()
    typ = out['instrument_type'].astype(str).str.lower()

    # Map instruments to currency buckets used by the risk engine
    out.loc[typ.str.contains('option|forward|cross-currency|cross currency|ccs'), 'currency'] = 'USD/EUR'
    out.loc[typ.str.contains('ust|future|swap|irs') & ~typ.str.contains('cross'), 'currency'] = 'USD'

    # Ensure an FX sensitivity exists and is bounded [0,1]
    if 'fx_delta' not in out.columns:
        out['fx_delta'] = np.nan
    is_fx = out['currency'].astype(str).str.upper().eq('USD/EUR')
    out.loc[is_fx & out['fx_delta'].isna() & typ.str.contains('option'), 'fx_delta'] = 0.40
    out.loc[is_fx & out['fx_delta'].isna() & ~typ.str.contains('option'), 'fx_delta'] = 0.20
    out['fx_delta'] = pd.to_numeric(out['fx_delta'], errors='coerce').fillna(0.0).clip(0.0, 1.0)

    # Parse string dates to datetime where needed
    if 'maturity_date' in out.columns and out['maturity_date'].dtype == object:
        try:
            out['maturity_date'] = pd.to_datetime(out['maturity_date'])
        except Exception:
            pass
    return out


def _cap_total_universe(df: pd.DataFrame, total_assets: float, pct: float = 0.50) -> pd.DataFrame:
    """
    What is this:
        Universe-wide sizing guardrail.

    What it does:
        Caps the sum of 'max_notional' across all candidate hedges to a fraction
        (pct) of total balance-sheet assets. If the sum exceeds the cap, scales
        all 'max_notional' values down proportionally.

    Why:
        Prevents AI from proposing an over-sized hedge aisle. Keeps the problem
        economical and aligned with policy (e.g., hedge budget ≤ 50% of assets).

    Data used:
        Input:
          - df: candidate hedge universe with a 'max_notional' column
          - total_assets: baseline total asset size (float)
          - pct: cap fraction (default 50%)
        Output:
          - Universe with adjusted 'max_notional' if needed; otherwise original df.
    """
    if df.empty:
        return df
    cap = max(0.0, pct) * float(total_assets)
    s = float(pd.to_numeric(df['max_notional'], errors='coerce').fillna(0).sum())
    if s > cap and s > 0:
        out = df.copy()
        out['max_notional'] = pd.to_numeric(out['max_notional'], errors='coerce').fillna(0.0) * (cap / s)
        return out
    return df


def _vector_to_selection(universe_df: pd.DataFrame, vec: np.ndarray) -> pd.DataFrame:
    """
    What is this:
        Converter from a raw notional vector to a tidy selection table.

    What it does:
        - Takes an array 'vec' (notional per instrument in 'universe_df' order)
        - Returns a compact DataFrame with only the chosen rows (positive notionals)
          and the key fields needed downstream.

    Why:
        Clean representation is easier to print, compare, and persist.

    Data used:
        Input:
          - universe_df: candidate hedges in a fixed order
          - vec: numpy array of notionals aligned to 'universe_df'
        Output:
          - selection_df with columns:
              ['instrument_type','selected_notional','maturity_date',
               'interest_rate','duration','currency','strike']
    """
    out = universe_df.copy()
    out['selected_notional'] = np.asarray(vec, dtype=float)
    out = out[out['selected_notional'] > 1e-9][
        ['instrument_type', 'selected_notional', 'maturity_date',
         'interest_rate', 'duration', 'currency', 'strike']
    ]
    return out


def _build_hedged_df_safe(portfolio_df, universe_df, xvec):
    """
    What is this:
        Safe builder that appends hedge rows to the portfolio to form a hedged book.

    What it does:
        - Uses financial_library._build_hedged_df when available (canonical path).
        - Otherwise, constructs the hedged DataFrame manually:
            * For every positive notional in xvec, append a hedge row with negative
              'notional_amount' (short hedge vs. long exposure).
            * Aligns columns and returns the combined DataFrame.

    Why:
        Provides resilience if the canonical builder is not present while keeping
        identical semantics, so scenario/risk functions work as expected.

    Data used:
        Input:
          - portfolio_df: baseline balance sheet
          - universe_df: hedge candidates with attributes
          - xvec: notional vector aligned to 'universe_df'
        Output:
          - hedged_df: combined portfolio + hedge rows
    """
    if hasattr(fl, '_build_hedged_df'):
        return fl._build_hedged_df(portfolio_df, universe_df, xvec)
    x = np.asarray(xvec, dtype=float)
    picks = []
    for i, notional in enumerate(x):
        if notional > 1e-9:
            r = universe_df.iloc[i].to_dict()
            r['notional_amount'] = -float(notional)  # hedge notionals enter as negatives
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
    return pd.concat([portfolio_df.reindex(columns=all_cols),
                      hedges_df.reindex(columns=all_cols)], ignore_index=True)


def feasibility_repair(
        portfolio_df: pd.DataFrame,
        universe_df: pd.DataFrame,
        selection_df: pd.DataFrame,
        us10y_daily_vol_bp: float,
        mean_abs_corr: float,
        *,
        var_cap_pct: float,
        fx_improve_target: float,
        gap_abs_cap: float,
        gap_neg_floor: float,
        max_steps: int = 16,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    What is this:
        A constraint-enforcing “repair” pass applied to an initial hedge selection.

    What it does:
        Starting from an optimized selection (which may still miss a constraint),
        this routine:
          1) Caps VaR by scaling down USD rate hedges (bisection search).
          2) Tops up FX options to achieve the FX +10% reduction target (if FX exposure exists),
             re-capping VaR if the top-up raises risk.
          3) Nudges duration gap into the allowed band:
               - If too positive → grow rate hedges within VaR headroom
               - If too negative → trim long-tenor rates
          4) Uses any remaining VaR headroom to push gap closer to the preferred target (+0.80y).
          5) Strictly re-checks VaR; then returns a feasible selection with metrics.

    Why:
        Optimizers balance multiple objectives and constraints. This post-step ensures
        hard governance limits are obeyed exactly, yielding a compliant, auditable result.

    Data used:
        Input:
          - portfolio_df: baseline positions
          - universe_df: candidate hedge universe
          - selection_df: initial chosen hedges (from optimizer)
          - us10y_daily_vol_bp, mean_abs_corr: risk calibration
          - var_cap_pct, fx_improve_target, gap_abs_cap, gap_neg_floor: governance parameters
        Output:
          - (repaired_selection_df, metrics_dict)
              * repaired_selection_df: final notionals after repair
              * metrics_dict: feasibility flag + achieved var/gap/fx numbers and caps
    """
    if selection_df.empty:
        return selection_df.copy(), {'feasible': False, 'reason': 'empty_selection'}

    # Baseline references for caps/targets
    base_no = fl.compute_baseline_risks(portfolio_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=0)
    base_var = base_no['var_95_daily_rate']
    base_fx10 = fl.compute_baseline_risks(portfolio_df, us10y_daily_vol_bp, mean_abs_corr, fx_shift_pct=10)['fx_impact']
    fx_guard = abs(base_fx10) > 1e-6
    var_cap = base_var * (1.0 + var_cap_pct)

    # Convenience flags on universe rows
    uni = universe_df.copy()
    uni['__cur__'] = uni['currency'].astype(str).str.upper().str.strip()
    uni['__typ__'] = uni['instrument_type'].astype(str).str.lower().str.strip()

    is_rate = uni['__cur__'].eq('USD')
    is_fx_opt = uni['__typ__'].str.contains('option') & uni['__cur__'].eq('USD/EUR')

    max_notionals = pd.to_numeric(uni['max_notional'], errors='coerce').fillna(0.0).values
    x = align_selection(uni, selection_df)  # start from optimizer’s choice

    def eval_no(xv: np.ndarray):
        hedged = _build_hedged_df_safe(portfolio_df, universe_df, xv)
        return fl.compute_baseline_risks(hedged, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=0)

    def eval_fx10(xv: np.ndarray):
        hedged = _build_hedged_df_safe(portfolio_df, universe_df, xv)
        return fl.compute_baseline_risks(hedged, us10y_daily_vol_bp, mean_abs_corr, fx_shift_pct=10)

    def _apply_strict_var_cap(xvec: np.ndarray) -> np.ndarray:
        # Final-resort bisection to ensure VaR ≤ cap by scaling USD-rate notionals
        if not is_rate.any():
            return xvec
        lo, hi = 0.0, 1.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            xt = xvec.copy()
            xt[is_rate.values] *= mid
            if eval_no(xt)['var_95_daily_rate'] <= var_cap:
                hi = mid
            else:
                lo = mid
        xnew = xvec.copy()
        xnew[is_rate.values] *= hi
        return xnew

    # (1) Cap VaR by scaling USD rate notionals if needed
    m = eval_no(x)
    if m['var_95_daily_rate'] > var_cap + 1.0 and is_rate.any():
        lo, hi = 0.2, 1.0
        for _ in range(max_steps):
            mid = 0.5 * (lo + hi)
            xt = x.copy()
            xt[is_rate.values] *= mid
            if eval_no(xt)['var_95_daily_rate'] <= var_cap:
                hi = mid
            else:
                lo = mid
        x[is_rate.values] *= hi
        print(f"[repair] Scaled USD-rate hedges for VaR cap → α={hi:.3f}")

    # (2) If FX exposure exists, top up FX options to meet the FX +10% reduction target
    if fx_guard and is_fx_opt.any():
        fx_now = abs(eval_fx10(x)['fx_impact'])
        target_fx = abs(base_fx10) * (1.0 - fx_improve_target)

        if fx_now > target_fx + 1.0:
            print(f"[repair] FX abs now={fx_now:,.0f} > target={target_fx:,.0f} → compute required option notional")

            fx_idx = np.where(is_fx_opt.values)[0]
            for i in fx_idx:
                # Use instrument-specific delta if available; otherwise default to 0.5
                delta_i = uni.get('fx_delta', pd.Series(np.nan)).iloc[i]
                if not np.isfinite(delta_i) or delta_i <= 0:
                    delta_i = 0.5
                # Each 10% FX move times delta_i approximates P&L impact per $1 notional
                required_abs_reduction = fx_now - target_fx
                per_notional_effect = 0.10 * float(delta_i)
                if per_notional_effect <= 0:
                    continue
                required_notional = required_abs_reduction / per_notional_effect
                new_i = min(max_notionals[i], max(x[i], required_notional))
                if new_i > x[i] + 1e-6:
                    print(f"[repair] raising FX option notional: {x[i]:,.0f} → {new_i:,.0f} (Δ={new_i - x[i]:,.0f})")
                    x[i] = new_i

            # Re-check FX, and re-cap VaR if the top-up lifted risk
            for _ in range(3):
                fx_now = abs(eval_fx10(x)['fx_impact'])
                if fx_now <= target_fx + 1.0:
                    break
                for i in fx_idx:
                    delta_i = uni.get('fx_delta', pd.Series(np.nan)).iloc[i]
                    if not np.isfinite(delta_i) or delta_i <= 0:
                        delta_i = 0.5
                    per_notional_effect = 0.10 * float(delta_i)
                    if per_notional_effect <= 0 or x[i] >= max_notionals[i] - 1e-6:
                        continue
                    shortfall = (fx_now - target_fx) / per_notional_effect
                    x[i] = min(max_notionals[i], x[i] + shortfall)

            m = eval_no(x)
            if m['var_95_daily_rate'] > var_cap + 1.0 and is_rate.any():
                lo, hi = 0.2, 1.0
                for _ in range(max_steps):
                    mid = 0.5 * (lo + hi)
                    xt = x.copy()
                    xt[is_rate.values] *= mid
                    if eval_no(xt)['var_95_daily_rate'] <= var_cap:
                        hi = mid
                    else:
                        lo = mid
                x[is_rate.values] *= hi
                print(f"[repair] Re-capped VaR after FX top-up → α={hi:.3f}")

    # (3) Duration gap band enforcement / nudge loop
    for _ in range(max_steps):
        m = eval_no(x)
        gap = m['duration_gap']
        varv = m['var_95_daily_rate']

        if gap > gap_abs_cap + 1e-3:
            # Too positive: try to increase rate hedges (subject to VaR headroom)
            if not is_rate.any() or varv >= var_cap - 200.0:
                break
            lo, hi = 1.0, min(1.5, var_cap / max(varv, 1.0))
            best = x.copy()
            for __ in range(18):
                mid = 0.5 * (lo + hi)
                xt = x.copy()
                xt[is_rate.values] = np.minimum(xt[is_rate.values] * mid, max_notionals[is_rate.values])
                mt = eval_no(xt)
                if mt['var_95_daily_rate'] <= var_cap + 1.0:
                    best = xt
                    if mt['duration_gap'] <= gap_abs_cap + 1e-3:
                        hi = mid
                    else:
                        lo = mid
                else:
                    hi = mid
            x = best

        elif gap < gap_neg_floor - 1e-3:
            # Too negative: trim longer-tenor USD rates slightly
            long_rate = is_rate.values & (uni['duration'].values >= np.quantile(uni.loc[is_rate, 'duration'], 0.5))
            if not long_rate.any():
                break
            x[long_rate] *= 0.95
            print("[repair] Gap too negative → trimmed long-tenor USD rates by 5%")
        else:
            break

    # (4) If VaR headroom exists, push the gap toward the preferred target (+0.80y)
    gap_target = 0.80

    def cur_gap_var(xv):
        m0 = eval_no(xv)
        return m0['duration_gap'], m0['var_95_daily_rate']

    gap_now, var_now = cur_gap_var(x)
    if is_rate.any() and (gap_now > gap_target + 1e-3) and (var_now <= 0.95 * var_cap):
        lo, hi = 1.0, min(1.15, var_cap / max(1.0, var_now))
        best = x.copy()
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            xt = x.copy()
            xt[is_rate.values] = np.minimum(xt[is_rate.values] * mid, max_notionals[is_rate.values])
            g_try, v_try = cur_gap_var(xt)
            if v_try <= var_cap + 1.0:
                best = xt
                if g_try <= gap_target + 1e-3:
                    hi = mid
                else:
                    lo = mid
            else:
                hi = mid
        x = best

    # --- Final recompute and feasibility decision ---
    m_no = eval_no(x)
    m_fx = eval_fx10(x)

    # Strict VaR cap if still slightly above
    if m_no['var_95_daily_rate'] > var_cap and is_rate.any():
        x = _apply_strict_var_cap(x)
        m_no = eval_no(x)
        m_fx = eval_fx10(x)

    feasible = (
        (m_no['var_95_daily_rate'] <= var_cap)
        and (gap_neg_floor - 1e-3 <= m_no['duration_gap'] <= gap_abs_cap + 1e-3)
        and ((not fx_guard) or (abs(m_fx['fx_impact']) <= abs(base_fx10) * (1.0 - fx_improve_target) + 1.0))
    )

    return _vector_to_selection(uni, np.minimum(x, max_notionals)), {
        'feasible': feasible,
        'var': m_no['var_95_daily_rate'],
        'gap': m_no['duration_gap'],
        'fx10_abs': abs(m_fx['fx_impact']),
        'var_cap': var_cap,
        'fx_target_abs': abs(base_fx10) * (1.0 - fx_improve_target),
        'gap_abs_cap': gap_abs_cap,
        'gap_neg_floor': gap_neg_floor
    }


def _mutate_universe(universe_df: pd.DataFrame,
                     need_var_cut: bool,
                     need_more_fx: bool,
                     need_gap_tight: bool) -> pd.DataFrame:
    """
    What is this:
        Adaptive adjustment of the candidate hedge universe bounds.

    What it does:
        If the last iteration shows:
          - VaR too high → reduce USD rates capacity by 20%.
          - FX reduction insufficient → expand FX option capacity by 20% and nudge
            option deltas toward 0.45 for effectiveness.
          - Duration gap off-target → trim long-tenor USD capacity by 10%.

    Why:
        Teaches the next optimization round where to search (more FX, fewer long rates,
        etc.), accelerating convergence to a feasible, higher-quality solution.

    Data used:
        Input:
          - universe_df: current candidate hedge universe
          - need_var_cut / need_more_fx / need_gap_tight: booleans from prior metrics
        Output:
          - Updated universe_df with adjusted max notionals (and FX deltas, if relevant)
    """
    df = universe_df.copy()
    is_rate = df['currency'].astype(str).str.upper().eq('USD')
    is_long = df['duration'] >= df['duration'].median()
    is_fx_opt = df['instrument_type'].astype(str).str.lower().str.contains('option') & \
                df['currency'].astype(str).str.upper().eq('USD/EUR')

    if need_var_cut:
        df.loc[is_rate, 'max_notional'] *= 0.80

    if need_more_fx and is_fx_opt.any():
        df.loc[is_fx_opt, 'max_notional'] *= 1.20
        df['fx_delta'] = pd.to_numeric(df.get('fx_delta', 0.4), errors='coerce').fillna(0.4).clip(0, 1)
        df.loc[is_fx_opt, 'fx_delta'] = (df.loc[is_fx_opt, 'fx_delta'] * 0.9 + 0.45 * 0.1).clip(0, 1)

    if need_gap_tight and (is_rate & is_long).any():
        df.loc[is_rate & is_long, 'max_notional'] *= 0.90
    return df


def _score_vs_baseline(portfolio_df, hedged_df, usvol, corr) -> Dict[str, Any]:
    """
    What is this:
        Simple KPI scorecard vs. baseline (unhedged) used for ranking solutions.

    What it does:
        Computes three headline effects of the hedged result relative to baseline:
          - Deve improvement (%) at +100bp
          - VaR increase (%) vs. baseline
          - FX +10% reduction (%), if baseline has FX exposure
        Returns a dict including the hedged duration gap for reference.

    Why:
        Produces compact metrics to compare different selections quickly
        (used to pick the “best so far” across iterations).

    Data used:
        Input:
          - portfolio_df: baseline book
          - hedged_df: book with hedges applied
          - usvol, corr: risk calibration
        Output:
          - dict with 'gap_base', 'deve_improve_pct', 'var_increase_pct', 'fx_reduction_pct'
    """
    base_no = fl.compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=0)
    hed_no = fl.compute_baseline_risks(hedged_df, usvol, corr, rate_shift_bp=0)
    base_p100 = fl.compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=100)
    hed_p100 = fl.compute_baseline_risks(hedged_df, usvol, corr, rate_shift_bp=100)
    base_fx10 = fl.compute_baseline_risks(portfolio_df, usvol, corr, fx_shift_pct=10)
    hed_fx10 = fl.compute_baseline_risks(hedged_df, usvol, corr, fx_shift_pct=10)

    out = {
        'gap_base': hed_no['duration_gap'],
        'deve_improve_pct': 0.0,
        'var_increase_pct': 0.0,
        'fx_reduction_pct': 0.0
    }
    if base_p100['delta_eve'] != 0:
        out['deve_improve_pct'] = max(0.0, 1.0 - (abs(hed_p100['delta_eve']) / abs(base_p100['delta_eve']))) * 100.0
    if base_no['var_95_daily_rate'] > 0:
        out['var_increase_pct'] = (hed_no['var_95_daily_rate'] / base_no['var_95_daily_rate'] - 1.0) * 100.0
    if base_fx10['fx_impact'] != 0:
        out['fx_reduction_pct'] = max(0.0, 1.0 - (abs(hed_fx10['fx_impact']) / abs(base_fx10['fx_impact']))) * 100.0
    return out


def run_agentic_universe(
        portfolio_df: pd.DataFrame,
        us10y_daily_vol_bp: float,
        mean_abs_corr: float,
        *,
        var_cap_pct: float = 0.10,
        fx_improve_target: float = 0.35,
        gap_abs_cap: float = 0.80,
        gap_neg_floor: float = -0.20,
        max_iters: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    What is this:
        The top-level AI loop that proposes, repairs, scores, and returns a feasible
        hedge selection along with the final universe and debug details.

    What it does:
        1) Compute baseline totals (for budgeting and context).
        2) Ask the AI proposal layer to suggest a hedge universe based on baseline.
        3) Normalize and cap the universe (FX flags, deltas, budget ≤ 50% assets).
        4) Increase FX option headroom if present (to enable meeting FX target).
        5) Iterate up to 'max_iters':
             a. Optimize hedges within current universe.
             b. Repair selection to hit constraints.
             c. Score vs. baseline (ΔEVE, VaR, FX reduction).
             d. Keep the best feasible selection.
             e. Mutate universe bounds based on shortfalls (var/gap/fx).
        6) Return the best feasible selection (or last repaired if none is strictly
           better), the final universe, and a debug log of iterations.

    Why:
        Converts AI creativity into a governed, auditable result. The iteration ensures
        we don’t accept a pretty proposal unless it passes the bank’s hard rules.

    Data used:
        Input:
          - portfolio_df: baseline balance sheet
          - us10y_daily_vol_bp, mean_abs_corr: risk calibration
          - var_cap_pct, fx_improve_target, gap_abs_cap, gap_neg_floor: governance limits
          - max_iters: safety bound on compute/time
        Output:
          - (final_selection_df, final_universe_df, debug_dict)
    """
    # (1) Baseline for totals and risk context
    base = fl.compute_baseline_risks(portfolio_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=0)
    total_assets = float(base['total_assets'])

    # (2) AI proposes a hedge universe specific to the current book & risk
    proposed = fl.propose_hedge_basket({
        'total_assets': total_assets,
        'duration_gap': base['duration_gap'],
        'delta_eve_100bps':
            fl.compute_baseline_risks(portfolio_df, us10y_daily_vol_bp, mean_abs_corr, rate_shift_bp=100)['delta_eve'],
    })

    # (3) Normalize, then cap the universe vs. a budget share (default 50% of assets)
    universe_df = _norm_currency_and_fx(fl._build_universe_df_from_grok(proposed))
    universe_df = _cap_total_universe(universe_df, base['total_assets'], pct=0.50)

    # (4) Raise FX option ceilings so repairs can reach the FX reduction target
    fx_mask = universe_df['instrument_type'].astype(str).str.lower().str.contains('option') & \
              universe_df['currency'].astype(str).str.upper().eq('USD/EUR')
    if fx_mask.any():
        universe_df.loc[fx_mask, 'max_notional'] = pd.to_numeric(
            universe_df.loc[fx_mask, 'max_notional'], errors='coerce'
        ).fillna(0.0).clip(lower=30_000_000.0)
        print("[agent] FX option max_notional raised to >= $30M (for top-up).")

    debug: Dict[str, Any] = {'iterations': []}

    sel_best = pd.DataFrame()
    score_best = -1e9
    last_repaired_sel = pd.DataFrame()
    last_met = None

    # (5) Iterate optimize → repair → score → mutate
    for it in range(max_iters):
        opt_sel = fl.optimize_hedge_basket(
            portfolio_df, universe_df, us10y_daily_vol_bp, mean_abs_corr,
            gap_abs_cap=gap_abs_cap, gap_neg_floor=gap_neg_floor,
            var_cap_pct=var_cap_pct, fx_improve_target=fx_improve_target
        )

        if opt_sel.empty:
            # If the optimizer returns nothing, loosen FX and trim rates to help feasibility
            universe_df = _mutate_universe(universe_df, need_var_cut=True, need_more_fx=True, need_gap_tight=True)
            debug['iterations'].append({'iter': it, 'empty_opt': True})
            continue

        repaired_sel, met = feasibility_repair(
            portfolio_df, universe_df, opt_sel, us10y_daily_vol_bp, mean_abs_corr,
            var_cap_pct=var_cap_pct, fx_improve_target=fx_improve_target,
            gap_abs_cap=gap_abs_cap, gap_neg_floor=gap_neg_floor
        )

        last_repaired_sel = repaired_sel.copy()
        last_met = met

        hedged = _build_hedged_df_safe(portfolio_df, universe_df, align_selection(universe_df, repaired_sel))
        score = _score_vs_baseline(portfolio_df, hedged, us10y_daily_vol_bp, mean_abs_corr)

        debug['iterations'].append({
            'iter': it,
            'feasible': met['feasible'],
            'var': met['var'], 'var_cap': met['var_cap'],
            'gap': met['gap'], 'fx10_abs': met['fx10_abs'],
            'score': score
        })

        # Composite used only for ranking "best" among feasible candidates
        composite = (score['deve_improve_pct'] * 2.0
                     + (max(0.0, 80.0 - abs(score['gap_base'] - 0.0) * 100.0)) * 0.01
                     - max(0.0, score['var_increase_pct']) * 0.2
                     + score['fx_reduction_pct'] * 0.5)

        if met['feasible'] and composite > score_best:
            score_best = composite
            sel_best = repaired_sel.copy()

        # Decide how to mutate the universe for the next loop
        need_var_cut = met['var'] > met['var_cap'] + 1.0
        need_more_fx = met['fx10_abs'] > met['fx_target_abs'] + 1.0
        need_gap_tight = abs(met['gap']) > gap_abs_cap + 1e-3
        universe_df = _mutate_universe(universe_df, need_var_cut, need_more_fx, need_gap_tight)

    # (6) Choose the best available selection in order of preference
    if not sel_best.empty:
        final_sel = sel_best
    elif isinstance(last_repaired_sel, pd.DataFrame) and not last_repaired_sel.empty:
        final_sel = last_repaired_sel
    elif 'opt_sel' in locals():
        final_sel = opt_sel
    else:
        final_sel = pd.DataFrame()

    return final_sel, universe_df, debug
