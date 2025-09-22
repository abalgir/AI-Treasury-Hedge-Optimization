# orchestrator.py
# --------------------------------------------------------------------------------------
# PURPOSE :
# --------------------------------------------------------------------------------------
# What is this:
#   The "orchestrator" runs an end-to-end treasury-hedging evaluation. It executes three
#   comparable paths under the same rules and scenarios:
#     (A) Traditional-Minimal  : legacy 3-line static catalog
#     (B) Traditional-Expanded : larger static catalog (mirrors AI families)
#     (C) AI (Agent)           : AI-proposed dynamic catalog tailored to this portfolio
#
# What it does:
#   1) Loads the current balance sheet (assets & liabilities) and reference rate data.
#   2) Calibrates risk inputs from recent market history.
#   3) Runs the two traditional menus through the same optimizer & guardrails.
#   4) Lets an AI agent propose a bespoke hedge universe; optimizes & applies guardrails.
#   5) Runs a common set of risk scenarios for apples-to-apples comparison.
#   6) Prints a head-to-head summary and saves a machine-readable audit file.
#
# Why:
#   To demonstrate, with evidence, whether an AI-curated hedge menu can outperform
#   fixed menus across key risk metrics (Duration Gap, ΔEVE @ +100bp, VaR, and FX
#   shock reduction) while respecting guardrails (e.g., VaR cap).
#
# Data used:
#   - treasury_portfolio.csv  : balance sheet positions (amounts, durations, currency).
#   - prices.csv              : static rate references (kept for parity; not used later).
#   - Market history (fetched): recent US 10Y yield & EUR/USD for volatility & correlation.
#   - hedge_universe_min.csv  : written here as the minimal traditional catalog.
#   - hedge_universe_expanded.csv : written here as the expanded traditional catalog.
#   - run_summary.json        : output artifact with selections and scenario tables.
# --------------------------------------------------------------------------------------

import json
import random
import numpy as np

# --------------------------------------------------------------------------------------
# IMPORTED ENGINES (kept as black boxes for CFO clarity)
# --------------------------------------------------------------------------------------
# What is this:
#   External modules that do the heavy lifting (AI proposal, risk math, optimization).
#
# What they do:
#   - agent: proposes an AI-generated hedge "universe" (menu of candidate hedges).
#   - financial_library: loads data, calibrates vol/corr, computes risks, optimizes,
#     post-processes hedge selections, and runs scenario simulations.
#
# Why:
#   Keeps this file focused on orchestration (the business workflow) while delegating
#   technical details to tested components.
#
# Data used:
#   These modules read/write CSV/JSON artifacts as directed by the orchestrator below.
import agent  # agentic loop
import financial_library as fl
from financial_library import (
    load_portfolio, load_static_rates, create_hedge_universe,
    get_historical_data, calibrate_vol_and_corr,
    compute_baseline_risks, optimize_hedge_basket,
    simulate_scenarios, postprocess_hedge_selection,
    _build_hedged_df, DEFAULT_CONSTRAINTS, align_selection
)


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _seed_everything(seed: int = 42):
    """
    What is this:
        Deterministic seeding helper.

    What it does:
        Sets the random number generators to a fixed seed so that identical inputs
        always produce identical outputs.

    Why:
        Reproducibility is essential for governance, audit, and explainability. It
        avoids “mysterious” differences between two runs.

    Data used:
        None (only affects internal randomness within this process).
    """
    random.seed(seed)
    np.random.seed(seed)


def _write_run_artifact(path, payload: dict):
    """
    What is this:
        JSON exporter for an audit-friendly run summary.

    What it does:
        Writes 'payload' (a Python dictionary of constraints, selections, and
        scenario results) to a JSON file at 'path', with readable indentation.

    Why:
        Creates a single portable artifact (“source of truth”) that Risk, Internal
        Audit, or consultants can open to reproduce tables/charts for the white paper
        and governance packs.

    Data used:
        Input:
          - payload: dictionary assembled later from computed results.
        Output:
          - A JSON file (e.g., 'run_summary.json') written to disk.
    """
    try:
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        print(f"Artifact write failed: {e}")


def _evaluate_universe_for_closed_loop(portfolio_df, universe_df, usvol, corr):
    """
    What is this:
        A closed-loop "fitness check" for AI proposals.

    What it does:
        Given a candidate hedge universe (the menu proposed by AI), it:
          1) Optimizes hedge notionals within that menu.
          2) Post-processes to tighten constraints (cap VaR; top-up FX options to
             meet the FX-reduction target; nudge duration gap into the band).
          3) Rebuilds a hedged portfolio from the chosen hedges.
          4) Computes key risk metrics baseline vs hedged (no shift, +100bp, FX +10%).
          5) Tests guardrails (gap, VaR, FX reduction) and computes a composite score.
          6) Returns pass/fail flags and feedback for the AI’s next iteration.

    Why:
        Enables iterative improvement of the AI’s proposed menu via measured feedback.

    Data used:
        Input:
          - portfolio_df: the bank’s current balance sheet positions.
          - universe_df: the AI-proposed hedge menu to evaluate.
          - usvol, corr: market volatility and correlation (calibrated earlier).
        Output:
          - Dictionary with gap/var/fx flags, a composite score, and feedback text.
    """
    opt = optimize_hedge_basket(portfolio_df, universe_df, usvol, corr)
    if opt.empty:
        return {
            'duration_gap_ok': False, 'var_ok': False, 'fx_ok': False,
            'overshoot_neg_gap': False, 'score': -999, 'feedback': "Empty optimal selection."
        }

    opt_pp = postprocess_hedge_selection(
        portfolio_df, universe_df, opt, usvol, corr,
        gap_abs_cap=0.80, gap_neg_floor=-0.20, var_cap_pct=0.10, fx_improve_target=0.35,
        max_iter=10, verbose=False
    )

    xvec = align_selection(universe_df, opt_pp)
    hedged_df = _build_hedged_df(portfolio_df, universe_df, xvec)

    base_no = compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=0)
    hed_no  = compute_baseline_risks(hedged_df,  usvol, corr, rate_shift_bp=0)
    base_fx10 = compute_baseline_risks(portfolio_df, usvol, corr, fx_shift_pct=10)
    hed_fx10  = compute_baseline_risks(hedged_df,  usvol, corr, fx_shift_pct=10)
    base_p100 = compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=100)
    hed_p100  = compute_baseline_risks(hedged_df,  usvol, corr, rate_shift_bp=100)

    var_cap = base_no['var_95_daily_rate'] * 1.10
    fx_target_abs = abs(base_fx10['fx_impact']) * (1.0 - 0.35)
    gap_ok = (-0.20 <= hed_no['duration_gap'] <= 0.80)
    var_ok = (hed_no['var_95_daily_rate'] <= var_cap + 1.0)
    fx_ok  = (abs(hed_fx10['fx_impact']) <= fx_target_abs + 1.0) if abs(base_fx10['fx_impact']) > 1e-6 else True
    deve_improve = max(0.0, 1.0 - (abs(hed_p100['delta_eve']) / max(1e-6, abs(base_p100['delta_eve'])))) * 100.0

    feedback = []
    if not gap_ok: feedback.append(f"- |Duration gap| = {abs(hed_no['duration_gap']):.2f}y (cap 0.80y).")
    if not var_ok: feedback.append(f"- VaR {hed_no['var_95_daily_rate']:.0f} > cap {var_cap:.0f}.")
    if not fx_ok:  feedback.append("- FX +10% reduction below 35%.")
    if deve_improve < 45.0: feedback.append(f"- ΔEVE +100bp improvement {deve_improve:.0f}% < 45% target.")

    score = (deve_improve * 2.0
             - max(0.0, abs(hed_no['duration_gap']) - 0.80) * 100.0
             - max(0.0, (hed_no['var_95_daily_rate'] / max(1e-6, base_no['var_95_daily_rate']) - 1.10)) * 100.0
             + (35.0 - max(0.0, (abs(hed_fx10['fx_impact']) / max(1e-6, abs(base_fx10['fx_impact'])) * 100.0 - 65.0))))

    return {
        'duration_gap_ok': gap_ok, 'var_ok': var_ok, 'fx_ok': fx_ok,
        'overshoot_neg_gap': (hed_no['duration_gap'] < -0.20),
        'score': score, 'feedback': "\n".join(feedback) or "OK"
    }


def _run_traditional(label: str,
                     portfolio_df,
                     universe_df,
                     usvol: float,
                     corr: float):
    """
    What is this:
        Helper to run a *static* benchmark through the same optimizer & guardrails
        used elsewhere (no AI loop, no evaluator feedback).

    What it does:
        - Optimizes selection vs the given universe, then post-processes to enforce
          VaR cap, FX reduction, and the duration-gap band.
        - Builds hedged book and runs scenarios.

    Why:
        Avoids duplication and guarantees identical treatment across benchmarks.

    Data used:
        Input: portfolio_df, universe_df, calibrated usvol & corr.
        Output: selection, hedged_df, scenarios table.
    """
    print(f"\n--- {label} (Traditional) ---")
    opt = optimize_hedge_basket(portfolio_df, universe_df, usvol, corr)
    sel = postprocess_hedge_selection(
        portfolio_df, universe_df, opt, usvol, corr,
        gap_abs_cap=0.80, gap_neg_floor=-0.20, var_cap_pct=0.10, fx_improve_target=0.35,
        max_iter=12, verbose=True
    )
    xvec = align_selection(universe_df, sel)
    hedged = _build_hedged_df(portfolio_df, universe_df, xvec)
    scen = simulate_scenarios(portfolio_df, hedged, usvol, corr)

    print(f"\nOptimal Hedge Basket ({label}):")
    print("Warning: No hedges selected." if sel.empty else sel)
    print(f"\nScenario Results ({label}):")
    print(scen.to_string(index=False))
    return sel, hedged, scen


# --------------------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------------------
def main():
    """
    What is this:
        The end-to-end workflow entry point.

    What it does:
        1) Seeds randomness.
        2) Loads portfolio & static rates.
        3) Builds two traditional catalogs (minimal & expanded).
        4) Fetches market data and calibrates risk inputs.
        5) Computes baseline risks (pre-hedge).
        6) Runs the two traditional benchmarks through optimizer + guardrails + scenarios.
        7) Runs the AI path with closed-loop evaluator; scenarios as well.
        8) Prints head-to-head KPIs and writes an audit JSON artifact.

    Why:
        One command produces a transparent, reproducible comparison.

    Data used:
        Input: treasury_portfolio.csv, prices.csv, market data (US10Y, EUR/USD).
        Output: console tables & run_summary.json.
    """
    print("Starting treasury risk computation, hedging proposal, and scenario testing workflow...")
    _seed_everything(42)  # determinism for reproducibility

    # 1) Load
    portfolio_df = load_portfolio('treasury_portfolio.csv')
    _ = load_static_rates('prices.csv')  # parity; not directly used below

    # 2) Hedge universes (traditional catalogs)
    #    If your create_hedge_universe doesn’t support 'mode', fall back to minimal.
    try:
        hard_uni_min = create_hedge_universe('hedge_universe_min.csv', mode='minimal')
    except TypeError:
        hard_uni_min = create_hedge_universe('hedge_universe_min.csv')
    try:
        hard_uni_exp = create_hedge_universe('hedge_universe_expanded.csv', mode='expanded')
    except TypeError:
        hard_uni_exp = hard_uni_min  # fallback: reuse minimal if expanded not available

    # 3) Market data and calibration
    hist = get_historical_data(tickers=['^TNX', 'EURUSD=X'])
    usvol, corr = calibrate_vol_and_corr(hist, portfolio_df)

    # 4) Baseline (pre-hedge)
    base_no = compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=0)
    base_p100 = compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=100)

    # 5) Run both traditional benchmarks (identical treatment)
    hard_min_sel, hard_min_hedged, hard_min_scen = _run_traditional("Traditional-Minimal", portfolio_df, hard_uni_min, usvol, corr)
    hard_exp_sel, hard_exp_hedged, hard_exp_scen = _run_traditional("Traditional-Expanded", portfolio_df, hard_uni_exp, usvol, corr)

    # 6) AI path: plug in evaluator for closed-loop refinement
    fl._UNIVERSE_EVALUATOR = lambda df: _evaluate_universe_for_closed_loop(portfolio_df, df, usvol, corr)

    print("\n--- Grok Suggested Hedges (Agent) ---")
    grok_sel, grok_universe, dbg = agent.run_agentic_universe(
        portfolio_df, usvol, corr,
        var_cap_pct=0.10, fx_improve_target=0.35, gap_abs_cap=0.80, gap_neg_floor=-0.20, max_iters=3
    )
    grok_xvec = align_selection(grok_universe, grok_sel)
    grok_hedged = _build_hedged_df(portfolio_df, grok_universe, grok_xvec)
    grok_scen = simulate_scenarios(portfolio_df, grok_hedged, usvol, corr)

    print("\nOptimal Hedge Basket (Grok-Agent):")
    print("Warning: No hedges selected by agent." if grok_sel.empty else grok_sel)
    print("\nScenario Results (Grok-Agent):")
    print(grok_scen.to_string(index=False))

    # 7) Head-to-head summary (all three paths)
    def _metrics(df_hedged):
        return {
            'no': compute_baseline_risks(df_hedged, usvol, corr, rate_shift_bp=0),
            'p100': compute_baseline_risks(df_hedged, usvol, corr, rate_shift_bp=100),
            'fx10': compute_baseline_risks(df_hedged, usvol, corr, fx_shift_pct=10)
        }

    hard_min_m = _metrics(hard_min_hedged)
    hard_exp_m = _metrics(hard_exp_hedged)
    grok_m     = _metrics(grok_hedged)

    def pct_improve(a, b):
        """
        What is this:
            Utility to compute percentage improvement from a baseline value 'a'
            to an after-hedge value 'b'.

        What it does:
            Returns max(0, 1 - |b|/|a|) * 100 (%). If 'b' is much smaller than 'a'
            in absolute terms, the improvement is large.

        Why:
            Provides a clean, consistent ΔEVE improvement metric for the head-to-head
            comparison.
        """
        return max(0.0, 1.0 - (abs(b) / max(1e-6, abs(a)))) * 100.0

    var_cap = base_no['var_95_daily_rate'] * 1.10
    base_fx_abs = abs(compute_baseline_risks(portfolio_df, usvol, corr, fx_shift_pct=10)['fx_impact'])

    def fx_red(m_fx10):  # %
        return (1.0 - abs(m_fx10['fx_impact']) / max(1e-6, base_fx_abs)) * 100.0 if base_fx_abs != 0 else 0.0

    print("\nBaseline Risk Metrics (Pre-Hedge):")
    for k, v in base_no.items():
        if k in ('weighted_asset_duration', 'weighted_liab_duration', 'duration_gap'):
            print(f"{k.replace('_', ' ').title()}: {v:.2f}")
        elif isinstance(v, (int, float)):
            print(f"{k.replace('_', ' ').title()}: ${v:,.0f}")
        else:
            print(f"{k.replace('_', ' ').title()}: {v}")

    print("\n=== Head-to-Head (measured) ===")
    print(f"Gap (base):  minimal {hard_min_m['no']['duration_gap']:.3f}y  "
          f"|  expanded {hard_exp_m['no']['duration_gap']:.3f}y  "
          f"|  AI {grok_m['no']['duration_gap']:.3f}y")
    print(f"ΔEVE +100bp improvement:  "
          f"minimal {pct_improve(base_p100['delta_eve'], hard_min_m['p100']['delta_eve']):.1f}%  "
          f"|  expanded {pct_improve(base_p100['delta_eve'], hard_exp_m['p100']['delta_eve']):.1f}%  "
          f"|  AI {pct_improve(base_p100['delta_eve'], grok_m['p100']['delta_eve']):.1f}%")
    print(f"VaR:  minimal {hard_min_m['no']['var_95_daily_rate']:.0f} (cap {var_cap:.0f})  "
          f"|  expanded {hard_exp_m['no']['var_95_daily_rate']:.0f} (cap {var_cap:.0f})  "
          f"|  AI {grok_m['no']['var_95_daily_rate']:.0f} (cap {var_cap:.0f})")
    print(f"FX +10% reduction:  "
          f"minimal {fx_red(hard_min_m['fx10']):.1f}%  "
          f"|  expanded {fx_red(hard_exp_m['fx10']):.1f}%  "
          f"|  AI {fx_red(grok_m['fx10']):.1f}%")

    # 8) Audit artifact
    artifact = {
        "constraints": DEFAULT_CONSTRAINTS.__dict__,
        "baseline": base_no,
        "traditional_minimal": {
            "selection": (hard_min_sel.to_dict(orient='records') if not hard_min_sel.empty else []),
            "scenario_table": hard_min_scen.to_dict(orient='records')
        },
        "traditional_expanded": {
            "selection": (hard_exp_sel.to_dict(orient='records') if not hard_exp_sel.empty else []),
            "scenario_table": hard_exp_scen.to_dict(orient='records')
        },
        "agent": {
            "selection": (grok_sel.to_dict(orient='records') if not grok_sel.empty else []),
            "scenario_table": grok_scen.to_dict(orient='records')
        }
    }
    _write_run_artifact("run_summary.json", artifact)

    print("\nWorkflow completed.")


if __name__ == "__main__":
    main()
