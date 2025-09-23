# orchestrator.py
# --------------------------------------------------------------------------------------
# PURPOSE :
# --------------------------------------------------------------------------------------
# What is this:
#   The "orchestrator" runs an end-to-end treasury-hedging evaluation. It executes three
#   comparable hedge paths under the same governance guardrails:
#     (A) Traditional-Minimal  : legacy 3-instrument static catalog
#     (B) Traditional-Expanded : broader static catalog mirroring AI families
#     (C) AI (Agent)           : AI-proposed dynamic catalog tailored to portfolio + market
#
# Extension:
#   A proof-of-concept **Macro Scenario** is added using LangChain + FMP data. Live
#   macro indicators (GDP, inflation, unemployment) are fetched, transformed into a
#   stress vector (+bp shift, FX % move), and applied to each hedged portfolio to see
#   how AI vs static hedges perform under a macro-informed stress.
#
# Why:
#   Demonstrate whether an AI-curated hedge menu delivers superior risk reduction
#   compared to fixed catalogs — not only under standard stresses, but also under
#   live macro-economic shocks.
#
# Data used:
#   - treasury_portfolio.csv       : balance sheet positions
#   - prices.csv                   : static rate references
#   - hedge_universe_min.csv       : minimal static shelf
#   - hedge_universe_expanded.csv  : expanded static shelf
#   - market history (^TNX, EUR/USD): for vol & correlation calibration
#   - macro_scenario.py            : fetches macro data, generates stress vector
#   - run_summary.json             : output artifact (audit + reporting)
# --------------------------------------------------------------------------------------

import json
import random
import numpy as np

# External modules (optimization, agent loop, risk engines)
import agent
import financial_library as fl
from financial_library import (
    load_portfolio, load_static_rates, create_hedge_universe,
    get_historical_data, calibrate_vol_and_corr,
    compute_baseline_risks, optimize_hedge_basket,
    simulate_scenarios, postprocess_hedge_selection,
    _build_hedged_df, DEFAULT_CONSTRAINTS, align_selection
)

# NEW: macro-scenario PoC
from macro_scenario import fetch_macro_indicators, generate_macro_scenario


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _seed_everything(seed: int = 42):
    """
    What is this:
        Deterministic seeding helper.

    What it does:
        Sets random number generators to a fixed seed so runs are repeatable.

    Why:
        Reproducibility is critical for governance, ALCO approval, and audit.
        Without a fixed seed, two runs with identical inputs might yield slightly
        different hedge baskets. Fixing the seed guarantees deterministic outputs.

    Why 42:
        The number itself is arbitrary; "42" is a convention in data science.
    """
    random.seed(seed)
    np.random.seed(seed)


def _write_run_artifact(path, payload: dict):
    """Audit artifact writer (saves results as JSON)."""
    try:
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        print(f"Artifact write failed: {e}")


def _evaluate_universe_for_closed_loop(portfolio_df, universe_df, usvol, corr):
    """
    Closed-loop evaluator for AI hedge universes.

    What:
        Optimizes, post-processes, measures risk metrics, and checks guardrails.
    Why:
        Provides feedback to the AI agent so it iteratively improves proposals.
    """
    opt = optimize_hedge_basket(portfolio_df, universe_df, usvol, corr)
    if opt.empty:
        return {'duration_gap_ok': False, 'var_ok': False, 'fx_ok': False,
                'overshoot_neg_gap': False, 'score': -999,
                'feedback': "Empty optimal selection."}

    opt_pp = postprocess_hedge_selection(
        portfolio_df, universe_df, opt, usvol, corr,
        gap_abs_cap=0.80, gap_neg_floor=-0.20,
        var_cap_pct=0.10, fx_improve_target=0.35,
        max_iter=10, verbose=False
    )
    xvec = align_selection(universe_df, opt_pp)
    hedged_df = _build_hedged_df(portfolio_df, universe_df, xvec)

    # Baseline vs hedged metrics under different shocks
    base_no  = compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=0)
    hed_no   = compute_baseline_risks(hedged_df,  usvol, corr, rate_shift_bp=0)
    base_fx  = compute_baseline_risks(portfolio_df, usvol, corr, fx_shift_pct=10)
    hed_fx   = compute_baseline_risks(hedged_df,  usvol, corr, fx_shift_pct=10)
    base_p100= compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=100)
    hed_p100 = compute_baseline_risks(hedged_df,  usvol, corr, rate_shift_bp=100)

    # Guardrails:
    # - Duration gap must be within -0.20y to +0.80y
    # - Post-hedge VaR ≤ 110% of baseline
    # - FX impact must drop ≥ 35% under +10% shock
    # - ΔEVE improvement target ≈ 45% under +100bp
    var_cap = base_no['var_95_daily_rate'] * 1.10
    fx_target_abs = abs(base_fx['fx_impact']) * (1.0 - 0.35)
    gap_ok = (-0.20 <= hed_no['duration_gap'] <= 0.80)
    var_ok = (hed_no['var_95_daily_rate'] <= var_cap + 1.0)
    fx_ok  = (abs(hed_fx['fx_impact']) <= fx_target_abs + 1.0) if abs(base_fx['fx_impact']) > 1e-6 else True
    deve_improve = max(0.0, 1.0 - (abs(hed_p100['delta_eve']) / max(1e-6, abs(base_p100['delta_eve'])))) * 100.0

    feedback = []
    if not gap_ok: feedback.append(f"- Gap out of band ({hed_no['duration_gap']:.2f}y).")
    if not var_ok: feedback.append(f"- VaR {hed_no['var_95_daily_rate']:.0f} > cap {var_cap:.0f}.")
    if not fx_ok:  feedback.append("- FX reduction < 35%.")
    if deve_improve < 45.0: feedback.append(f"- ΔEVE improvement {deve_improve:.0f}% < 45% target.")

    score = (deve_improve * 2.0
             - max(0.0, abs(hed_no['duration_gap']) - 0.80) * 100.0
             - max(0.0, (hed_no['var_95_daily_rate'] / max(1e-6, base_no['var_95_daily_rate']) - 1.10)) * 100.0
             + (35.0 - max(0.0, (abs(hed_fx['fx_impact']) / max(1e-6, abs(base_fx['fx_impact'])) * 100.0 - 65.0))))

    return {'duration_gap_ok': gap_ok, 'var_ok': var_ok, 'fx_ok': fx_ok,
            'overshoot_neg_gap': (hed_no['duration_gap'] < -0.20),
            'score': score, 'feedback': "\n".join(feedback) or "OK"}


def _run_traditional(label, portfolio_df, universe_df, usvol, corr):
    """Helper to run static shelves (Minimal, Expanded)."""
    print(f"\n--- {label} (Traditional) ---")
    opt = optimize_hedge_basket(portfolio_df, universe_df, usvol, corr)
    sel = postprocess_hedge_selection(
        portfolio_df, universe_df, opt, usvol, corr,
        gap_abs_cap=0.80, gap_neg_floor=-0.20,
        var_cap_pct=0.10, fx_improve_target=0.35,
        max_iter=12, verbose=True
    )
    xvec = align_selection(universe_df, sel)
    hedged = _build_hedged_df(portfolio_df, universe_df, xvec)
    scen = simulate_scenarios(portfolio_df, hedged, usvol, corr)
    return sel, hedged, scen


def run_macro_scenario(portfolio_df, hedged_df, usvol, corr, macro_scenario):
    """
    Apply a LangChain-generated macro scenario to baseline & hedged portfolios.

    Input:
        macro_scenario: dict with {"rates_shift_bp": X, "usd_fx_move_pct": Y}
    Output:
        base_metrics, hedged_metrics (dicts with ΔEVE, VaR, FX impact, gap)
    """
    rates_shift = macro_scenario.get("rates_shift_bp", 0)
    fx_shift    = macro_scenario.get("usd_fx_move_pct", 0)
    base_metrics   = compute_baseline_risks(portfolio_df, usvol, corr,
                                            rate_shift_bp=rates_shift,
                                            fx_shift_pct=fx_shift)
    hedged_metrics = compute_baseline_risks(hedged_df, usvol, corr,
                                            rate_shift_bp=rates_shift,
                                            fx_shift_pct=fx_shift)
    return base_metrics, hedged_metrics


# --------------------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------------------
def main():
    print("Starting treasury risk computation, hedging proposal, and scenario testing workflow...")
    _seed_everything(42)

    # 1) Load data
    portfolio_df = load_portfolio('treasury_portfolio.csv')
    _ = load_static_rates('prices.csv')

    # 2) Hedge universes (Minimal, Expanded)
    try: hard_uni_min = create_hedge_universe('hedge_universe_min.csv', mode='minimal')
    except TypeError: hard_uni_min = create_hedge_universe('hedge_universe_min.csv')
    try: hard_uni_exp = create_hedge_universe('hedge_universe_expanded.csv', mode='expanded')
    except TypeError: hard_uni_exp = hard_uni_min

    # 3) Market calibration
    hist = get_historical_data(tickers=['^TNX', 'EURUSD=X'])
    usvol, corr = calibrate_vol_and_corr(hist, portfolio_df)

    # 4) Baseline risk
    base_no  = compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=0)
    base_p100= compute_baseline_risks(portfolio_df, usvol, corr, rate_shift_bp=100)

    # 5) Traditional shelves
    hard_min_sel, hard_min_hedged, hard_min_scen = _run_traditional("Traditional-Minimal", portfolio_df, hard_uni_min, usvol, corr)
    hard_exp_sel, hard_exp_hedged, hard_exp_scen = _run_traditional("Traditional-Expanded", portfolio_df, hard_uni_exp, usvol, corr)

    # 6) AI path
    fl._UNIVERSE_EVALUATOR = lambda df: _evaluate_universe_for_closed_loop(portfolio_df, df, usvol, corr)
    grok_sel, grok_universe, dbg = agent.run_agentic_universe(
        portfolio_df, usvol, corr,
        var_cap_pct=0.10, fx_improve_target=0.35,
        gap_abs_cap=0.80, gap_neg_floor=-0.20, max_iters=3
    )
    grok_xvec = align_selection(grok_universe, grok_sel)
    grok_hedged = _build_hedged_df(portfolio_df, grok_universe, grok_xvec)
    grok_scen   = simulate_scenarios(portfolio_df, grok_hedged, usvol, corr)

    # 7) Macro-driven stress (PoC)
    try:
        indicators = fetch_macro_indicators()
        if indicators:
            macro_scenario = generate_macro_scenario(indicators)
            print("\n--- Macro-Driven Stress Scenario (LangChain) ---")
            print("Indicators:", indicators)
            print("Shock Vector:", macro_scenario)

            # Run macro stress on each hedge path
            base_min, hed_min = run_macro_scenario(portfolio_df, hard_min_hedged, usvol, corr, macro_scenario)
            base_exp, hed_exp = run_macro_scenario(portfolio_df, hard_exp_hedged, usvol, corr, macro_scenario)
            base_ai,  hed_ai  = run_macro_scenario(portfolio_df, grok_hedged,   usvol, corr, macro_scenario)

            print("\nMacro Scenario Results (ΔEVE, VaR, FX Impact):")
            print("Minimal :", hed_min)
            print("Expanded:", hed_exp)
            print("AI      :", hed_ai)
        else:
            print("⚠️ No macro indicators fetched.")
            macro_scenario, hed_min, hed_exp, hed_ai = {}, {}, {}, {}
    except Exception as e:
        print("⚠️ Macro scenario failed:", e)
        macro_scenario, hed_min, hed_exp, hed_ai = {}, {}, {}, {}

    # 8) Write audit artifact
    artifact = {
        "constraints": DEFAULT_CONSTRAINTS.__dict__,
        "baseline": base_no,
        "traditional_minimal": {"selection": hard_min_sel.to_dict(orient='records'),
                                "scenario_table": hard_min_scen.to_dict(orient='records')},
        "traditional_expanded": {"selection": hard_exp_sel.to_dict(orient='records'),
                                 "scenario_table": hard_exp_scen.to_dict(orient='records')},
        "agent": {"selection": grok_sel.to_dict(orient='records'),
                  "scenario_table": grok_scen.to_dict(orient='records')},
        "macro_scenario": {"indicators": indicators,
                           "shock_vector": macro_scenario,
                           "results": {"minimal": hed_min,
                                       "expanded": hed_exp,
                                       "ai": hed_ai}}
    }
    _write_run_artifact("run_summary.json", artifact)

    print("\nWorkflow completed.")


if __name__ == "__main__":
    main()
