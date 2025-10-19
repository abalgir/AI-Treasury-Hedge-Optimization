KPIMinds LLC

# AI-Driven Treasury Hedging: Dynamic Universe Selection for ALM

This repository demonstrates an AI-driven approach to treasury hedging, outperforming traditional static hedge baskets for Asset-Liability Management (ALM) and Interest Rate Risk in the Banking Book (IRRBB) compliance. Designed for bank treasurers and consultants, it compares three hedging strategies—Minimal (3 instruments), Expanded (8 instruments), and AI-generated (dynamic)—on a simulated portfolio, using identical optimization and risk guardrails. The AI approach, powered by Grok (xAI) with OpenAI, dynamically proposes hedge universes, achieving superior duration gap tightening (2.22y vs. 2.34y), ΔEVE improvement (10.9% vs. 2.0%), and FX risk reduction (35% vs. 11%) while respecting VaR caps. This aligns with Bank of Thailand (BoT) guidelines and supports treasury sales for Southeast Asian banks facing FX volatility (e.g., PromptPay flows).
The white paper Revolutionizing_Treasury_Hedging.pdf provides detailed methodology and insights for implementing AI-driven hedging.

Code and results are open for public use, with setup instructions for reproducibility. Ideal for banks and consultants optimizing treasury operations.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Portfolio and Hedge Universe Format](#portfolio-and-hedge-universe-format)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
This project compares three hedging strategies for a complex portfolio (321M assets, 2.37y duration gap, 3.6M FX exposure):
- **Traditional–Minimal**: Static 3-instrument universe (IRS, UST future, FX option).
- **Traditional–Expanded**: Static 8-instrument universe (IRS, OIS, futures, FX options/forward, cross-currency swap), pre-defined but broader.
- **AI (Agent)**: Dynamic universe proposed by Grok, refined via closed-loop feedback, optimized with SciPy, and guardrail-checked.

All paths use the same:
- **Optimizer**: SciPy SLSQP, minimizing 0.6*|gap| + 0.3*|ΔEVE| + 0.1*VaR.
- **Guardrails**: VaR ≤ 1.10× baseline, FX +10% reduction ≥ 35%, gap in [-0.20y, 0.80y].
- **Scenarios**: Rate shifts (±100/200bps), vol shock (+20%), FX shock (+10%).
- **Macro Scenarios (LangChain)**: In addition to predefined stresses (parallel rate shifts, vol shocks, FX +10%), the framework integrates **LangChain** to fetch live macro indicators (GDP, inflation, unemployment) and automatically translate them into stress vectors (+bp shifts, FX % moves). These are run through the same scenario harness, allowing treasurers to validate hedges against *real-world economic narratives*, not just abstract shocks.


The AI path outperforms, automating universe curation to adapt to market volatility (e.g., US10Y at 5.53bp, corr 0.68) and portfolio complexity (e.g., MBS, CLOs, FX swaps), reducing costs and cycle times for treasury sales (e.g., FX hedges for Thai exporters).

## Key Results
From a controlled run (fixed seed 42, 2024-09-22 to 2025-09-22 data):
- **Duration Gap (Hedged)**: AI 2.218y vs. Expanded 2.260y vs. Minimal 2.344y (AI best, 5% tighter).
- **ΔEVE (+100bps Improvement)**: AI 10.9% vs. Expanded 8.0% vs. Minimal 2.0% (AI 5.5x Minimal).
- **VaR (95%, Daily)**: AI 671,560 vs. Expanded 658,088 vs. Minimal 630,341 (all under cap 683,113).
- **FX (+10% Reduction)**: AI 35% (ties Expanded) vs. Minimal 11% (AI/Expanded meet target).
- **Composite Score** (40% ΔEVE + 30% FX - 20% VaR increase - 10% |gap| deviation): AI +42 vs. Expanded +35 vs. Minimal +10.

**Why AI Wins**: Dynamic universe curation (7-9 instruments, tailored tenors/FX headroom) + agent repairs (e.g., FX top-up to 45.4M, rate scaling α=0.2) adapt to high gap/FX, unlike static shelves. This reduces trading costs, carry drag, and ALCO cycles, aligning with BoT’s digital finance push.

## Repository Structure
- orchestrator.py        # Main script: runs Minimal, Expanded, AI paths
- macro_scenario.py      # LangChain integration: fetches GDP, inflation, unemployment from FMP and generates stress vectors + narrative
- agent.py               # AI loop: proposes, optimizes, repairs universe
- financial_library.py   # Core functions: I/O, risks, optimization, Grok calls
- treasury_portfolio.csv # Sample portfolio (assets, liabilities, FX swaps)
- hedge_universe.csv     # Hard-coded + AI-appended universes
- run_summary.json       # Output artifact: selections, scenarios, KPIs
- README.md              # This file
- LICENSE                # MIT LICENSE

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/treasury-ai-hedging.git
   cd treasury-ai-hedging

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install pandas numpy scipy yfinance requests python-dotenv

XAI_API_KEY=your_xai_key  # Required for Grok

Usage
Run the main script to compare hedging strategies:
python orchestrator.py

Portfolio and Hedge Universe Format

treasury_portfolio.csv:

Columns: instrument_type, notional_amount, maturity_date, interest_rate, duration, currency, volatility, correlation_to_us10y, fx_delta (optional).
Example: Bond,100000000,2028-09-21,3.5,2.5,USD,8.0,0.9,
Notes: Positive notional for assets, negative for liabilities; includes FX swaps (USD/EUR).

hedge_universe.csv:

Columns: instrument_type, max_notional, maturity_date, interest_rate, duration, currency, volatility, correlation_to_us10y, strike, fx_delta.
Example: Interest Rate Swap (Pay Fixed),20000000,2028-09-21,4.0,2.8,USD,10.0,0.9,,
Notes: AI appends to this; hard-coded overwrites with 3 instruments unless mode='expanded'.

Assumptions and Limitations

Assumptions:

Parametric VaR (1.65 * vol * DV01) approximates legacy systems models.
FX sensitivity fixed at 0.5 (options/forwards); real models use Black-Scholes.
No basis risk or multi-currency correlations modeled.

Limitations:

Prototype-level simplifications; production needs basis risk, behavioral NMDs.
API costs (Grok/OpenAI/Anthropic) apply for frequent runs.
It has been tested on one portfolio; use with care.


Contributing
Contributions are welcome! Please:

Fork and submit pull requests with clear, descriptive titles.
Test changes with the provided CSVs.

License
This project is licensed under the MIT License — free to use, modify, and distribute, with no warranty.
Contact

GitHub: abalgir
Email: admin@kpiminds.com

