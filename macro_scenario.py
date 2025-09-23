# macro_scenario.py
# --------------------------------------------------------------------------------------
# PURPOSE :
# --------------------------------------------------------------------------------------
# What is this:
#   This module extends the treasury hedging framework by incorporating
#   **macro-economic scenarios**. It fetches live US indicators (GDP, inflation,
#   unemployment) from Financial Modeling Prep (FMP) and, using LangChain + an
#   OpenAI LLM, converts them into a **quantitative stress vector** plus a
#   narrative explanation.
#
# Why:
#   Traditional stress scenarios (parallel shifts, FX ±10%, volatility shocks)
#   are static and predefined. By using LangChain, we can:
#     - Pull in real-world economics (e.g., GDP slowdown, inflation surprise).
#     - Translate them into market-relevant shocks (+bp yield shift, FX % move).
#     - Apply them through the same simulation harness as legacy stresses.
#   This ensures hedge robustness is tested not just against abstract shocks,
#   but also against **macro narratives auditors and ALCO members care about**.
#
# Data used:
#   - FMP Economic Indicators API (GDP, inflation, unemployment).
#   - LangChain LLM (ChatOpenAI) for stress vector + narrative.
# --------------------------------------------------------------------------------------

import requests
import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# --------------------------------------------------------------------------------------
# Load API keys and environment variables
# --------------------------------------------------------------------------------------
# What:
#   Loads credentials (FMP API key, OpenAI API key via LangChain).
# Why:
#   Keeps secrets out of the code; supports portability across dev/prod.
# Data used:
#   - .env file with FMP_API_KEY and OpenAI credentials.
# --------------------------------------------------------------------------------------
load_dotenv()   

FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/stable/economic-indicators"


def fetch_macro_indicators():
    """
    What is this:
        Fetches selected macro indicators (GDP, inflation, unemployment)
        from Financial Modeling Prep’s stable economic endpoint.

    What it does:
        Queries each indicator by name, parses JSON, and builds a dictionary
        {indicator: value} for later scenario generation.

    Why:
        Provides an objective economic state vector. Instead of hardcoding
        assumptions, we ground scenarios in current macro data.

    Data used:
        Input:
          - FMP stable endpoint (https://financialmodelingprep.com/stable/economic-indicators).
        Output:
          - Dictionary of values, e.g. {"GDP": 30353.9, "inflationRate": 2.37, "unemploymentRate": 4.3}.
    """
    wanted = ["GDP", "inflationRate", "unemploymentRate"]
    indicators = {}
    for name in wanted:
        url = f"{BASE_URL}?name={name}&apikey={FMP_API_KEY}"
        resp = requests.get(url).json()
        if resp and isinstance(resp, list):
            indicators[name] = resp[0].get("value")
    return indicators


def generate_macro_scenario(indicators, llm=None):
    """
    What is this:
        Uses a LangChain LLM to map raw macro indicators into a stress vector
        (basis point shift, FX % move) plus a governance-friendly narrative.

    What it does:
        1. Prepares a structured prompt with GDP, inflation, unemployment.
        2. LLM outputs a JSON object with:
           - rates_shift_bp: shift in basis points for rates.
           - usd_fx_move_pct: % move in USD vs majors.
           - narrative: a one-paragraph human explanation.
        3. Parses JSON safely and returns it.

    Why:
        Moves scenario generation from *manual assumption* to *AI-assisted,
        economics-aware*. This ensures treasury stress tests align with
        plausible macro narratives and remain explainable to committees.

    Data used:
        Input:
          - indicators dict from fetch_macro_indicators()
        Output:
          - scenario dict: {"rates_shift_bp": 50, "usd_fx_move_pct": -3, "narrative": "..."}
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define prompt template
    prompt = PromptTemplate(
        input_variables=["gdp", "inflation", "unemployment"],
        template="""
        You are a treasury risk analyst. Based on the following macro indicators:
        - US GDP: {gdp} (billions USD)
        - Inflation rate: {inflation}%
        - Unemployment rate: {unemployment}%

        Propose a stress scenario for treasury hedging. Respond strictly in JSON with:
        {{
          "rates_shift_bp": (positive = yields up, negative = yields down),
          "usd_fx_move_pct": (positive = USD strengthens, negative = USD weakens),
          "narrative": "one-paragraph explanation"
        }}
        """
    )

    # New LangChain pipeline style: prompt | llm
    chain = prompt | llm
    raw = chain.invoke({
        "gdp": indicators.get("GDP"),
        "inflation": indicators.get("inflationRate"),
        "unemployment": indicators.get("unemploymentRate")
    }).content

    # Parse JSON safely (strip any extra text)
    scenario = json.loads(raw[raw.find("{"): raw.rfind("}")+1])
    return scenario


if __name__ == "__main__":
    # Governance demo mode:
    #   Running this file standalone fetches macro indicators,
    #   generates a scenario, and prints both.
    indicators = fetch_macro_indicators()
    print("Fetched indicators:", indicators)

    if indicators:
        scenario = generate_macro_scenario(indicators)
        print("Scenario:", scenario)
    else:
        print("⚠️ No indicators fetched – check API key or endpoint.")
