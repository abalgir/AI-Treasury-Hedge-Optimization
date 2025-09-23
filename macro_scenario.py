# file: macro_scenario.py
import requests
import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()   

FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/stable/economic-indicators"

def fetch_macro_indicators():
    """Fetches selected macro indicators from FMP stable endpoint."""
    wanted = ["GDP", "inflationRate", "unemploymentRate"]
    indicators = {}
    for name in wanted:
        url = f"{BASE_URL}?name={name}&apikey={FMP_API_KEY}"
        resp = requests.get(url).json()
        if resp and isinstance(resp, list):
            indicators[name] = resp[0].get("value")
    return indicators

def generate_macro_scenario(indicators, llm=None):
    """Uses LangChain LLM to map macro indicators into a stress scenario."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

    # new LangChain pipeline style
    chain = prompt | llm
    raw = chain.invoke({
        "gdp": indicators.get("GDP"),
        "inflation": indicators.get("inflationRate"),
        "unemployment": indicators.get("unemploymentRate")
    }).content

    scenario = json.loads(raw[raw.find("{"): raw.rfind("}")+1])
    return scenario

if __name__ == "__main__":
    indicators = fetch_macro_indicators()
    print("Fetched indicators:", indicators)

    if indicators:
        scenario = generate_macro_scenario(indicators)
        print("Scenario:", scenario)
    else:
        print("⚠️ No indicators fetched – check API key or endpoint.")
