"""
Example usage of AgentSeer Individual Agent API Endpoints

This script demonstrates how to call each agent endpoint individually
and then use the Governor agent to synthesize the results.
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


def call_agent(endpoint: str, ticker: str) -> Dict[str, Any]:
    """Call an individual agent endpoint."""
    url = f"{BASE_URL}/api/agents/{endpoint}"
    payload = {"ticker": ticker}

    print(f"🔄 Calling {endpoint} agent for ${ticker}...")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def call_governor(ticker: str, agent_summaries: Dict[str, str]) -> Dict[str, Any]:
    """Call the Governor agent with summaries from other agents."""
    url = f"{BASE_URL}/api/agents/governor"
    payload = {
        "ticker": ticker,
        "sec_summary": agent_summaries.get("sec"),
        "news_summary": agent_summaries.get("news"),
        "social_summary": agent_summaries.get("social"),
        "chart_summary": agent_summaries.get("chart"),
        "analyst_summary": agent_summaries.get("analyst"),
    }

    print(f"🎯 Calling governor agent for ${ticker}...")
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def analyze_stock_step_by_step(ticker: str):
    """
    Analyze a stock by calling each agent individually,
    then synthesize with the Governor agent.
    """
    print(f"\n{'='*70}")
    print(f"📊 Step-by-Step Analysis for ${ticker}")
    print(f"{'='*70}\n")

    # Step 1: Call individual agents
    agent_summaries = {}
    agents = ["sec", "news", "social", "chart", "analyst"]

    for agent in agents:
        try:
            result = call_agent(agent, ticker)
            agent_summaries[agent] = result.get("summary", "Not available")
            print(f"✅ {agent.upper()} agent completed")
            print(f"   Status: {result['status']}")
            print(f"   Timestamp: {result['timestamp']}")
            print()
        except Exception as e:
            print(f"❌ {agent.upper()} agent failed: {e}\n")
            agent_summaries[agent] = f"Error: {str(e)}"

    # Step 2: Call Governor agent with all summaries
    try:
        gov_result = call_governor(ticker, agent_summaries)
        print(f"✅ Governor agent completed")
        print(f"   Status: {gov_result['status']}")
        print(f"   Timestamp: {gov_result['timestamp']}")
        print()

        # Display Governor's summary
        print(f"\n{'='*70}")
        print(f"🎯 GOVERNOR'S FINAL SUMMARY")
        print(f"{'='*70}\n")
        print(gov_result.get("summary", "No summary available"))
        print()

        return gov_result

    except Exception as e:
        print(f"❌ Governor agent failed: {e}\n")
        return None


def quick_single_agent_test(ticker: str, agent: str):
    """Quick test of a single agent endpoint."""
    print(f"\n{'='*70}")
    print(f"🧪 Quick Test: {agent.upper()} Agent for ${ticker}")
    print(f"{'='*70}\n")

    try:
        result = call_agent(agent, ticker)
        print(f"✅ Success!")
        print(f"\nResponse:")
        print(json.dumps(result, indent=2)[:500] + "...")

    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "AAPL"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python example_api_usage.py <TICKER>\n")

    # Option 1: Full step-by-step analysis
    analyze_stock_step_by_step(ticker)

    # Option 2: Quick single agent test (uncomment to use)
    # quick_single_agent_test(ticker, "news")

    print(f"\n{'='*70}")
    print(f"📚 For interactive API docs, visit: {BASE_URL}/docs")
    print(f"{'='*70}\n")
