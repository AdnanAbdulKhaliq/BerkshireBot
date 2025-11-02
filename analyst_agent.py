"""
Analyst Agent - Analyst Swarm Component (Enhanced)

This agent aggregates analyst ratings from multiple free sources:
1. Yahoo Finance (via yfinance) - Recommendations, price targets, analyst opinions
2. Financial Modeling Prep API (free tier) - Analyst estimates and upgrades
3. Web scraping fallback for recent news on analyst actions

Environment Variables Required:
    - GEMINI_API_KEY: Your Google Gemini API key
    - FMP_API_KEY: Financial Modeling Prep API key (optional, free tier available)
"""

import os
import json
import time
from datetime import datetime, timedelta
from functools import lru_cache
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# Financial data libraries
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"⚠️ Missing required package: {e}")
    print("Run: pip install yfinance pandas numpy")
    yf = None
    pd = None
    np = None

try:
    import requests
except ImportError:
    print("⚠️ requests not installed. Run: pip install requests")
    requests = None

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- ENVIRONMENT VALIDATION ---


def validate_environment() -> None:
    """Validate all required API keys are set."""
    required_vars = {"GEMINI_API_KEY": "Gemini LLM for analysis"}

    missing = []
    for var_name, description in required_vars.items():
        if var_name not in os.environ:
            missing.append(f"  ❌ {var_name}: {description}")

    if missing:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing)
        raise EnvironmentError(error_msg)

    # Optional API key
    if "FMP_API_KEY" not in os.environ:
        print("ℹ️ FMP_API_KEY not set - some features limited")

    print("✅ Analyst_Agent: Environment variables validated")


# Validate at module load time
validate_environment()

# --- SETUP LLM ---

llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest", temperature=0.1, api_key=os.environ["GEMINI_API_KEY"]
)

# --- DATA FETCHING FROM MULTIPLE SOURCES ---


def convert_to_serializable(obj):
    """Convert pandas/numpy objects to JSON-serializable types."""
    import pandas as pd
    import numpy as np

    if isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
        return obj.isoformat() if hasattr(obj, "isoformat") else str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    elif pd.isna(obj):
        return None
    return obj


def fetch_yahoo_finance_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch analyst data from Yahoo Finance using yfinance.
    This is completely free and provides rich analyst data.
    """
    if yf is None:
        return {"error": "yfinance not installed"}

    try:
        stock = yf.Ticker(ticker)

        # Get recommendations (buy/hold/sell ratings over time)
        recent_recs = []
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
                # Get most recent 10 recommendations and convert to serializable format
                recent_df = recommendations.tail(10).reset_index()
                for _, row in recent_df.iterrows():
                    rec = {}
                    for col in recent_df.columns:
                        val = row[col]
                        rec[col] = convert_to_serializable(val)
                    recent_recs.append(rec)
        except Exception as e:
            print(f"⚠️ Could not fetch recommendations: {e}")

        # Get analyst price targets
        analyst_info = {}
        try:
            info = stock.info
            analyst_info = {
                "targetHighPrice": convert_to_serializable(info.get("targetHighPrice")),
                "targetLowPrice": convert_to_serializable(info.get("targetLowPrice")),
                "targetMeanPrice": convert_to_serializable(info.get("targetMeanPrice")),
                "targetMedianPrice": convert_to_serializable(
                    info.get("targetMedianPrice")
                ),
                "recommendationMean": convert_to_serializable(
                    info.get("recommendationMean")
                ),
                "recommendationKey": info.get("recommendationKey"),
                "numberOfAnalystOpinions": convert_to_serializable(
                    info.get("numberOfAnalystOpinions")
                ),
                "currentPrice": convert_to_serializable(info.get("currentPrice")),
                "previousClose": convert_to_serializable(info.get("previousClose")),
            }
        except Exception as e:
            print(f"⚠️ Could not fetch analyst info: {e}")

        # Get upgrades/downgrades
        upgrades_list = []
        try:
            upgrades = stock.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                # Get last 30 days
                thirty_days_ago = datetime.now() - timedelta(days=30)
                recent_upgrades = upgrades[upgrades.index >= thirty_days_ago]

                # Convert to list of dicts with serializable date strings
                # Apply stricter recency filters
                for date_idx, row in recent_upgrades.iterrows():
                    upgrade_date = (
                        date_idx
                        if hasattr(date_idx, "date")
                        else datetime.strptime(str(date_idx)[:10], "%Y-%m-%d")
                    )
                    days_ago = (datetime.now() - upgrade_date).days

                    # Only include if within last 30 days, but prioritize last 7 days
                    if days_ago <= 30:
                        upgrade_entry = {
                            "date": (
                                date_idx.strftime("%Y-%m-%d")
                                if hasattr(date_idx, "strftime")
                                else str(date_idx)
                            ),
                            "days_ago": days_ago,  # Track recency
                            "firm": convert_to_serializable(row.get("Firm", "Unknown")),
                            "toGrade": convert_to_serializable(
                                row.get("ToGrade", "N/A")
                            ),
                            "fromGrade": convert_to_serializable(
                                row.get("FromGrade", "N/A")
                            ),
                            "action": convert_to_serializable(row.get("Action", "N/A")),
                            "recency_weight": (
                                1.0 if days_ago <= 7 else 0.5
                            ),  # Higher weight for last 7 days
                        }
                        upgrades_list.append(upgrade_entry)
        except Exception as e:
            print(f"⚠️ Could not fetch upgrades/downgrades: {e}")

        return {
            "source": "Yahoo Finance",
            "recent_recommendations": recent_recs,
            "analyst_info": analyst_info,
            "upgrades_downgrades": upgrades_list,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
        return {"error": f"Yahoo Finance error: {str(e)}"}


def fetch_fmp_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch analyst data from Financial Modeling Prep API (free tier).
    Provides analyst estimates, upgrades/downgrades, and price targets.
    """
    if requests is None or "FMP_API_KEY" not in os.environ:
        return {"error": "FMP API not available"}

    api_key = os.environ.get("FMP_API_KEY")
    base_url = "https://financialmodelingprep.com/api/v3"

    try:
        results = {}

        # 1. Get analyst estimates
        try:
            url = f"{base_url}/analyst-estimates/{ticker}?apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                results["analyst_estimates"] = response.json()
        except:
            pass

        # 2. Get upgrades/downgrades
        try:
            url = f"{base_url}/upgrades-downgrades?symbol={ticker}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Get only recent ones (last 30 days)
                thirty_days_ago = datetime.now() - timedelta(days=30)
                recent = [
                    item
                    for item in data
                    if datetime.strptime(
                        item.get("publishedDate", "2000-01-01"), "%Y-%m-%d %H:%M:%S"
                    )
                    >= thirty_days_ago
                ]
                results["upgrades_downgrades"] = recent[:10]  # Limit to 10 most recent
        except:
            pass

        # 3. Get price target consensus
        try:
            url = f"{base_url}/price-target-consensus?symbol={ticker}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                results["price_target_consensus"] = response.json()
        except:
            pass

        results["source"] = "Financial Modeling Prep"
        return results

    except Exception as e:
        return {"error": f"FMP API error: {str(e)}"}


def fetch_marketwatch_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch analyst data from MarketWatch (web scraping fallback).
    Provides additional analyst ratings and estimates.
    """
    if requests is None:
        return {"error": "requests not available"}

    try:
        # MarketWatch analyst ratings page
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}/analystestimates"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            # Basic parsing - in production, you'd use BeautifulSoup
            # For now, just mark as available
            return {"source": "MarketWatch", "status": "available", "url": url}
        else:
            return {"error": f"MarketWatch returned status {response.status_code}"}

    except Exception as e:
        return {"error": f"MarketWatch error: {str(e)}"}


@lru_cache(maxsize=100)
def cached_analyst_fetch(ticker: str, timestamp_hour: int) -> Dict[str, Any]:
    """
    Cached fetch from all sources. Cache expires every hour.
    """
    print(f"🔍 Analyst_Agent: Fetching analyst data for {ticker}...")

    # Fetch from Yahoo Finance (primary source - free and reliable)
    yahoo_data = fetch_yahoo_finance_data(ticker)

    # Fetch from FMP (secondary source - optional)
    fmp_data = fetch_fmp_data(ticker)

    # Fetch from MarketWatch (tertiary source - optional)
    mw_data = fetch_marketwatch_data(ticker)

    return {
        "ticker": ticker,
        "yahoo_finance": yahoo_data,
        "fmp": fmp_data,
        "marketwatch": mw_data,
        "fetch_timestamp": datetime.now().isoformat(),
    }


def safe_json_dumps(obj):
    """Safely convert objects to JSON string with custom serialization."""

    def json_serializer(o):
        if pd and isinstance(o, (pd.Timestamp, pd.DatetimeTZDtype)):
            return o.isoformat() if hasattr(o, "isoformat") else str(o)
        elif np and isinstance(o, (np.integer, np.floating)):
            return float(o)
        elif np and isinstance(o, np.ndarray):
            return o.tolist()
        elif pd and isinstance(o, pd.Series):
            return o.to_dict()
        elif pd and isinstance(o, pd.DataFrame):
            return o.to_dict("records")
        elif pd and pd.isna(o):
            return None
        return str(o)

    return json.dumps(obj, default=json_serializer, indent=2)


def run_analyst_search(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute analyst data search with caching and error handling.
    """
    ticker = input_dict.get("ticker", "").upper()

    if not ticker:
        return {"raw_data_json": "Error: No ticker provided", "ticker": ""}

    try:
        # Cache key changes every hour
        cache_key = int(time.time() // 3600)
        results = cached_analyst_fetch(ticker, cache_key)

        print(f"✅ Analyst_Agent: Retrieved analyst data for {ticker}")

        return {"raw_data_json": safe_json_dumps(results), "ticker": ticker}

    except Exception as e:
        print(f"❌ Analyst_Agent: Search failed - {e}")
        import traceback

        traceback.print_exc()
        return {
            "raw_data_json": f"Error retrieving analyst data: {str(e)}",
            "ticker": ticker,
        }


# --- ENHANCED PROMPT TEMPLATE ---

prompt_template = ChatPromptTemplate.from_template(
    """You are an expert "Analyst Agent" for a financial firm. Your task is to 
analyze analyst ratings and price targets for stock ${ticker} from multiple data sources.

**CRITICAL: RECENCY WEIGHTING**
When analyzing analyst actions, apply strong recency bias:
- Actions from the last 7 days: VERY HIGH importance (weight: 1.0)
- Actions from 8-14 days ago: HIGH importance (weight: 0.7)
- Actions from 15-30 days ago: MODERATE importance (weight: 0.4)
- Anything older: LOW importance (mention but don't over-emphasize)

**NEW SCORING SYSTEM (-2 to +2):**
We're using a more intuitive scale where:
- **+2.0** = Strong Buy (very bullish)
- **+1.0** = Buy (bullish)
- **0.0** = Hold/Neutral
- **-1.0** = Sell (bearish)
- **-2.0** = Strong Sell (very bearish)

To convert from the traditional 1-5 scale (if present in data):
- 1.0-1.5 → +2.0 (Strong Buy)
- 1.5-2.5 → +1.0 (Buy)
- 2.5-3.5 → 0.0 (Hold)
- 3.5-4.5 → -1.0 (Sell)
- 4.5-5.0 → -2.0 (Strong Sell)

**Data Sources You're Analyzing:**
1. **Yahoo Finance**: Current analyst consensus, price targets, and recent recommendations
2. **Financial Modeling Prep**: Detailed upgrades/downgrades with firm names and dates
3. **MarketWatch**: Additional analyst estimates (when available)

**Your Analysis Must Prioritize:**

1. **Recent Actions** (Last 7 Days):
   - These should be prominently featured and heavily weighted
   - If there are upgrades in the last 7 days, this is VERY significant
   - If there are downgrades in the last 7 days, this is a WARNING SIGNAL

2. **Momentum Analysis**:
   - Are recent actions (last 7-14 days) more bullish or bearish than the overall consensus?
   - Is sentiment improving or deteriorating?

3. **Consensus Rating with Recency Adjustment**:
   - Start with the overall consensus from Yahoo Finance
   - Adjust your final score based on recent momentum
   - Example: If overall is "Hold" but last 3 actions in 7 days were upgrades → adjust toward "Buy"

4. **Price Targets**: Extract consensus price target with confidence based on analyst agreement

**Output Format Requirements:**

Return your analysis in this EXACT JSON format:
{{
  "consensus": "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell",
  "consensus_score": -2.0 to +2.0,
  "recent_momentum": "Very Bullish" | "Bullish" | "Neutral" | "Bearish" | "Very Bearish",
  "momentum_score": -2.0 to +2.0,
  "recency_adjusted_score": -2.0 to +2.0,
  "average_target": 0.0,
  "target_high": 0.0,
  "target_low": 0.0,
  "current_price": 0.0,
  "upside_percent": 0.0,
  "number_of_analysts": 0,
  "recent_activity": [
    {{
      "date": "YYYY-MM-DD",
      "days_ago": 0,
      "recency_tier": "Last 7 days" | "8-14 days" | "15-30 days",
      "firm": "Analyst Firm Name",
      "action": "upgrade" | "downgrade" | "initiated" | "reiterated",
      "from_grade": "Hold",
      "to_grade": "Buy",
      "impact": "very_positive" | "positive" | "neutral" | "negative" | "very_negative"
    }}
  ],
  "analysis_summary": "A detailed 3-4 sentence summary that EMPHASIZES recent actions in the last 7-14 days, explains the consensus, discusses price target and upside, and notes whether sentiment is improving or deteriorating.",
  "detailed_analysis": {{
    "consensus_breakdown": "Explain the consensus WITH emphasis on how recent actions (last 7 days) compare to the overall view. If recent momentum differs from consensus, highlight this.",
    "price_target_analysis": "Analysis of the price target range, confidence level, and whether recent actions suggest targets may need revision",
    "recent_trends": "CRITICAL SECTION: Detailed analysis of the last 7-14 days of analyst actions. What's the trend? Are upgrades accelerating? Are downgrades piling up? This is your MOST IMPORTANT section.",
    "analyst_divergence": "Discussion of any significant disagreements, especially recent ones",
    "key_catalysts": ["catalyst 1", "catalyst 2"],
    "key_risks": ["risk 1", "risk 2"],
    "investment_strategy": "Recommended approach based on RECENT momentum and analyst views. If recent actions are very bullish, recommend accumulation. If bearish, recommend caution.",
    "time_horizon": "short-term" | "medium-term" | "long-term",
    "confidence_level": "high" | "medium" | "low"
  }},
  "data_quality": "excellent" | "good" | "limited" | "poor",
  "data_sources_used": ["Yahoo Finance", "FMP", "MarketWatch"]
}}

**Example Recency-Weighted Analysis:**
If you see:
- Overall consensus: Hold (0.0)
- Last 7 days: 2 upgrades, 0 downgrades
- 8-14 days: 1 upgrade, 0 downgrades
- 15-30 days: 0 upgrades, 1 downgrade

Your analysis should say:
- Consensus: Hold → Buy (upgrading due to recent momentum)
- Recency-adjusted score: +0.5 to +1.0 (bullish momentum)
- Recent momentum: "Bullish"
- Strategy: "Accumulate on dips - recent analyst upgrades signal improving sentiment"

**Raw Data from APIs:**

<api_data>
{raw_data_json}
</api_data>

Return ONLY the JSON object, no additional text or markdown."""
)


# --- REPORT GENERATION ---


def generate_detailed_report(ticker: str, analysis: Dict[str, Any]) -> str:
    """
    Generate a comprehensive markdown report from analysis results.

    Args:
        ticker: Stock ticker symbol
        analysis: Detailed analysis JSON from LLM

    Returns:
        Formatted markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    consensus = analysis.get("consensus", "N/A")
    consensus_score = analysis.get("consensus_score", 0.0)
    recency_score = analysis.get("recency_adjusted_score", consensus_score)
    momentum = analysis.get("recent_momentum", "Neutral")
    momentum_score = analysis.get("momentum_score", 0.0)
    avg_target = analysis.get("average_target", 0.0)
    target_high = analysis.get("target_high", 0.0)
    target_low = analysis.get("target_low", 0.0)
    current_price = analysis.get("current_price", 0.0)
    upside = analysis.get("upside_percent", 0.0)
    num_analysts = analysis.get("number_of_analysts", 0)

    # Consensus emoji with new -2 to +2 scale
    def get_rating_emoji(score):
        if score >= 1.5:
            return "🟢🟢"
        elif score >= 0.5:
            return "🟢"
        elif score >= -0.5:
            return "🟡"
        elif score >= -1.5:
            return "🔴"
        else:
            return "🔴🔴"

    consensus_emoji = get_rating_emoji(consensus_score)
    momentum_emoji = get_rating_emoji(momentum_score)

    report = f"""
# Professional Analyst Report: ${ticker}
**Generated:** {timestamp}
**Agent:** Analyst Agent (Enhanced with Recency Weighting)
**Model:** Gemini Pro
**Data Sources:** Yahoo Finance, Financial Modeling Prep, MarketWatch

---

## Executive Summary

{analysis.get('analysis_summary', 'No summary available.')}

---

## Wall Street Consensus (New -2 to +2 Scale)

{consensus_emoji} **Overall Rating:** {consensus}  
**Consensus Score:** {consensus_score:+.2f} / 2.0  
**Analyst Coverage:** {num_analysts} professional analysts

{momentum_emoji} **Recent Momentum:** {momentum}  
**Momentum Score:** {momentum_score:+.2f} / 2.0  
**Recency-Adjusted Score:** {recency_score:+.2f} / 2.0

### Scoring System
- **+2.0 to +1.5**: Strong Buy 🟢🟢
- **+1.5 to +0.5**: Buy 🟢
- **+0.5 to -0.5**: Hold 🟡
- **-0.5 to -1.5**: Sell 🔴
- **-1.5 to -2.0**: Strong Sell 🔴🔴

### Consensus Interpretation
"""

    detailed = analysis.get("detailed_analysis", {})
    report += (
        f"{detailed.get('consensus_breakdown', 'No detailed breakdown available.')}\n\n"
    )

    report += f"""
---

## Price Target Analysis

**Current Price:** ${current_price:,.2f}  
**Average Target:** ${avg_target:,.2f}  
**Upside Potential:** {upside:+.1f}%  
**Target Range:** ${target_low:,.2f} - ${target_high:,.2f}

### Price Target Breakdown
"""

    report += f"{detailed.get('price_target_analysis', 'No detailed price target analysis available.')}\n\n"

    # Recent Activity with recency tiers
    report += "---\n\n## Recent Analyst Activity (Prioritized by Recency)\n\n"

    activity = analysis.get("recent_activity", [])
    if activity:
        # Group by recency tier
        last_7_days = [a for a in activity if a.get("days_ago", 999) <= 7]
        days_8_14 = [a for a in activity if 7 < a.get("days_ago", 999) <= 14]
        days_15_30 = [a for a in activity if 14 < a.get("days_ago", 999) <= 30]

        if last_7_days:
            report += "### 🔥 LAST 7 DAYS (Highest Priority)\n\n"
            for item in sorted(last_7_days, key=lambda x: x.get("days_ago", 0)):
                report += format_activity_item(item)

        if days_8_14:
            report += "### 📊 8-14 DAYS AGO (High Priority)\n\n"
            for item in sorted(days_8_14, key=lambda x: x.get("days_ago", 0)):
                report += format_activity_item(item)

        if days_15_30:
            report += "### 📅 15-30 DAYS AGO (Moderate Priority)\n\n"
            for item in sorted(days_15_30, key=lambda x: x.get("days_ago", 0)):
                report += format_activity_item(item)
    else:
        report += "*No recent analyst actions found in the last 30 days.*\n\n"

    report += f"""
### 🎯 Recent Trends Analysis (CRITICAL)
{detailed.get('recent_trends', 'No trend analysis available.')}

---

## Analyst Divergence & Disagreements

{detailed.get('analyst_divergence', 'Analysts appear to be in general agreement.')}

---

## Key Catalysts (Bull Case)

"""

    catalysts = detailed.get("key_catalysts", [])
    if catalysts:
        for catalyst in catalysts:
            report += f"- {catalyst}\n"
    else:
        report += "*No specific catalysts identified.*\n"

    report += "\n---\n\n## Key Risks (Bear Case)\n\n"

    risks = detailed.get("key_risks", [])
    if risks:
        for risk in risks:
            report += f"- {risk}\n"
    else:
        report += "*No specific risks identified.*\n"

    report += f"""

---

## Investment Strategy & Recommendations

### Recommended Approach (Based on Recent Momentum)
{detailed.get('investment_strategy', 'No specific strategy recommended.')}

### Time Horizon
**{detailed.get('time_horizon', 'medium-term').replace('-', ' ').title()}** outlook

### Confidence Level
**{detailed.get('confidence_level', 'medium').capitalize()}** confidence in analyst consensus

---

## Risk Factors to Monitor

Based on analyst coverage, investors should monitor:

"""

    if risks:
        for i, risk in enumerate(risks, 1):
            report += f"{i}. {risk}\n"
    else:
        report += "- Market conditions and sector performance\n"
        report += "- Company earnings and guidance\n"
        report += "- Competitive landscape changes\n"

    report += f"""

---

## Data Quality Assessment

**Overall Quality:** {analysis.get('data_quality', 'unknown').capitalize()}  
**Sources Used:** {', '.join(analysis.get('data_sources_used', ['Yahoo Finance']))}  
**Number of Analysts:** {num_analysts}  
**Last Updated:** {timestamp}

---

## Methodology Notes

This report synthesizes data from multiple sources with **strong recency weighting**:
- **Last 7 days**: Weighted 100% (highest priority)
- **8-14 days ago**: Weighted 70%
- **15-30 days ago**: Weighted 40%

### New Scoring System (-2 to +2)
This intuitive scale makes it easier to understand analyst sentiment:
- **+2.0**: Strong Buy (very bullish)
- **+1.0**: Buy (bullish)
- **0.0**: Hold/Neutral
- **-1.0**: Sell (bearish)
- **-2.0**: Strong Sell (very bearish)

### Data Sources
- **Yahoo Finance**: Consensus ratings, price targets, recent actions
- **Financial Modeling Prep**: Enhanced analyst estimates and upgrades
- **MarketWatch**: Additional validation when available

---

## Disclaimer

This report aggregates professional analyst opinions for informational purposes only. It is NOT financial advice. Analyst ratings can be biased, and price targets are often not achieved. **Recent analyst actions are emphasized but should not be the sole basis for investment decisions.**

Always conduct your own due diligence and consult with a qualified financial advisor.

**Past performance does not guarantee future results.**

---

*Report generated by AgentSeer Analyst Agent (Enhanced)*  
*Timestamp: {timestamp}*  
*Ticker: ${ticker}*
"""

    return report.strip()


def format_activity_item(item: Dict[str, Any]) -> str:
    """Format a single analyst activity item."""
    action = item.get("action", "N/A").lower()
    action_emoji = {
        "upgrade": "⬆️",
        "downgrade": "⬇️",
        "initiated": "🆕",
        "reiterated": "↔️",
        "main": "➡️",
    }.get(action, "•")

    impact = item.get("impact", "neutral")
    impact_emoji = {
        "very_positive": "🟢🟢",
        "positive": "🟢",
        "neutral": "⚪",
        "negative": "🔴",
        "very_negative": "🔴🔴",
    }.get(impact, "⚪")

    date_str = item.get("date", "Recent")
    days_ago = item.get("days_ago", "?")
    firm = item.get("firm", "Unknown Firm")
    action_text = item.get("action", "N/A").capitalize()
    from_grade = item.get("from_grade", "N/A")
    to_grade = item.get("to_grade", "N/A")

    result = f"#### {action_emoji} {firm} - {date_str} ({days_ago} days ago) {impact_emoji}\n"
    result += f"**Action:** {action_text}\n"
    if from_grade != "N/A" and to_grade != "N/A":
        result += f"**Change:** {from_grade} → {to_grade}\n"
    result += "\n"

    return result


def save_report(ticker: str, report: str, output_dir: str = "reports") -> str:
    """
    Save the detailed report to a text file.

    Args:
        ticker: Stock ticker symbol
        report: Formatted report content
        output_dir: Directory to save reports (created if doesn't exist)

    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_analyst_report_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    # Write report to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 Detailed report saved to: {filepath}")
    return filepath


# --- CREATE ENHANCED CHAIN ---

agent_chain = (
    RunnablePassthrough.assign(api_results=run_analyst_search)
    | RunnablePassthrough.assign(
        ticker=lambda x: x["api_results"]["ticker"],
        raw_data_json=lambda x: x["api_results"]["raw_data_json"],
    )
    | prompt_template
    | llm
    | JsonOutputParser()
)


# --- MAIN AGENT FUNCTION ---


def run_analyst_agent(ticker: str, save_to_file: bool = True) -> tuple[str, str]:
    """
    Execute the enhanced Analyst Agent analysis.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        save_to_file: Whether to save the detailed report to a file

    Returns:
        Tuple of (summary_report, detailed_report)
    """
    print(f"\n{'='*60}")
    print(f"🤖 Analyst Agent: Analyzing analyst ratings for ${ticker}")
    print(f"{'='*60}\n")

    try:
        # Execute the analysis chain
        analysis_json = agent_chain.invoke({"ticker": ticker.upper()})

        # Generate detailed report first
        detailed_report = generate_detailed_report(ticker, analysis_json)

        # Save to file if requested
        if save_to_file:
            save_report(ticker, detailed_report)

        # Extract data for summary
        consensus = analysis_json.get("consensus", "N/A")
        consensus_score = analysis_json.get("consensus_score", 0.0)
        recency_score = analysis_json.get("recency_adjusted_score", consensus_score)
        momentum = analysis_json.get("recent_momentum", "Neutral")
        momentum_score = analysis_json.get("momentum_score", 0.0)
        avg_target = analysis_json.get("average_target", 0.0)
        target_high = analysis_json.get("target_high", 0.0)
        target_low = analysis_json.get("target_low", 0.0)
        current_price = analysis_json.get("current_price", 0.0)
        upside = analysis_json.get("upside_percent", 0.0)
        num_analysts = analysis_json.get("number_of_analysts", 0)
        activity = analysis_json.get("recent_activity", [])
        summary = analysis_json.get("analysis_summary", "No summary provided.")
        data_quality = analysis_json.get("data_quality", "unknown")
        sources = analysis_json.get("data_sources_used", [])

        # Determine emoji with new scale
        def get_rating_emoji(score):
            if score >= 1.5:
                return "🟢🟢"
            elif score >= 0.5:
                return "🟢"
            elif score >= -0.5:
                return "🟡"
            elif score >= -1.5:
                return "🔴"
            else:
                return "🔴🔴"

        consensus_emoji = get_rating_emoji(consensus_score)
        momentum_emoji = get_rating_emoji(momentum_score)

        # Format the summary report (for console output)
        summary_report = f"""
**Analyst Agent Report: ${ticker}**

{consensus_emoji} **Wall Street Consensus: {consensus}** (Score: {consensus_score:+.2f}/2.0)
{momentum_emoji} **Recent Momentum: {momentum}** (Score: {momentum_score:+.2f}/2.0)
**Recency-Adjusted Score:** {recency_score:+.2f}/2.0

📊 **Price Target Analysis**
* **Current Price:** ${current_price:,.2f}
* **Average Target:** ${avg_target:,.2f} ({upside:+.1f}% upside)
* **Target Range:** ${target_low:,.2f} - ${target_high:,.2f}
* **Analyst Coverage:** {num_analysts} analysts

📈 **Recent Analyst Activity (Prioritized by Recency):**
"""

        if not activity:
            summary_report += "* *No recent upgrades or downgrades found.*\n"
        else:
            # Show most recent 3, grouped by recency
            last_7_days = [a for a in activity if a.get("days_ago", 999) <= 7]
            days_8_14 = [a for a in activity if 7 < a.get("days_ago", 999) <= 14]

            shown = 0
            if last_7_days:
                summary_report += "\n🔥 **LAST 7 DAYS (High Priority):**\n"
                for item in sorted(last_7_days, key=lambda x: x.get("days_ago", 0))[:2]:
                    action_emoji = {
                        "upgrade": "⬆️",
                        "downgrade": "⬇️",
                        "initiated": "🆕",
                        "reiterated": "↔️",
                    }.get(item.get("action", "").lower(), "•")

                    summary_report += f"  {action_emoji} **{item.get('firm', 'Unknown')}** ({item.get('days_ago', '?')} days ago)"
                    if item.get("from_grade") and item.get("to_grade"):
                        summary_report += f": {item['from_grade']} → {item['to_grade']}"
                    summary_report += "\n"
                    shown += 1

            if shown < 3 and days_8_14:
                summary_report += "\n📊 **8-14 DAYS AGO:**\n"
                for item in sorted(days_8_14, key=lambda x: x.get("days_ago", 0))[
                    : 3 - shown
                ]:
                    action_emoji = {
                        "upgrade": "⬆️",
                        "downgrade": "⬇️",
                        "initiated": "🆕",
                        "reiterated": "↔️",
                    }.get(item.get("action", "").lower(), "•")

                    summary_report += f"  {action_emoji} **{item.get('firm', 'Unknown')}** ({item.get('days_ago', '?')} days ago)"
                    if item.get("from_grade") and item.get("to_grade"):
                        summary_report += f": {item['from_grade']} → {item['to_grade']}"
                    summary_report += "\n"

        summary_report += f"""

📝 **Analysis Summary:**
{summary}

📊 **Data Quality:** {data_quality.capitalize()}
🔍 **Sources:** {', '.join(sources) if sources else 'Yahoo Finance'}

💡 **New Scoring:** Using -2 to +2 scale (more intuitive)
   • +2.0 = Strong Buy 🟢🟢 | +1.0 = Buy 🟢 | 0.0 = Hold 🟡
   • -1.0 = Sell 🔴 | -2.0 = Strong Sell 🔴🔴

---
*Full detailed report with recency weighting saved to file*
*Agent: Analyst Agent (Enhanced) | Model: Gemini Pro*
        """

        print("✅ Analyst_Agent: Analysis complete")
        return summary_report.strip(), detailed_report

    except Exception as e:
        print(f"❌ Analyst_Agent: Chain execution failed - {e}")
        import traceback

        traceback.print_exc()

        error_report = f"""
**Analyst Agent Report: ${ticker}**

⚠️ **Error:** Analysis could not be completed.

**Details:** {str(e)}

**Troubleshooting:**
- Ensure yfinance is installed: `pip install yfinance`
- Check your internet connection
- Verify the ticker symbol is correct
        """
        return error_report.strip(), error_report.strip()


# --- CLI ENTRY POINT ---

if __name__ == "__main__":
    import sys

    # Check dependencies
    if yf is None:
        print("\n❌ Missing required package: yfinance")
        print("Install with: pip install yfinance\n")
        sys.exit(1)

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "AAPL"  # Default example
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python analyst_agent.py <TICKER>\n")

    # Run the agent
    summary, detailed = run_analyst_agent(ticker, save_to_file=True)

    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY OUTPUT (Console)")
    print("=" * 60 + "\n")
    print(summary)
    print("\n" + "=" * 60)
    print("\n📄 Full detailed report saved to 'reports/' directory")
