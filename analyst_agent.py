"""
Analyst Agent - Analyst Swarm Component (Enhanced with Computational Recency)

This agent aggregates analyst ratings from multiple free sources with
proper computational recency weighting before LLM analysis.

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

    if "FMP_API_KEY" not in os.environ:
        print("ℹ️ FMP_API_KEY not set - some features limited")

    print("✅ Analyst_Agent: Environment variables validated")


validate_environment()

# --- SETUP LLM ---

llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest", temperature=0.1, api_key=os.environ["GEMINI_API_KEY"]
)

# --- COMPUTATIONAL RECENCY WEIGHTING ---


def calculate_weighted_sentiment_score(upgrades_list: List[Dict]) -> Dict[str, Any]:
    """
    Calculate a weighted sentiment score based on recency and action type.
    This provides objective metrics BEFORE LLM analysis.

    Args:
        upgrades_list: List of analyst actions with days_ago field

    Returns:
        Dictionary with weighted score, trend, and statistics
    """
    if not upgrades_list:
        return {
            "weighted_score": 0.0,
            "trend": "neutral",
            "recent_action_count": 0,
            "last_7_days_net": 0,
            "last_14_days_net": 0,
            "last_30_days_net": 0,
            "momentum_indicator": "no_data",
        }

    # Initialize counters
    total_weight = 0
    weighted_score = 0

    # Count actions by time period
    last_7_days = {"upgrades": 0, "downgrades": 0, "neutral": 0}
    last_14_days = {"upgrades": 0, "downgrades": 0, "neutral": 0}
    last_30_days = {"upgrades": 0, "downgrades": 0, "neutral": 0}

    for action in upgrades_list:
        days_ago = action.get("days_ago", 999)
        action_type = action.get("action", "").lower()

        # Determine base score (-1 to +1)
        if "upgrade" in action_type or "up" in action_type:
            base_score = 1.0
            action_category = "upgrades"
        elif "downgrade" in action_type or "down" in action_type:
            base_score = -1.0
            action_category = "downgrades"
        else:
            base_score = 0.0
            action_category = "neutral"

        # Apply recency weight (exponential decay)
        if days_ago <= 7:
            weight = 1.0
            last_7_days[action_category] += 1
            last_14_days[action_category] += 1
            last_30_days[action_category] += 1
        elif days_ago <= 14:
            weight = 0.7
            last_14_days[action_category] += 1
            last_30_days[action_category] += 1
        elif days_ago <= 30:
            weight = 0.4
            last_30_days[action_category] += 1
        else:
            weight = 0.1

        weighted_score += base_score * weight
        total_weight += weight

    # Calculate final score
    final_score = weighted_score / total_weight if total_weight > 0 else 0.0

    # Calculate net sentiment for each period
    last_7_net = last_7_days["upgrades"] - last_7_days["downgrades"]
    last_14_net = last_14_days["upgrades"] - last_14_days["downgrades"]
    last_30_net = last_30_days["upgrades"] - last_30_days["downgrades"]

    # Determine momentum
    if last_7_net > 0 and last_14_net >= 0:
        momentum = "accelerating_bullish"
    elif last_7_net > 0:
        momentum = "bullish"
    elif last_7_net < 0 and last_14_net <= 0:
        momentum = "accelerating_bearish"
    elif last_7_net < 0:
        momentum = "bearish"
    else:
        momentum = "neutral"

    # Determine trend
    if final_score > 0.4:
        trend = "strong_bullish"
    elif final_score > 0.15:
        trend = "bullish"
    elif final_score < -0.4:
        trend = "strong_bearish"
    elif final_score < -0.15:
        trend = "bearish"
    else:
        trend = "neutral"

    return {
        "weighted_score": round(final_score, 3),
        "trend": trend,
        "recent_action_count": last_7_days["upgrades"] + last_7_days["downgrades"],
        "last_7_days_net": last_7_net,
        "last_14_days_net": last_14_net,
        "last_30_days_net": last_30_net,
        "momentum_indicator": momentum,
        "last_7_days_detail": last_7_days,
        "last_14_days_detail": last_14_days,
        "last_30_days_detail": last_30_days,
    }


def adjust_consensus_with_momentum(
    base_consensus_score: float, momentum_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Adjust the base consensus score using recent momentum.

    Args:
        base_consensus_score: Original consensus score (1-5 scale or -2 to +2)
        momentum_metrics: Output from calculate_weighted_sentiment_score

    Returns:
        Adjusted score and explanation
    """
    momentum_score = momentum_metrics["weighted_score"]
    momentum_indicator = momentum_metrics["momentum_indicator"]

    # Convert momentum to adjustment factor (-1 to +1 on our -2 to +2 scale)
    adjustment = momentum_score * 0.5  # Scale down the adjustment

    # Apply adjustment
    adjusted_score = base_consensus_score + adjustment

    # Clamp to valid range
    adjusted_score = max(-2.0, min(2.0, adjusted_score))

    return {
        "adjusted_score": round(adjusted_score, 2),
        "adjustment_amount": round(adjustment, 2),
        "momentum_influence": momentum_indicator,
        "explanation": f"Base score {base_consensus_score:+.2f} adjusted by {adjustment:+.2f} due to {momentum_indicator} momentum",
    }


# --- DATA CONVERSION UTILITIES ---


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


# --- DATA FETCHING ---


def fetch_yahoo_finance_data(ticker: str) -> Dict[str, Any]:
    """Fetch analyst data from Yahoo Finance using yfinance."""
    if yf is None:
        return {"error": "yfinance not installed"}

    try:
        stock = yf.Ticker(ticker)

        # Get recommendations
        recent_recs = []
        try:
            recommendations = stock.recommendations
            if recommendations is not None and not recommendations.empty:
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

        # Get upgrades/downgrades with recency tracking
        upgrades_list = []
        try:
            upgrades = stock.upgrades_downgrades
            if upgrades is not None and not upgrades.empty:
                thirty_days_ago = datetime.now() - timedelta(days=30)
                recent_upgrades = upgrades[upgrades.index >= thirty_days_ago]

                for date_idx, row in recent_upgrades.iterrows():
                    upgrade_date = (
                        date_idx
                        if hasattr(date_idx, "date")
                        else datetime.strptime(str(date_idx)[:10], "%Y-%m-%d")
                    )
                    days_ago = (datetime.now() - upgrade_date).days

                    if days_ago <= 30:
                        upgrade_entry = {
                            "date": (
                                date_idx.strftime("%Y-%m-%d")
                                if hasattr(date_idx, "strftime")
                                else str(date_idx)
                            ),
                            "days_ago": days_ago,
                            "firm": convert_to_serializable(row.get("Firm", "Unknown")),
                            "toGrade": convert_to_serializable(
                                row.get("ToGrade", "N/A")
                            ),
                            "fromGrade": convert_to_serializable(
                                row.get("FromGrade", "N/A")
                            ),
                            "action": convert_to_serializable(row.get("Action", "N/A")),
                            "recency_weight": (
                                1.0 if days_ago <= 7 else 0.7 if days_ago <= 14 else 0.4
                            ),
                        }
                        upgrades_list.append(upgrade_entry)
        except Exception as e:
            print(f"⚠️ Could not fetch upgrades/downgrades: {e}")

        # NEW: Calculate computational momentum metrics
        momentum_metrics = calculate_weighted_sentiment_score(upgrades_list)

        return {
            "source": "Yahoo Finance",
            "recent_recommendations": recent_recs,
            "analyst_info": analyst_info,
            "upgrades_downgrades": upgrades_list,
            "momentum_metrics": momentum_metrics,  # NEW
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
        return {"error": f"Yahoo Finance error: {str(e)}"}


def fetch_fmp_data(ticker: str) -> Dict[str, Any]:
    """Fetch analyst data from Financial Modeling Prep API."""
    if requests is None or "FMP_API_KEY" not in os.environ:
        return {"error": "FMP API not available"}

    api_key = os.environ.get("FMP_API_KEY")
    base_url = "https://financialmodelingprep.com/api/v3"

    try:
        results = {}

        # Get analyst estimates
        try:
            url = f"{base_url}/analyst-estimates/{ticker}?apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                results["analyst_estimates"] = response.json()
        except:
            pass

        # Get upgrades/downgrades
        try:
            url = f"{base_url}/upgrades-downgrades?symbol={ticker}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                thirty_days_ago = datetime.now() - timedelta(days=30)
                recent = [
                    item
                    for item in data
                    if datetime.strptime(
                        item.get("publishedDate", "2000-01-01"), "%Y-%m-%d %H:%M:%S"
                    )
                    >= thirty_days_ago
                ]
                results["upgrades_downgrades"] = recent[:10]
        except:
            pass

        # Get price target consensus
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


@lru_cache(maxsize=100)
def cached_analyst_fetch(ticker: str, timestamp_hour: int) -> Dict[str, Any]:
    """Cached fetch from all sources. Cache expires every hour."""
    print(f"🔍 Analyst_Agent: Fetching analyst data for {ticker}...")

    yahoo_data = fetch_yahoo_finance_data(ticker)
    fmp_data = fetch_fmp_data(ticker)

    return {
        "ticker": ticker,
        "yahoo_finance": yahoo_data,
        "fmp": fmp_data,
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
    """Execute analyst data search with caching and error handling."""
    ticker = input_dict.get("ticker", "").upper()

    if not ticker:
        return {"raw_data_json": "Error: No ticker provided", "ticker": ""}

    try:
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


# --- ENHANCED PROMPT WITH COMPUTATIONAL METRICS ---

prompt_template = ChatPromptTemplate.from_template(
    """You are an expert "Analyst Agent" analyzing analyst ratings for ${ticker}.

**IMPORTANT: COMPUTATIONAL METRICS PROVIDED**

The data includes pre-calculated momentum metrics based on recent analyst actions:
- weighted_score: Recency-weighted sentiment (-1.0 to +1.0)
- trend: Overall trend classification
- momentum_indicator: Recent momentum direction
- last_7_days_net: Net upgrades minus downgrades in last 7 days
- last_14_days_net: Net for last 14 days
- last_30_days_net: Net for last 30 days

**YOU MUST:**
1. Use these computational metrics as your PRIMARY signal for recent momentum
2. Apply the provided weighted_score to adjust the consensus
3. Cite specific numbers from the momentum metrics

**SCORING SYSTEM (-2 to +2):**
- **+2.0** = Strong Buy (very bullish)
- **+1.0** = Buy (bullish)
- **0.0** = Hold/Neutral
- **-1.0** = Sell (bearish)
- **-2.0** = Strong Sell (very bearish)

Convert traditional 1-5 scale:
- 1.0-1.5 → +2.0 (Strong Buy)
- 1.5-2.5 → +1.0 (Buy)
- 2.5-3.5 → 0.0 (Hold)
- 3.5-4.5 → -1.0 (Sell)
- 4.5-5.0 → -2.0 (Strong Sell)

**ADJUST CONSENSUS WITH MOMENTUM:**
1. Start with base consensus from analyst_info
2. Look at momentum_metrics.weighted_score
3. If momentum_metrics.last_7_days_net is strongly positive/negative, this should significantly influence your recency_adjusted_score
4. The recency_adjusted_score should differ from consensus_score when recent momentum diverges

**Output Format:**

{{
  "consensus": "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell",
  "consensus_score": -2.0 to +2.0,
  "recent_momentum": "Very Bullish" | "Bullish" | "Neutral" | "Bearish" | "Very Bearish",
  "momentum_score": -2.0 to +2.0,
  "recency_adjusted_score": -2.0 to +2.0,
  "computational_metrics": {{
    "weighted_score": "from momentum_metrics",
    "last_7_days_net": "from momentum_metrics",
    "momentum_indicator": "from momentum_metrics"
  }},
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
  "analysis_summary": "Detailed summary emphasizing the computational metrics",
  "detailed_analysis": {{
    "consensus_breakdown": "Explain consensus and how computational momentum metrics affect it",
    "price_target_analysis": "Analysis of price targets",
    "recent_trends": "CITE the computational metrics: weighted_score of X, last_7_days_net of Y means...",
    "analyst_divergence": "Any disagreements",
    "key_catalysts": ["catalyst 1", "catalyst 2"],
    "key_risks": ["risk 1", "risk 2"],
    "investment_strategy": "Strategy based on momentum metrics",
    "time_horizon": "short-term" | "medium-term" | "long-term",
    "confidence_level": "high" | "medium" | "low"
  }},
  "data_quality": "excellent" | "good" | "limited" | "poor",
  "data_sources_used": ["Yahoo Finance", "FMP"]
}}

**Raw Data:**

<api_data>
{raw_data_json}
</api_data>

Return ONLY the JSON object."""
)


# --- REPORT GENERATION (keeping existing implementation) ---


def generate_detailed_report(ticker: str, analysis: Dict[str, Any]) -> str:
    """Generate comprehensive markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    consensus = analysis.get("consensus", "N/A")
    consensus_score = analysis.get("consensus_score", 0.0)
    recency_score = analysis.get("recency_adjusted_score", consensus_score)
    momentum = analysis.get("recent_momentum", "Neutral")
    momentum_score = analysis.get("momentum_score", 0.0)

    # Extract computational metrics
    comp_metrics = analysis.get("computational_metrics", {})

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
**Agent:** Analyst Agent (Enhanced with Computational Recency)
**Model:** Gemini Pro

---

## Executive Summary

{analysis.get('analysis_summary', 'No summary available.')}

---

## Wall Street Consensus (With Computational Momentum)

{consensus_emoji} **Overall Rating:** {consensus}  
**Consensus Score:** {consensus_score:+.2f} / 2.0  
**Analyst Coverage:** {analysis.get('number_of_analysts', 0)} professional analysts

{momentum_emoji} **Recent Momentum:** {momentum}  
**Momentum Score:** {momentum_score:+.2f} / 2.0  
**Recency-Adjusted Score:** {recency_score:+.2f} / 2.0

### Computational Momentum Metrics
- **Weighted Score:** {comp_metrics.get('weighted_score', 'N/A')}
- **Last 7 Days Net:** {comp_metrics.get('last_7_days_net', 'N/A')}
- **Momentum Indicator:** {comp_metrics.get('momentum_indicator', 'N/A')}

### Scoring System
- **+2.0 to +1.5**: Strong Buy 🟢🟢
- **+1.5 to +0.5**: Buy 🟢
- **+0.5 to -0.5**: Hold 🟡
- **-0.5 to -1.5**: Sell 🔴
- **-1.5 to -2.0**: Strong Sell 🔴🔴

---

## Price Target Analysis

**Current Price:** ${analysis.get('current_price', 0):,.2f}  
**Average Target:** ${analysis.get('average_target', 0):,.2f}  
**Upside Potential:** {analysis.get('upside_percent', 0):+.1f}%  
**Target Range:** ${analysis.get('target_low', 0):,.2f} - ${analysis.get('target_high', 0):,.2f}

---

## Recent Analyst Activity

"""

    activity = analysis.get("recent_activity", [])
    if activity:
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
        report += "*No recent analyst actions found.*\n\n"

    detailed = analysis.get("detailed_analysis", {})

    report += f"""

---

## Detailed Analysis

### Recent Trends (Based on Computational Metrics)
{detailed.get('recent_trends', 'No trend analysis available.')}

### Consensus Breakdown
{detailed.get('consensus_breakdown', 'No breakdown available.')}

### Investment Strategy
{detailed.get('investment_strategy', 'No strategy provided.')}

---

## Risk Factors

"""

    risks = detailed.get("key_risks", [])
    if risks:
        for risk in risks:
            report += f"- {risk}\n"
    else:
        report += "*No specific risks identified.*\n"

    report += "\n## Key Catalysts\n\n"

    catalysts = detailed.get("key_catalysts", [])
    if catalysts:
        for catalyst in catalysts:
            report += f"- {catalyst}\n"
    else:
        report += "*No specific catalysts identified.*\n"

    report += f"""

---

## Methodology

This report uses **computational recency weighting** applied to analyst actions:
- Actions are weighted by recency (last 7 days = 1.0x, 8-14 days = 0.7x, etc.)
- A weighted sentiment score is calculated objectively
- The LLM then interprets this score along with qualitative factors

**Time Horizon:** {detailed.get('time_horizon', 'medium-term').replace('-', ' ').title()}  
**Confidence Level:** {detailed.get('confidence_level', 'medium').capitalize()}  
**Data Quality:** {analysis.get('data_quality', 'unknown').capitalize()}

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

    result = f"#### {action_emoji} {item.get('firm', 'Unknown')} - {item.get('date', 'Recent')} ({item.get('days_ago', '?')} days ago) {impact_emoji}\n"
    result += f"**Action:** {item.get('action', 'N/A').capitalize()}\n"
    if item.get("from_grade") != "N/A" and item.get("to_grade") != "N/A":
        result += f"**Change:** {item['from_grade']} → {item['to_grade']}\n"
    result += "\n"

    return result


def save_report(ticker: str, report: str, output_dir: str = "reports") -> str:
    """Save report to file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_analyst_report_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

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
    Execute the enhanced Analyst Agent analysis with computational recency weighting.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        save_to_file: Whether to save the detailed report to a file

    Returns:
        Tuple of (summary_report, detailed_report)
    """
    print(f"\n{'='*60}")
    print(f"🤖 Analyst Agent (Enhanced): Analyzing ${ticker}")
    print(f"{'='*60}\n")

    try:
        # Execute the analysis chain
        analysis_json = agent_chain.invoke({"ticker": ticker.upper()})

        # Generate detailed report
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
        comp_metrics = analysis_json.get("computational_metrics", {})

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

        summary_report = f"""
**Analyst Agent Report: ${ticker}**

{consensus_emoji} **Wall Street Consensus: {consensus}** (Score: {consensus_score:+.2f}/2.0)
{momentum_emoji} **Recent Momentum: {momentum}** (Score: {momentum_score:+.2f}/2.0)
**Recency-Adjusted Score:** {recency_score:+.2f}/2.0

📊 **Computational Momentum Metrics:**
* **Weighted Score:** {comp_metrics.get('weighted_score', 'N/A')}
* **Last 7 Days Net:** {comp_metrics.get('last_7_days_net', 'N/A')} analyst actions
* **Momentum Indicator:** {comp_metrics.get('momentum_indicator', 'N/A')}

📈 **Price Target Analysis:**
* **Current Price:** ${analysis_json.get('current_price', 0):,.2f}
* **Average Target:** ${analysis_json.get('average_target', 0):,.2f} ({analysis_json.get('upside_percent', 0):+.1f}% upside)
* **Analyst Coverage:** {analysis_json.get('number_of_analysts', 0)} analysts

📝 **Summary:**
{analysis_json.get('analysis_summary', 'No summary available.')}

💡 **Enhanced with Computational Recency Weighting**
This analysis uses objective mathematical weighting of recent analyst actions,
not just LLM interpretation. Recent actions (last 7 days) are weighted 1.0x,
8-14 days at 0.7x, and 15-30 days at 0.4x.

---
*Full detailed report saved to file*
*Agent: Analyst Agent (Enhanced) | Model: Gemini Pro*
        """

        print("✅ Analyst_Agent: Analysis complete")

        # Return comprehensive dictionary like news_agent
        return {
            "ticker": ticker,
            "agent": "Analyst",
            "consensus": consensus,
            "consensus_score": consensus_score,
            "recency_adjusted_score": recency_score,
            "recent_momentum": momentum,
            "momentum_score": momentum_score,
            "computational_metrics": comp_metrics,
            "current_price": analysis_json.get("current_price", 0),
            "average_target": analysis_json.get("average_target", 0),
            "target_high": analysis_json.get("target_high", 0),
            "target_low": analysis_json.get("target_low", 0),
            "upside_percent": analysis_json.get("upside_percent", 0),
            "number_of_analysts": analysis_json.get("number_of_analysts", 0),
            "recent_activity": analysis_json.get("recent_activity", []),
            "analysis_summary": analysis_json.get("analysis_summary", ""),
            "data_quality": analysis_json.get("data_quality", "unknown"),
            "data_sources_used": analysis_json.get("data_sources_used", []),
            "summary_report": summary_report.strip(),
            "detailed_report": detailed_report,
        }

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
        return {
            "ticker": ticker,
            "agent": "Analyst",
            "error": str(e),
            "summary_report": error_report.strip(),
            "detailed_report": error_report.strip(),
        }


# --- CLI ENTRY POINT ---

if __name__ == "__main__":
    import sys

    if yf is None:
        print("\n❌ Missing required package: yfinance")
        print("Install with: pip install yfinance\n")
        sys.exit(1)

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "AAPL"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python analyst_agent.py <TICKER>\n")

    result = run_analyst_agent(ticker, save_to_file=True)

    print("\n" + "=" * 60)
    print("SUMMARY OUTPUT (Console)")
    print("=" * 60 + "\n")
    print(result.get("summary_report", ""))
    print("\n" + "=" * 60)
    print("\n📄 Full detailed report saved to 'reports/' directory")
