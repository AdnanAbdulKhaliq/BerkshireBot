"""
Risk Assessment Agent - Analyst Swarm Component (Enhanced with Baseline Metrics)

This agent produces comprehensive risk assessment with both quantitative baseline
metrics and LLM-powered qualitative analysis.

Environment Variables Required:
    - GEMINI_API_KEY: Your Google Gemini API key
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

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

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- ENVIRONMENT VALIDATION ---


def validate_environment() -> None:
    """Validate all required API keys are set."""
    required_vars = {"GEMINI_API_KEY": "Gemini LLM for risk analysis"}

    missing = []
    for var_name, description in required_vars.items():
        if var_name not in os.environ:
            missing.append(f"  ❌ {var_name}: {description}")

    if missing:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing)
        raise EnvironmentError(error_msg)

    print("✅ Risk_Assessment_Agent: All environment variables validated")


validate_environment()

# --- SETUP MODEL ---

llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest", temperature=0.2, api_key=os.environ["GEMINI_API_KEY"]
)

# --- QUANTITATIVE BASELINE RISK METRICS ---


def calculate_baseline_risk_metrics(ticker: str) -> Dict[str, Any]:
    """
    Calculate objective, quantitative risk metrics as a baseline.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with calculated risk metrics and scores
    """
    if yf is None:
        return {"error": "yfinance not installed", "metrics_available": False}

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get historical data for volatility calculations
        hist = stock.history(period="1y")

        if hist.empty:
            return {"error": "No historical data available", "metrics_available": False}

        metrics = {}

        # 1. VOLATILITY RISK (0-100 scale)
        try:
            # Calculate annualized volatility
            returns = hist["Close"].pct_change().dropna()
            daily_volatility = returns.std()
            annual_volatility = daily_volatility * np.sqrt(252)

            # Convert to 0-100 scale (0.5 volatility = 50, 1.0 = 100)
            volatility_score = min(100, annual_volatility * 100)

            metrics["volatility"] = {
                "annual_volatility": float(annual_volatility),
                "risk_score": float(volatility_score),
                "risk_level": (
                    "Low"
                    if volatility_score < 30
                    else (
                        "Medium"
                        if volatility_score < 50
                        else "High" if volatility_score < 70 else "Critical"
                    )
                ),
                "description": f"Annualized volatility of {annual_volatility:.1%}",
            }
        except Exception as e:
            metrics["volatility"] = {"error": str(e), "risk_score": None}

        # 2. MARKET CORRELATION RISK (Beta)
        try:
            beta = info.get("beta")
            if beta is not None:
                beta = float(beta)
                # Beta deviation from 1.0 indicates correlation risk
                # Beta < 1: less volatile than market
                # Beta > 1: more volatile than market
                beta_deviation = abs(beta - 1.0)
                correlation_score = min(100, beta_deviation * 50)

                metrics["market_correlation"] = {
                    "beta": beta,
                    "risk_score": float(correlation_score),
                    "risk_level": (
                        "Low"
                        if correlation_score < 25
                        else (
                            "Medium"
                            if correlation_score < 50
                            else "High" if correlation_score < 75 else "Critical"
                        )
                    ),
                    "description": f"Beta of {beta:.2f} (market correlation)",
                }
            else:
                metrics["market_correlation"] = {
                    "error": "Beta not available",
                    "risk_score": None,
                }
        except Exception as e:
            metrics["market_correlation"] = {"error": str(e), "risk_score": None}

        # 3. LEVERAGE RISK (Debt-to-Equity)
        try:
            debt_to_equity = info.get("debtToEquity")
            if debt_to_equity is not None:
                debt_to_equity = float(debt_to_equity) / 100  # Convert from percentage
                # D/E > 2.0 is high risk
                leverage_score = min(100, debt_to_equity * 50)

                metrics["leverage"] = {
                    "debt_to_equity": debt_to_equity,
                    "risk_score": float(leverage_score),
                    "risk_level": (
                        "Low"
                        if leverage_score < 30
                        else (
                            "Medium"
                            if leverage_score < 60
                            else "High" if leverage_score < 80 else "Critical"
                        )
                    ),
                    "description": f"Debt-to-Equity ratio of {debt_to_equity:.2f}",
                }
            else:
                metrics["leverage"] = {"error": "D/E not available", "risk_score": None}
        except Exception as e:
            metrics["leverage"] = {"error": str(e), "risk_score": None}

        # 4. LIQUIDITY RISK (Current Ratio)
        try:
            current_ratio = info.get("currentRatio")
            if current_ratio is not None:
                current_ratio = float(current_ratio)
                # Lower current ratio = higher risk
                # Ideal is > 1.5, critical if < 1.0
                if current_ratio >= 1.5:
                    liquidity_score = 20
                elif current_ratio >= 1.0:
                    liquidity_score = 50
                elif current_ratio >= 0.5:
                    liquidity_score = 75
                else:
                    liquidity_score = 90

                metrics["liquidity"] = {
                    "current_ratio": current_ratio,
                    "risk_score": float(liquidity_score),
                    "risk_level": (
                        "Low"
                        if liquidity_score < 30
                        else (
                            "Medium"
                            if liquidity_score < 60
                            else "High" if liquidity_score < 80 else "Critical"
                        )
                    ),
                    "description": f"Current ratio of {current_ratio:.2f}",
                }
            else:
                metrics["liquidity"] = {
                    "error": "Current ratio not available",
                    "risk_score": None,
                }
        except Exception as e:
            metrics["liquidity"] = {"error": str(e), "risk_score": None}

        # 5. VALUATION RISK (P/E Ratio)
        try:
            pe_ratio = info.get("trailingPE")
            if pe_ratio is not None and pe_ratio > 0:
                pe_ratio = float(pe_ratio)
                # High P/E suggests overvaluation risk
                # P/E < 15: low risk, P/E > 50: high risk
                if pe_ratio < 15:
                    valuation_score = 20
                elif pe_ratio < 25:
                    valuation_score = 40
                elif pe_ratio < 40:
                    valuation_score = 60
                elif pe_ratio < 60:
                    valuation_score = 80
                else:
                    valuation_score = 95

                metrics["valuation"] = {
                    "pe_ratio": pe_ratio,
                    "risk_score": float(valuation_score),
                    "risk_level": (
                        "Low"
                        if valuation_score < 30
                        else (
                            "Medium"
                            if valuation_score < 60
                            else "High" if valuation_score < 80 else "Critical"
                        )
                    ),
                    "description": f"P/E ratio of {pe_ratio:.1f}",
                }
            else:
                metrics["valuation"] = {
                    "error": "P/E not available",
                    "risk_score": None,
                }
        except Exception as e:
            metrics["valuation"] = {"error": str(e), "risk_score": None}

        # 6. PRICE MOMENTUM RISK (52-week high/low distance)
        try:
            current_price = info.get("currentPrice") or hist["Close"].iloc[-1]
            week_52_high = info.get("fiftyTwoWeekHigh")
            week_52_low = info.get("fiftyTwoWeekLow")

            if week_52_high and week_52_low and current_price:
                # Calculate distance from 52-week range
                range_size = week_52_high - week_52_low
                distance_from_high = (week_52_high - current_price) / range_size

                # Near 52-week low = higher risk
                momentum_score = distance_from_high * 100

                metrics["price_momentum"] = {
                    "current_price": float(current_price),
                    "week_52_high": float(week_52_high),
                    "week_52_low": float(week_52_low),
                    "distance_from_high_pct": float(distance_from_high * 100),
                    "risk_score": float(momentum_score),
                    "risk_level": (
                        "Low"
                        if momentum_score < 30
                        else (
                            "Medium"
                            if momentum_score < 60
                            else "High" if momentum_score < 80 else "Critical"
                        )
                    ),
                    "description": f"Price is {distance_from_high:.1%} below 52-week high",
                }
            else:
                metrics["price_momentum"] = {
                    "error": "52-week data not available",
                    "risk_score": None,
                }
        except Exception as e:
            metrics["price_momentum"] = {"error": str(e), "risk_score": None}

        # Calculate overall baseline risk score (weighted average)
        valid_scores = [
            m["risk_score"] for m in metrics.values() if m.get("risk_score") is not None
        ]

        if valid_scores:
            overall_score = sum(valid_scores) / len(valid_scores)
            overall_level = (
                "Low"
                if overall_score < 30
                else (
                    "Medium"
                    if overall_score < 50
                    else "High" if overall_score < 70 else "Critical"
                )
            )
        else:
            overall_score = None
            overall_level = "Unknown"

        return {
            "ticker": ticker,
            "metrics_available": True,
            "individual_metrics": metrics,
            "overall_baseline_score": float(overall_score) if overall_score else None,
            "overall_baseline_level": overall_level,
            "metrics_count": len(valid_scores),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"❌ Error calculating baseline metrics: {e}")
        return {"error": str(e), "metrics_available": False}


def format_baseline_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format baseline metrics into readable text for LLM prompt.

    Args:
        metrics: Output from calculate_baseline_risk_metrics

    Returns:
        Formatted string
    """
    if not metrics.get("metrics_available"):
        return "**Baseline Risk Metrics:** Not available\n"

    formatted = "**QUANTITATIVE BASELINE RISK METRICS:**\n\n"
    formatted += f"Overall Baseline Risk Score: {metrics.get('overall_baseline_score', 'N/A'):.1f}/100\n"
    formatted += (
        f"Overall Baseline Risk Level: {metrics.get('overall_baseline_level', 'N/A')}\n"
    )
    formatted += f"Metrics Available: {metrics.get('metrics_count', 0)}/6\n\n"

    individual = metrics.get("individual_metrics", {})

    for metric_name, metric_data in individual.items():
        if metric_data.get("risk_score") is not None:
            formatted += f"**{metric_name.replace('_', ' ').title()}:**\n"
            formatted += f"  - Risk Score: {metric_data['risk_score']:.1f}/100\n"
            formatted += f"  - Risk Level: {metric_data['risk_level']}\n"
            formatted += f"  - Description: {metric_data['description']}\n\n"
        else:
            formatted += (
                f"**{metric_name.replace('_', ' ').title()}:** Data not available\n\n"
            )

    return formatted


# --- ENHANCED PROMPT WITH BASELINE METRICS ---

risk_assessment_prompt = ChatPromptTemplate.from_template(
    """You are a "Risk Assessment Specialist" evaluating investment risks for ${ticker}.

**QUANTITATIVE BASELINE METRICS PROVIDED:**

{baseline_metrics}

**IMPORTANT:** These are OBJECTIVE, CALCULATED metrics based on financial data.
Your LLM-generated risk scores should be INFORMED BY but not necessarily MATCH these baseline scores.

**YOUR MANDATE:**
Analyze the Governor's memo and produce a detailed risk assessment that:
1. **Acknowledges the baseline metrics** where available
2. **Adjusts scores based on qualitative factors** from the memo
3. **Explains divergences** between baseline and your assessment

For each risk category:
- Start with the baseline score if available
- Adjust based on qualitative evidence from the memo
- Explain your reasoning

**SCORING (0-100):**
- **0-25:** Low Risk
- **26-50:** Medium Risk
- **51-75:** High Risk
- **76-100:** Critical Risk

**RISK CATEGORIES:**

1. **Market Risk** - Use baseline market_correlation + memo context
2. **Company-Specific Risk** - Use baseline leverage + liquidity + memo
3. **Sentiment Risk** - Primarily from memo (no baseline)
4. **Technical Risk** - Use baseline price_momentum + volatility + memo
5. **Regulatory/Legal Risk** - Primarily from memo (no baseline)
6. **Macroeconomic Risk** - From memo context (no baseline)

**GOVERNOR'S INVESTMENT MEMO:**

<investment_memo>
{investment_memo}
</investment_memo>

Return your risk assessment in this JSON format:
{{
  "overall_risk_score": 0-100,
  "overall_risk_level": "Low" | "Medium" | "High" | "Critical",
  "overall_assessment": "2-3 sentence executive summary",
  "baseline_metrics_used": true | false,
  "baseline_vs_qualitative_note": "Explanation of how baseline metrics were factored in",
  "risk_categories": [
    {{
      "category": "Market Risk",
      "risk_score": 0-100,
      "risk_level": "Low" | "Medium" | "High" | "Critical",
      "baseline_score": "from baseline metrics or null",
      "adjustment_reason": "Why your score differs from baseline",
      "confidence": "high" | "medium" | "low",
      "key_risk_factors": [
        {{
          "factor": "Specific risk factor",
          "evidence": "Quote from memo or baseline metric",
          "source_agent": "Which agent or 'baseline_metrics'",
          "severity": "high" | "medium" | "low"
        }}
      ],
      "mitigations": [
        {{
          "strategy": "Mitigation strategy",
          "effectiveness": "high" | "medium" | "low",
          "implementation": "How to implement"
        }}
      ],
      "trend": "improving" | "stable" | "deteriorating" | "unknown",
      "analysis": "2-3 sentence analysis"
    }}
  ],
  "red_flags": [
    {{
      "flag": "Critical concern",
      "severity": "high" | "medium" | "low",
      "source_evidence": "Evidence",
      "immediate_action": "What investor should do"
    }}
  ],
  "risk_opportunities": [
    {{
      "opportunity": "Risk that could become opportunity",
      "scenario": "Under what conditions",
      "potential_upside": "What could be gained"
    }}
  ],
  "portfolio_implications": {{
    "recommended_position_size": "Conservative" | "Moderate" | "Aggressive" | "Avoid",
    "holding_period": "Short-term" | "Medium-term" | "Long-term",
    "diversification_needs": "Description",
    "risk_monitoring": ["Key metric to monitor"]
  }},
  "stress_test_scenarios": [
    {{
      "scenario": "What if scenario",
      "probability": "high" | "medium" | "low",
      "impact_on_investment": "Description",
      "preparedness": "How well positioned"
    }}
  ],
  "comparable_risk_analysis": {{
    "industry_comparison": "How this compares to industry",
    "historical_comparison": "How current risk compares historically",
    "market_comparison": "How this compares to broader market"
  }},
  "risk_reward_ratio": {{
    "potential_upside": "percentage or description",
    "potential_downside": "percentage or description",
    "ratio_assessment": "Assessment",
    "favorable_for": "Conservative investors" | "Moderate investors" | "Aggressive investors" | "None"
  }},
  "final_recommendation": {{
    "suitability": "Which investor types",
    "key_considerations": ["consideration 1", "consideration 2"],
    "monitoring_frequency": "How often to review",
    "exit_triggers": ["Event that should trigger exit"]
  }},
  "data_limitations": ["limitation 1", "limitation 2"],
  "assessment_confidence": "high" | "medium" | "low",
  "last_updated": "{timestamp}"
}}

<ticker>
{ticker}
</ticker>

<timestamp>
{timestamp}
</timestamp>

Return ONLY valid JSON."""
)

# --- CREATE RISK ASSESSMENT CHAIN ---


def prepare_risk_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare input with baseline metrics."""
    ticker = input_dict.get("ticker", "").upper()
    investment_memo = input_dict.get("investment_memo", "")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate baseline metrics
    print(f"📊 Calculating quantitative baseline risk metrics for {ticker}...")
    baseline_metrics = calculate_baseline_risk_metrics(ticker)
    formatted_baseline = format_baseline_metrics(baseline_metrics)

    # Print summary
    if baseline_metrics.get("metrics_available"):
        score = baseline_metrics.get("overall_baseline_score")
        level = baseline_metrics.get("overall_baseline_level")
        count = baseline_metrics.get("metrics_count")
        print(f"✅ Baseline Risk: {level} ({score:.1f}/100) from {count} metrics")
    else:
        print("⚠️ Baseline metrics not available")

    return {
        "ticker": ticker,
        "investment_memo": investment_memo,
        "baseline_metrics": formatted_baseline,
        "baseline_metrics_raw": baseline_metrics,
        "timestamp": timestamp,
    }


risk_chain = (
    RunnablePassthrough.assign(prepared_input=prepare_risk_input)
    | RunnablePassthrough.assign(
        ticker=lambda x: x["prepared_input"]["ticker"],
        investment_memo=lambda x: x["prepared_input"]["investment_memo"],
        baseline_metrics=lambda x: x["prepared_input"]["baseline_metrics"],
        timestamp=lambda x: x["prepared_input"]["timestamp"],
    )
    | risk_assessment_prompt
    | llm
    | JsonOutputParser()
)

# --- REPORT GENERATION (keeping existing structure, adding baseline info) ---


def get_risk_emoji(risk_level: str) -> str:
    """Get emoji for risk level visualization."""
    emoji_map = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🟠",
        "Critical": "🔴",
        "Unknown": "⚪",
    }
    return emoji_map.get(risk_level, "⚪")


def generate_risk_report(
    ticker: str, assessment: Dict[str, Any], baseline_metrics: Dict[str, Any]
) -> str:
    """Generate comprehensive risk report with baseline comparison."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    overall_score = assessment.get("overall_risk_score", 0)
    overall_level = assessment.get("overall_risk_level", "Unknown")
    overall_emoji = get_risk_emoji(overall_level)

    baseline_score = baseline_metrics.get("overall_baseline_score")
    baseline_level = baseline_metrics.get("overall_baseline_level")

    report = f"""
# Risk Assessment Report: ${ticker}
**Prepared by:** Risk Assessment Agent (Enhanced with Baseline Metrics)
**Date:** {timestamp}
**Assessment Confidence:** {assessment.get('assessment_confidence', 'unknown').capitalize()}

---

## Overall Risk Profile

{overall_emoji} **Final Risk Level:** {overall_level}  
**Final Risk Score:** {overall_score}/100

"""

    if baseline_score:
        baseline_emoji = get_risk_emoji(baseline_level)
        report += f"{baseline_emoji} **Baseline Risk Level:** {baseline_level}  \n"
        report += f"**Baseline Risk Score:** {baseline_score:.1f}/100  \n"
        report += (
            f"**Score Adjustment:** {overall_score - baseline_score:+.1f} points\n\n"
        )

    report += f"### Executive Assessment\n{assessment.get('overall_assessment', 'No assessment available.')}\n\n"

    if assessment.get("baseline_metrics_used"):
        report += f"**Baseline Integration:**  \n{assessment.get('baseline_vs_qualitative_note', 'Baseline metrics were considered.')}\n"

    report += "\n---\n\n"

    # Show baseline metrics summary
    if baseline_metrics.get("metrics_available"):
        report += "## Quantitative Baseline Metrics\n\n"
        individual = baseline_metrics.get("individual_metrics", {})

        for metric_name, metric_data in individual.items():
            if metric_data.get("risk_score") is not None:
                level_emoji = get_risk_emoji(metric_data["risk_level"])
                report += f"**{metric_name.replace('_', ' ').title()}:** {level_emoji} {metric_data['risk_level']} ({metric_data['risk_score']:.1f}/100)  \n"
                report += f"*{metric_data['description']}*\n\n"

        report += "---\n\n"

    report += "## Detailed Risk Analysis\n\n"

    categories = assessment.get("risk_categories", [])
    for category in categories:
        cat_name = category.get("category", "Unknown")
        risk_score = category.get("risk_score")
        risk_level = category.get("risk_level", "Unknown")
        risk_emoji = get_risk_emoji(risk_level)
        baseline_score_cat = category.get("baseline_score")

        report += f"### {risk_emoji} {cat_name}\n\n"
        report += f"**Risk Score:** {risk_score}/100  \n"
        report += f"**Risk Level:** {risk_level}  \n"

        if baseline_score_cat:
            report += f"**Baseline Score:** {baseline_score_cat}  \n"
            report += f"**Adjustment:** {category.get('adjustment_reason', 'No adjustment')}  \n"

        report += (
            f"**Confidence:** {category.get('confidence', 'unknown').capitalize()}  \n"
        )
        report += f"**Trend:** {category.get('trend', 'unknown').capitalize()}  \n\n"

        report += f"{category.get('analysis', 'No analysis available.')}\n\n"

        factors = category.get("key_risk_factors", [])
        if factors:
            report += "**Key Risk Factors:**\n\n"
            for factor in factors:
                sev_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    factor.get("severity"), "⚪"
                )
                report += f"{sev_emoji} **{factor.get('factor')}**  \n"
                report += f"   *Evidence:* {factor.get('evidence')}  \n"
                report += f"   *Source:* {factor.get('source_agent')}  \n\n"

        mitigations = category.get("mitigations", [])
        if mitigations:
            report += "**Risk Mitigation:**\n\n"
            for mitigation in mitigations:
                eff_emoji = {"high": "✅", "medium": "⚠️", "low": "❌"}.get(
                    mitigation.get("effectiveness"), "⚪"
                )
                report += f"{eff_emoji} **{mitigation.get('strategy')}**  \n"
                report += f"   *Effectiveness:* {mitigation.get('effectiveness').capitalize()}  \n"
                report += (
                    f"   *Implementation:* {mitigation.get('implementation')}  \n\n"
                )

        report += "---\n\n"

    # Rest of report (keeping existing structure)
    report += "## 🚨 Red Flags\n\n"

    red_flags = assessment.get("red_flags", [])
    if red_flags:
        for flag in red_flags:
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                flag.get("severity"), "⚪"
            )
            report += f"{severity_emoji} **{flag.get('flag')}**  \n"
            report += f"*Evidence:* {flag.get('source_evidence')}  \n"
            report += f"*Action:* {flag.get('immediate_action')}  \n\n"
    else:
        report += "*No critical red flags identified.*\n\n"

    report += "---\n\n## Portfolio Implications\n\n"

    portfolio = assessment.get("portfolio_implications", {})
    report += f"**Position Size:** {portfolio.get('recommended_position_size')}  \n"
    report += f"**Holding Period:** {portfolio.get('holding_period')}  \n"
    report += f"**Diversification:** {portfolio.get('diversification_needs')}  \n\n"

    monitoring = portfolio.get("risk_monitoring", [])
    if monitoring:
        report += "**Monitor:**\n"
        for metric in monitoring:
            report += f"- {metric}\n"

    report += "\n---\n\n## Risk-Reward Analysis\n\n"

    risk_reward = assessment.get("risk_reward_ratio", {})
    report += f"**Upside:** {risk_reward.get('potential_upside')}  \n"
    report += f"**Downside:** {risk_reward.get('potential_downside')}  \n"
    report += f"**Assessment:** {risk_reward.get('ratio_assessment')}  \n"
    report += f"**Suitable For:** {risk_reward.get('favorable_for')}  \n\n"

    report += "---\n\n## Final Recommendation\n\n"

    final = assessment.get("final_recommendation", {})
    report += f"**Suitability:** {final.get('suitability')}  \n"
    report += f"**Monitoring:** {final.get('monitoring_frequency')}  \n\n"

    considerations = final.get("key_considerations", [])
    if considerations:
        report += "**Key Considerations:**\n"
        for consideration in considerations:
            report += f"- {consideration}\n"

    report += f"""

---

## Methodology

This risk assessment combines:
1. **Quantitative Baseline Metrics**: Calculated from financial data
   - Volatility, Beta, Debt/Equity, Liquidity, Valuation, Price Momentum
2. **Qualitative Analysis**: From Governor Agent's memo
3. **LLM Synthesis**: Balancing quantitative and qualitative factors

**Baseline Metrics Available:** {baseline_metrics.get('metrics_count', 0)}/6  
**Assessment Confidence:** {assessment.get('assessment_confidence', 'unknown').upper()}

---

**Disclaimer:** This combines AI analysis with calculated financial metrics.
Not financial advice. Consult a qualified advisor.

**Timestamp:** {timestamp}  
**Ticker:** ${ticker}
"""

    return report.strip()


def save_risk_report(ticker: str, report: str, output_dir: str = "reports") -> str:
    """Save risk report to file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_risk_assessment_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 Risk assessment saved to: {filepath}")
    return filepath


# --- MAIN FUNCTION ---


def run_risk_assessment_agent(
    ticker: str, investment_memo: str, save_to_file: bool = True
) -> tuple[str, str]:
    """Execute Risk Assessment with baseline metrics."""
    print(f"\n{'='*60}")
    print(f"🤖 Risk Assessment Agent (Enhanced): Analyzing ${ticker}")
    print(f"{'='*60}\n")

    if not investment_memo:
        error_msg = "No investment memo provided"
        print(f"❌ {error_msg}")
        return error_msg, error_msg

    try:
        # Prepare input (calculates baseline metrics)
        prepared = prepare_risk_input(
            {"ticker": ticker.upper(), "investment_memo": investment_memo}
        )

        baseline_metrics_raw = prepared["baseline_metrics_raw"]

        # Execute chain
        assessment_json = risk_chain.invoke(
            {"ticker": ticker.upper(), "investment_memo": investment_memo}
        )

        # Generate detailed report
        detailed_report = generate_risk_report(
            ticker, assessment_json, baseline_metrics_raw
        )

        # Save to file
        if save_to_file:
            save_risk_report(ticker, detailed_report)

        # Create summary
        overall_score = assessment_json.get("overall_risk_score", 0)
        overall_level = assessment_json.get("overall_risk_level", "Unknown")
        overall_emoji = get_risk_emoji(overall_level)

        baseline_score = baseline_metrics_raw.get("overall_baseline_score")
        baseline_level = baseline_metrics_raw.get("overall_baseline_level")

        # Format baseline info safely
        if baseline_score is not None:
            baseline_info = f"{baseline_level} ({baseline_score:.1f}/100)"
            adjustment_info = f"{overall_score - baseline_score:+.1f} points"
        else:
            baseline_info = "N/A"
            adjustment_info = "Qualitative only"

        summary_report = f"""
**Risk Assessment Agent Summary: ${ticker}**

{overall_emoji} **Overall Risk:** {overall_level} ({overall_score}/100)

📊 **Baseline Comparison:**
* Baseline Risk: {baseline_info}
* Adjustment: {adjustment_info}
* Metrics Used: {baseline_metrics_raw.get('metrics_count', 0)}/6

📋 **Executive Assessment:**
{assessment_json.get('overall_assessment', 'No assessment available.')}

💡 **Baseline Integration:**
{assessment_json.get('baseline_vs_qualitative_note', 'See full report for details.')}

✅ **Enhanced with Quantitative Metrics**
This assessment combines calculated financial metrics (volatility, leverage,
liquidity, etc.) with qualitative analysis from the Governor's memo.

---
*Full risk assessment with baseline metrics saved to file*
*Agent: Risk Assessment Agent (Enhanced) | Model: Gemini Pro*
        """

        print("✅ Risk_Assessment_Agent: Analysis complete")

        # Return comprehensive dictionary like news_agent
        return {
            "ticker": ticker,
            "agent": "Risk_Assessment",
            "overall_risk_score": overall_score,
            "overall_risk_level": overall_level,
            "baseline_score": baseline_score,
            "baseline_level": baseline_level,
            "baseline_metrics": baseline_metrics_raw,
            "risk_categories": {
                "market_risk": assessment_json.get("market_risk", {}),
                "company_specific_risk": assessment_json.get(
                    "company_specific_risk", {}
                ),
                "sentiment_risk": assessment_json.get("sentiment_risk", {}),
                "technical_risk": assessment_json.get("technical_risk", {}),
                "regulatory_risk": assessment_json.get("regulatory_risk", {}),
                "macroeconomic_risk": assessment_json.get("macroeconomic_risk", {}),
            },
            "overall_assessment": assessment_json.get("overall_assessment", ""),
            "key_risk_factors": assessment_json.get("key_risk_factors", []),
            "risk_mitigation_suggestions": assessment_json.get(
                "risk_mitigation_suggestions", []
            ),
            "baseline_vs_qualitative_note": assessment_json.get(
                "baseline_vs_qualitative_note", ""
            ),
            "summary_report": summary_report.strip(),
            "detailed_report": detailed_report,
        }

    except Exception as e:
        print(f"❌ Risk_Assessment_Agent: Assessment failed - {e}")
        import traceback

        traceback.print_exc()

        error_report = f"""
**Risk Assessment Agent Report: ${ticker}**

⚠️ **Error:** Risk assessment could not be completed.

**Details:** {str(e)}
        """
        return {
            "ticker": ticker,
            "agent": "Risk_Assessment",
            "error": str(e),
            "summary_report": error_report.strip(),
            "detailed_report": error_report.strip(),
        }


# --- CLI ENTRY POINT ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "TSLA"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python risk_assessment_agent.py <TICKER>\n")

    # Mock investment memo
    mock_memo = """
# Investment Memo: $TSLA

## Executive Summary
Tesla shows strong technical momentum and positive analyst sentiment, but faces
significant regulatory challenges and high valuation concerns.

## Key Risks
- Regulatory challenges in Europe and China
- Competition intensifying
- Valuation stretched relative to peers
"""

    result = run_risk_assessment_agent(ticker, mock_memo, save_to_file=True)

    print("\n" + "=" * 60)
    print("RISK ASSESSMENT SUMMARY")
    print("=" * 60 + "\n")
    print(result.get("summary_report", ""))
    print("\n" + "=" * 60)
    print("\n📄 Full risk assessment saved to 'reports/' directory")
