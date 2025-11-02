"""
Risk Assessment Agent - Analyst Swarm Component

This agent takes the Governor Agent's synthesized investment memo and produces
a comprehensive risk assessment with quantitative scoring across multiple dimensions.

It evaluates:
- Market Risk
- Company-Specific Risk
- Sentiment Risk
- Technical Risk
- Regulatory/Legal Risk
- Macroeconomic Risk

Environment Variables Required:
    - GEMINI_API_KEY: Your Google Gemini API key
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- ENVIRONMENT VALIDATION ---

def validate_environment() -> None:
    """Validate all required API keys are set."""
    required_vars = {
        "GEMINI_API_KEY": "Gemini LLM for risk analysis"
    }
    
    missing = []
    for var_name, description in required_vars.items():
        if var_name not in os.environ:
            missing.append(f"  ❌ {var_name}: {description}")
    
    if missing:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing)
        raise EnvironmentError(error_msg)
    
    print("✅ Risk_Assessment_Agent: All environment variables validated")

# Validate at module load time
validate_environment()

# --- SETUP MODEL ---

# Initialize Gemini LLM with low temperature for consistent risk assessment
llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest",
    temperature=0.2,  # Low temperature for objective risk scoring
    api_key=os.environ["GEMINI_API_KEY"]
)

# --- PROMPT TEMPLATE ---

risk_assessment_prompt = ChatPromptTemplate.from_template(
    """You are a "Risk Assessment Specialist" - your role is to evaluate the investment 
risks for ${ticker} based on the comprehensive investment memo prepared by the Governor Agent.

**YOUR MANDATE:**
Analyze the Governor's memo and produce a detailed, quantitative risk assessment across 
six key dimensions. For each risk category, you must:
1. Provide a risk score (0-100, where 100 = maximum risk)
2. Assign a risk level (Low / Medium / High / Critical)
3. Identify specific risk factors from the memo
4. Provide actionable mitigation strategies

**RISK CATEGORIES TO ASSESS:**

1. **Market Risk** (Industry, competitive landscape, market conditions)
2. **Company-Specific Risk** (Financial health, management, operations)
3. **Sentiment Risk** (Investor perception, social media, analyst views)
4. **Technical Risk** (Price action, chart patterns, momentum)
5. **Regulatory/Legal Risk** (Compliance, litigation, regulatory changes)
6. **Macroeconomic Risk** (Interest rates, inflation, economic cycles)

**SCORING GUIDELINES:**
- **0-25:** Low Risk - Well-managed, minimal concerns
- **26-50:** Medium Risk - Some concerns, manageable with caution
- **51-75:** High Risk - Significant concerns, careful consideration needed
- **76-100:** Critical Risk - Severe concerns, potential red flags

**CRITICAL INSTRUCTIONS:**
- Base your assessment ONLY on evidence from the Governor's memo
- When the memo lacks information on a risk category, score it as "Unknown" (score: null)
- Be specific - cite which agent's findings support each risk factor
- Consider both consensus views AND conflicting perspectives
- Your overall risk score should be a weighted average, not a simple mean

**GOVERNOR'S INVESTMENT MEMO:**

<investment_memo>
{investment_memo}
</investment_memo>

Return your risk assessment in this EXACT JSON format:
{{
  "overall_risk_score": 0-100,
  "overall_risk_level": "Low" | "Medium" | "High" | "Critical",
  "overall_assessment": "2-3 sentence executive summary of the risk profile",
  "risk_categories": [
    {{
      "category": "Market Risk",
      "risk_score": 0-100 or null,
      "risk_level": "Low" | "Medium" | "High" | "Critical" | "Unknown",
      "confidence": "high" | "medium" | "low",
      "key_risk_factors": [
        {{
          "factor": "Specific risk factor",
          "evidence": "Quote or reference from memo",
          "source_agent": "Which agent identified this",
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
      "analysis": "2-3 sentence detailed analysis of this risk category"
    }}
  ],
  "red_flags": [
    {{
      "flag": "Critical concern description",
      "severity": "high" | "medium" | "low",
      "source_evidence": "Evidence from memo",
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
    "diversification_needs": "Description of diversification strategy",
    "risk_monitoring": ["Key metric to monitor", "Key metric to monitor"]
  }},
  "stress_test_scenarios": [
    {{
      "scenario": "What if scenario description",
      "probability": "high" | "medium" | "low",
      "impact_on_investment": "Description of impact",
      "preparedness": "How well positioned investor would be"
    }}
  ],
  "comparable_risk_analysis": {{
    "industry_comparison": "How this stock's risk compares to industry peers",
    "historical_comparison": "How current risk compares to historical levels",
    "market_comparison": "How this stock's risk compares to broader market"
  }},
  "risk_reward_ratio": {{
    "potential_upside": "percentage or description",
    "potential_downside": "percentage or description",
    "ratio_assessment": "Assessment of whether risk is justified by reward",
    "favorable_for": "Conservative investors" | "Moderate investors" | "Aggressive investors" | "None"
  }},
  "final_recommendation": {{
    "suitability": "Which investor types this is suitable for",
    "key_considerations": ["consideration 1", "consideration 2", "consideration 3"],
    "monitoring_frequency": "How often to review this investment",
    "exit_triggers": ["Event that should trigger exit consideration"]
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

Return ONLY valid JSON. Do not include any text before or after the JSON object."""
)

# --- CREATE RISK ASSESSMENT CHAIN ---

def prepare_risk_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare input for the Risk Assessment Agent chain.
    
    Args:
        input_dict: Contains 'ticker' and 'investment_memo' keys
    
    Returns:
        Dictionary with formatted inputs for the prompt
    """
    ticker = input_dict.get("ticker", "").upper()
    investment_memo = input_dict.get("investment_memo", "")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "ticker": ticker,
        "investment_memo": investment_memo,
        "timestamp": timestamp
    }

risk_chain = (
    RunnablePassthrough.assign(prepared_input=prepare_risk_input)
    | RunnablePassthrough.assign(
        ticker=lambda x: x["prepared_input"]["ticker"],
        investment_memo=lambda x: x["prepared_input"]["investment_memo"],
        timestamp=lambda x: x["prepared_input"]["timestamp"]
    )
    | risk_assessment_prompt
    | llm
    | JsonOutputParser()
)

# --- REPORT GENERATION ---

def get_risk_emoji(risk_level: str) -> str:
    """Get emoji for risk level visualization."""
    emoji_map = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🟠",
        "Critical": "🔴",
        "Unknown": "⚪"
    }
    return emoji_map.get(risk_level, "⚪")

def generate_risk_report(ticker: str, assessment: Dict[str, Any]) -> str:
    """
    Generate a comprehensive risk assessment report.
    
    Args:
        ticker: Stock ticker symbol
        assessment: Risk assessment JSON from agent
    
    Returns:
        Formatted markdown risk report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    overall_score = assessment.get('overall_risk_score', 0)
    overall_level = assessment.get('overall_risk_level', 'Unknown')
    overall_emoji = get_risk_emoji(overall_level)
    
    report = f"""
# Risk Assessment Report: ${ticker}
**Prepared by:** Risk Assessment Agent
**Date:** {timestamp}
**Assessment Confidence:** {assessment.get('assessment_confidence', 'unknown').capitalize()}

---

## Overall Risk Profile

{overall_emoji} **Risk Level:** {overall_level}  
**Risk Score:** {overall_score}/100

### Executive Assessment
{assessment.get('overall_assessment', 'No assessment available.')}

---

## Detailed Risk Analysis

"""
    
    # Risk categories breakdown
    categories = assessment.get('risk_categories', [])
    for category in categories:
        cat_name = category.get('category', 'Unknown Category')
        risk_score = category.get('risk_score')
        risk_level = category.get('risk_level', 'Unknown')
        risk_emoji = get_risk_emoji(risk_level)
        confidence = category.get('confidence', 'unknown')
        trend = category.get('trend', 'unknown')
        
        # Trend emoji
        trend_emoji = {
            'improving': '📈', 
            'stable': '➡️', 
            'deteriorating': '📉', 
            'unknown': '❓'
        }.get(trend, '❓')
        
        report += f"""
### {risk_emoji} {cat_name}

**Risk Score:** {risk_score if risk_score is not None else 'N/A'}/100  
**Risk Level:** {risk_level}  
**Confidence:** {confidence.capitalize()}  
**Trend:** {trend_emoji} {trend.capitalize()}

{category.get('analysis', 'No analysis available.')}

"""
        
        # Key risk factors
        factors = category.get('key_risk_factors', [])
        if factors:
            report += "**Key Risk Factors:**\n\n"
            for factor in factors:
                severity_emoji = {
                    'high': '🔴', 'medium': '🟡', 'low': '🟢'
                }.get(factor.get('severity', 'low'), '⚪')
                
                report += f"{severity_emoji} **{factor.get('factor', 'Unknown factor')}**\n"
                report += f"   - *Evidence:* {factor.get('evidence', 'No evidence')}\n"
                report += f"   - *Source:* {factor.get('source_agent', 'Unknown')}\n\n"
        
        # Mitigations
        mitigations = category.get('mitigations', [])
        if mitigations:
            report += "**Risk Mitigation Strategies:**\n\n"
            for mitigation in mitigations:
                effectiveness = mitigation.get('effectiveness', 'unknown')
                eff_emoji = {
                    'high': '✅', 'medium': '⚠️', 'low': '❌'
                }.get(effectiveness, '⚪')
                
                report += f"{eff_emoji} **{mitigation.get('strategy', 'Unknown strategy')}**\n"
                report += f"   - *Effectiveness:* {effectiveness.capitalize()}\n"
                report += f"   - *Implementation:* {mitigation.get('implementation', 'No details')}\n\n"
        
        report += "---\n\n"
    
    # Red Flags section
    report += "## 🚨 Red Flags & Critical Concerns\n\n"
    
    red_flags = assessment.get('red_flags', [])
    if red_flags:
        for flag in red_flags:
            severity = flag.get('severity', 'unknown')
            severity_emoji = {
                'high': '🔴', 'medium': '🟡', 'low': '🟢'
            }.get(severity, '⚪')
            
            report += f"{severity_emoji} **{flag.get('flag', 'Unknown concern')}**\n\n"
            report += f"*Evidence:* {flag.get('source_evidence', 'No evidence')}\n\n"
            report += f"*Immediate Action:* {flag.get('immediate_action', 'No action specified')}\n\n"
    else:
        report += "*No critical red flags identified.*\n\n"
    
    report += "---\n\n"
    
    # Risk Opportunities
    report += "## 💡 Risk Opportunities\n\n"
    report += "Risks that could become opportunities under certain conditions:\n\n"
    
    risk_opps = assessment.get('risk_opportunities', [])
    if risk_opps:
        for opp in risk_opps:
            report += f"**{opp.get('opportunity', 'Unknown opportunity')}**\n\n"
            report += f"*Scenario:* {opp.get('scenario', 'No scenario')}\n\n"
            report += f"*Potential Upside:* {opp.get('potential_upside', 'Unknown')}\n\n"
    else:
        report += "*No risk opportunities identified.*\n\n"
    
    report += "---\n\n"
    
    # Portfolio Implications
    report += "## 📊 Portfolio Implications\n\n"
    
    portfolio = assessment.get('portfolio_implications', {})
    
    report += f"**Recommended Position Size:** {portfolio.get('recommended_position_size', 'Unknown')}\n\n"
    report += f"**Suggested Holding Period:** {portfolio.get('holding_period', 'Unknown')}\n\n"
    report += f"**Diversification Strategy:** {portfolio.get('diversification_needs', 'No strategy provided')}\n\n"
    
    monitoring = portfolio.get('risk_monitoring', [])
    if monitoring:
        report += "**Key Metrics to Monitor:**\n\n"
        for metric in monitoring:
            report += f"- {metric}\n"
        report += "\n"
    
    report += "---\n\n"
    
    # Stress Test Scenarios
    report += "## 🧪 Stress Test Scenarios\n\n"
    
    stress_tests = assessment.get('stress_test_scenarios', [])
    if stress_tests:
        for scenario in stress_tests:
            prob = scenario.get('probability', 'unknown')
            prob_emoji = {
                'high': '🔴', 'medium': '🟡', 'low': '🟢'
            }.get(prob, '⚪')
            
            report += f"{prob_emoji} **{scenario.get('scenario', 'Unknown scenario')}**\n\n"
            report += f"*Probability:* {prob.capitalize()}\n\n"
            report += f"*Impact:* {scenario.get('impact_on_investment', 'Unknown')}\n\n"
            report += f"*Preparedness:* {scenario.get('preparedness', 'Unknown')}\n\n"
    else:
        report += "*No stress test scenarios provided.*\n\n"
    
    report += "---\n\n"
    
    # Comparative Risk Analysis
    report += "## 📈 Comparative Risk Analysis\n\n"
    
    comparison = assessment.get('comparable_risk_analysis', {})
    
    report += f"**Industry Comparison:**\n{comparison.get('industry_comparison', 'No comparison available')}\n\n"
    report += f"**Historical Comparison:**\n{comparison.get('historical_comparison', 'No comparison available')}\n\n"
    report += f"**Market Comparison:**\n{comparison.get('market_comparison', 'No comparison available')}\n\n"
    
    report += "---\n\n"
    
    # Risk-Reward Ratio
    report += "## ⚖️ Risk-Reward Analysis\n\n"
    
    risk_reward = assessment.get('risk_reward_ratio', {})
    
    report += f"**Potential Upside:** {risk_reward.get('potential_upside', 'Unknown')}\n\n"
    report += f"**Potential Downside:** {risk_reward.get('potential_downside', 'Unknown')}\n\n"
    report += f"**Assessment:** {risk_reward.get('ratio_assessment', 'No assessment')}\n\n"
    report += f"**Favorable For:** {risk_reward.get('favorable_for', 'Unknown')}\n\n"
    
    report += "---\n\n"
    
    # Final Recommendation
    report += "## 🎯 Final Recommendation\n\n"
    
    final = assessment.get('final_recommendation', {})
    
    report += f"**Suitability:** {final.get('suitability', 'Unknown')}\n\n"
    
    considerations = final.get('key_considerations', [])
    if considerations:
        report += "**Key Considerations:**\n\n"
        for consideration in considerations:
            report += f"- {consideration}\n"
        report += "\n"
    
    report += f"**Monitoring Frequency:** {final.get('monitoring_frequency', 'Unknown')}\n\n"
    
    triggers = final.get('exit_triggers', [])
    if triggers:
        report += "**Exit Triggers (Consider Selling If):**\n\n"
        for trigger in triggers:
            report += f"- {trigger}\n"
        report += "\n"
    
    report += "---\n\n"
    
    # Data Limitations
    limitations = assessment.get('data_limitations', [])
    if limitations:
        report += "## ⚠️ Data Limitations\n\n"
        for limitation in limitations:
            report += f"- {limitation}\n"
        report += "\n---\n\n"
    
    # Disclosure
    report += f"""
## Disclosure

This risk assessment was generated by an AI-powered Risk Assessment Agent based on 
the investment memo prepared by the Governor Agent, which synthesized insights from 
multiple specialist agents.

**Important:**
- This is NOT financial advice
- Risk assessments are based on available data and may change
- Past performance does not guarantee future results
- Consult a qualified financial advisor before making investment decisions

**Timestamp:** {timestamp}  
**Ticker:** ${ticker}  
**Risk Assessment Agent Model:** Gemini Pro Latest
"""
    
    return report.strip()

def save_risk_report(ticker: str, report: str, output_dir: str = "reports") -> str:
    """
    Save the risk assessment report to a file.
    
    Args:
        ticker: Stock ticker symbol
        report: Formatted report content
        output_dir: Directory to save reports
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_risk_assessment_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Risk assessment saved to: {filepath}")
    return filepath

# --- MAIN RISK ASSESSMENT FUNCTION ---

def run_risk_assessment_agent(
    ticker: str,
    investment_memo: str,
    save_to_file: bool = True
) -> tuple[str, str]:
    """
    Execute the Risk Assessment Agent analysis.
    
    Args:
        ticker: Stock ticker symbol
        investment_memo: The Governor Agent's investment memo
        save_to_file: Whether to save the report to a file
    
    Returns:
        Tuple of (summary_report, detailed_report)
    """
    print(f"\n{'='*60}")
    print(f"🤖 Risk Assessment Agent: Analyzing risk for ${ticker}")
    print(f"{'='*60}\n")
    
    if not investment_memo:
        error_msg = "No investment memo provided to Risk Assessment Agent"
        print(f"❌ {error_msg}")
        return error_msg, error_msg
    
    try:
        # Execute the risk assessment chain
        assessment_json = risk_chain.invoke({
            "ticker": ticker.upper(),
            "investment_memo": investment_memo
        })
        
        # Generate detailed risk report
        detailed_report = generate_risk_report(ticker, assessment_json)
        
        # Save to file if requested
        if save_to_file:
            save_risk_report(ticker, detailed_report)
        
        # Create concise summary for console
        overall_score = assessment_json.get('overall_risk_score', 0)
        overall_level = assessment_json.get('overall_risk_level', 'Unknown')
        overall_emoji = get_risk_emoji(overall_level)
        overall_assessment = assessment_json.get('overall_assessment', 'No assessment available.')
        
        red_flag_count = len(assessment_json.get('red_flags', []))
        
        portfolio_rec = assessment_json.get('portfolio_implications', {}).get('recommended_position_size', 'Unknown')
        
        summary_report = f"""
**Risk Assessment Agent Summary: ${ticker}**

{overall_emoji} **Overall Risk:** {overall_level} ({overall_score}/100)

📋 **Executive Assessment:**
{overall_assessment}

⚠️ **Critical Concerns:** {red_flag_count} red flag(s) identified

📊 **Portfolio Recommendation:**
* **Position Size:** {portfolio_rec}
* **Assessment Confidence:** {assessment_json.get('assessment_confidence', 'unknown').capitalize()}

---
*Full risk assessment report saved to file*
*Agent: Risk Assessment Agent | Model: Gemini Pro*
        """
        
        print("✅ Risk_Assessment_Agent: Analysis complete")
        return summary_report.strip(), detailed_report
        
    except Exception as e:
        print(f"❌ Risk_Assessment_Agent: Assessment failed - {e}")
        import traceback
        traceback.print_exc()
        
        error_report = f"""
**Risk Assessment Agent Report: ${ticker}**

⚠️ **Error:** Risk assessment could not be completed.

**Details:** {str(e)}
        """
        return error_report.strip(), error_report.strip()

# --- CLI ENTRY POINT ---

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "TSLA"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python risk_assessment_agent.py <TICKER>\n")
    
    # Mock investment memo for testing
    mock_memo = """
# Investment Memo: $TSLA

## Executive Summary
Tesla shows strong technical momentum and positive analyst sentiment, but faces
significant regulatory challenges and high valuation concerns. Social sentiment
is bullish but volatile.

## Critical Insights
1. Strong AI and FSD progress noted by multiple analysts
2. Valuation concerns at current levels
3. Regulatory uncertainty in key markets

## Consensus Views
- Long-term AI opportunity is significant
- Near-term execution risk exists

## Key Risks
- Regulatory challenges in Europe and China
- Competition intensifying
- Valuation stretched relative to peers
"""
    
    # Run the Risk Assessment Agent
    summary, detailed = run_risk_assessment_agent(ticker, mock_memo, save_to_file=True)
    
    # Display summary
    print("\n" + "="*60)
    print("RISK ASSESSMENT SUMMARY")
    print("="*60 + "\n")
    print(summary)
    print("\n" + "="*60)
    print("\n📄 Full risk assessment saved to 'reports/' directory")