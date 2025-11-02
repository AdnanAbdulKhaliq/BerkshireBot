"""
Governor Agent - Analyst Swarm Component (Enhanced with Report Validation)

This agent receives reports from all specialist agents and synthesizes them
with proper validation and quality checks.

Environment Variables Required:
    - GEMINI_API_KEY: Your Google Gemini API key
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- ENVIRONMENT VALIDATION ---


def validate_environment() -> None:
    """Validate all required API keys are set."""
    required_vars = {"GEMINI_API_KEY": "Gemini LLM for synthesis"}

    missing = []
    for var_name, description in required_vars.items():
        if var_name not in os.environ:
            missing.append(f"  ❌ {var_name}: {description}")

    if missing:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing)
        raise EnvironmentError(error_msg)

    print("✅ Governor_Agent: All environment variables validated")


validate_environment()

# --- SETUP MODEL ---

llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest", temperature=0.3, api_key=os.environ["GEMINI_API_KEY"]
)

# --- AGENT REPORT VALIDATION ---


def validate_agent_report(agent_name: str, report: str) -> Dict[str, Any]:
    """
    Validate that an agent report contains useful information.

    Args:
        agent_name: Name of the agent (e.g., "Social Agent")
        report: The report text from the agent

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "agent_name": agent_name,
        "valid": True,
        "quality": "good",
        "issues": [],
        "word_count": 0,
        "has_data": False,
        "has_analysis": False,
        "error_found": False,
    }

    # Check if report exists
    if not report:
        validation_result["valid"] = False
        validation_result["quality"] = "missing"
        validation_result["issues"].append("Report is empty or None")
        return validation_result

    report_lower = report.lower()
    report_stripped = report.strip()

    # Check minimum length
    if len(report_stripped) < 50:
        validation_result["valid"] = False
        validation_result["quality"] = "insufficient"
        validation_result["issues"].append(
            f"Report too short ({len(report_stripped)} chars)"
        )
        return validation_result

    # Count words
    validation_result["word_count"] = len(report_stripped.split())

    # Check for error indicators
    error_keywords = [
        "error:",
        "failed",
        "could not",
        "unable to",
        "exception",
        "traceback",
        "not available",
        "missing required",
        "api key",
        "⚠️ error",
    ]

    for keyword in error_keywords:
        if keyword in report_lower:
            validation_result["error_found"] = True
            validation_result["issues"].append(f"Error keyword found: '{keyword}'")

    # Check for data presence (numbers, metrics)
    has_numbers = any(char.isdigit() for char in report)
    has_bullets = "•" in report or "*" in report or "-" in report
    has_sections = "**" in report or "#" in report

    validation_result["has_data"] = has_numbers or has_bullets
    validation_result["has_analysis"] = (
        has_sections or validation_result["word_count"] > 100
    )

    # Determine quality level
    if validation_result["error_found"]:
        validation_result["valid"] = False
        validation_result["quality"] = "failed"
    elif validation_result["word_count"] < 100:
        validation_result["quality"] = "limited"
        validation_result["issues"].append("Report lacks depth (< 100 words)")
    elif not validation_result["has_data"]:
        validation_result["quality"] = "limited"
        validation_result["issues"].append("Report lacks quantitative data")
    elif validation_result["word_count"] > 500 and validation_result["has_data"]:
        validation_result["quality"] = "excellent"
    else:
        validation_result["quality"] = "good"

    return validation_result


def validate_all_reports(agent_reports: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate all agent reports and return summary.

    Args:
        agent_reports: Dictionary mapping agent names to their reports

    Returns:
        Dictionary with overall validation summary
    """
    validations = {}

    for agent_name, report in agent_reports.items():
        validations[agent_name] = validate_agent_report(agent_name, report)

    # Calculate overall statistics
    total_agents = len(validations)
    valid_count = sum(1 for v in validations.values() if v["valid"])
    failed_count = sum(1 for v in validations.values() if v["quality"] == "failed")
    excellent_count = sum(
        1 for v in validations.values() if v["quality"] == "excellent"
    )

    overall_quality = "poor"
    if valid_count == total_agents and excellent_count >= total_agents // 2:
        overall_quality = "excellent"
    elif valid_count >= total_agents * 0.8:
        overall_quality = "good"
    elif valid_count >= total_agents * 0.6:
        overall_quality = "fair"

    return {
        "individual_validations": validations,
        "total_agents": total_agents,
        "valid_count": valid_count,
        "failed_count": failed_count,
        "excellent_count": excellent_count,
        "overall_quality": overall_quality,
        "validation_timestamp": datetime.now().isoformat(),
    }


def generate_validation_summary(validation_results: Dict[str, Any]) -> str:
    """
    Generate a human-readable validation summary.

    Args:
        validation_results: Output from validate_all_reports

    Returns:
        Formatted string summary
    """
    summary = "\n" + "=" * 60 + "\n"
    summary += "📋 AGENT REPORT VALIDATION SUMMARY\n"
    summary += "=" * 60 + "\n\n"

    individual = validation_results["individual_validations"]

    for agent_name, validation in individual.items():
        status_emoji = "✅" if validation["valid"] else "❌"
        quality_emoji = {
            "excellent": "🌟",
            "good": "✅",
            "limited": "⚠️",
            "insufficient": "❌",
            "failed": "🔴",
            "missing": "❓",
        }.get(validation["quality"], "❓")

        summary += f"{status_emoji} **{agent_name}**\n"
        summary += f"   Quality: {quality_emoji} {validation['quality'].capitalize()}\n"
        summary += f"   Words: {validation['word_count']}\n"

        if validation["issues"]:
            summary += f"   Issues: {len(validation['issues'])}\n"
            for issue in validation["issues"][:2]:  # Show max 2 issues
                summary += f"      - {issue}\n"

        summary += "\n"

    summary += "-" * 60 + "\n"
    summary += f"📊 Overall Statistics:\n"
    summary += f"   Total Agents: {validation_results['total_agents']}\n"
    summary += f"   Valid Reports: {validation_results['valid_count']}/{validation_results['total_agents']}\n"
    summary += f"   Failed Reports: {validation_results['failed_count']}\n"
    summary += f"   Excellent Reports: {validation_results['excellent_count']}\n"
    summary += f"   Overall Quality: {validation_results['overall_quality'].upper()}\n"
    summary += "=" * 60 + "\n"

    return summary


def filter_valid_reports(
    agent_reports: Dict[str, str], validation_results: Dict[str, Any]
) -> Dict[str, str]:
    """
    Filter out invalid reports and add quality warnings to limited reports.

    Args:
        agent_reports: Original agent reports
        validation_results: Validation results from validate_all_reports

    Returns:
        Filtered dictionary with valid reports and quality annotations
    """
    filtered = {}
    individual = validation_results["individual_validations"]

    for agent_name, report in agent_reports.items():
        validation = individual.get(agent_name, {})

        if not validation.get("valid", False):
            # Replace failed report with error notice
            filtered[
                agent_name
            ] = f"""
**{agent_name} Report - VALIDATION FAILED**

⚠️ This report did not pass validation checks.

**Issues Identified:**
{chr(10).join(f"- {issue}" for issue in validation.get('issues', ['Unknown issue']))}

**Original Report Preview:**
{report[:200]}...

*The Governor Agent will synthesize without this report's data.*
"""
        elif validation.get("quality") == "limited":
            # Add quality warning but keep report
            filtered[
                agent_name
            ] = f"""
**{agent_name} Report - LIMITED DATA WARNING**

⚠️ This report passed validation but has limited data or depth.

{report}
"""
        else:
            # Valid and good quality - use as is
            filtered[agent_name] = report

    return filtered


# --- AGENT REPORT FORMATTING ---


def format_agent_reports(
    agent_reports: Dict[str, str], validation_results: Dict[str, Any]
) -> str:
    """
    Format all agent reports with validation annotations.

    Args:
        agent_reports: Dictionary mapping agent names to their report strings
        validation_results: Validation results

    Returns:
        Formatted string with all agent reports
    """
    formatted = ""

    agent_order = [
        "SEC Agent",
        "News Agent",
        "Social Agent",
        "Chart Agent",
        "Analyst Agent",
    ]
    individual = validation_results["individual_validations"]

    for agent_name in agent_order:
        report = agent_reports.get(agent_name, f"[{agent_name} report not available]")
        validation = individual.get(agent_name, {})

        # Add validation header
        quality_badge = f"[QUALITY: {validation.get('quality', 'unknown').upper()}]"
        valid_badge = "[VALID ✅]" if validation.get("valid", False) else "[INVALID ❌]"

        formatted += f"\n{'='*70}\n"
        formatted += f"{agent_name.upper()} REPORT {valid_badge} {quality_badge}\n"
        formatted += f"{'='*70}\n\n"
        formatted += report
        formatted += "\n\n"

    return formatted


# --- PROMPT TEMPLATE WITH VALIDATION AWARENESS ---

governor_prompt_template = ChatPromptTemplate.from_template(
    """You are the "Governor Agent" - the senior analyst who synthesizes insights from 
a team of specialist analysts into a single, balanced investment memo for ${ticker}.

**REPORT VALIDATION RESULTS:**

The following reports have been pre-validated:
- Valid reports: {valid_count}/{total_agents}
- Failed reports: {failed_count}
- Overall data quality: {overall_quality}

**IMPORTANT:** Some reports may have failed validation or contain limited data.
You MUST acknowledge data quality issues in your synthesis and adjust confidence accordingly.

**HOLISTIC AI PRINCIPLES:**
1. **Unbiased Analysis**: Present all perspectives fairly
2. **Source Attribution**: Cite which agent provided each insight
3. **Conflicting Views**: Explicitly highlight disagreements
4. **Consensus Building**: Identify areas of agreement
5. **Transparency**: Acknowledge gaps and data quality issues

**YOUR SPECIALIST TEAM REPORTS:**

{agent_reports}

**YOUR TASK:**
Synthesize these reports into a cohesive investment memo that:
1. Acknowledges which reports are valid vs. limited vs. failed
2. Weights insights based on data quality
3. Identifies the 3-5 most critical insights from VALID reports
4. Highlights consensus views (where 2+ valid agents agree)
5. Presents conflicting perspectives fairly
6. Provides a balanced conclusion reflecting data quality limitations

**CRITICAL RULES:**
- Only cite insights from VALID reports
- Explicitly note when key agents are missing or failed
- Lower confidence when data quality is poor
- Never fabricate data from failed reports

Return your synthesis in this EXACT JSON format:
{{
  "executive_summary": "3-4 sentence overview acknowledging data quality",
  "data_quality_acknowledgment": "Explicit statement about which agents provided valid data",
  "critical_insights": [
    {{
      "insight": "The key finding",
      "source_agents": ["Agent names - ONLY from valid reports"],
      "supporting_evidence": "Brief evidence summary",
      "confidence": "high" | "medium" | "low",
      "data_quality_note": "Any relevant quality concerns"
    }}
  ],
  "consensus_views": [
    {{
      "view": "What multiple VALID agents agree on",
      "supporting_agents": ["Only valid agents"],
      "strength": "strong" | "moderate" | "weak"
    }}
  ],
  "conflicting_perspectives": [
    {{
      "topic": "What agents disagree about",
      "perspective_a": {{
        "view": "First perspective",
        "source_agent": "Agent name",
        "evidence": "Supporting evidence"
      }},
      "perspective_b": {{
        "view": "Opposing perspective",
        "source_agent": "Agent name",
        "evidence": "Supporting evidence"
      }},
      "reconciliation": "How to interpret this conflict"
    }}
  ],
  "data_quality_assessment": {{
    "overall_quality": "excellent" | "good" | "fair" | "poor",
    "gaps_identified": ["gap 1", "gap 2"],
    "agent_coverage": {{
      "sec_agent": "excellent" | "good" | "limited" | "failed" | "missing",
      "news_agent": "excellent" | "good" | "limited" | "failed" | "missing",
      "social_agent": "excellent" | "good" | "limited" | "failed" | "missing",
      "chart_agent": "excellent" | "good" | "limited" | "failed" | "missing",
      "analyst_agent": "excellent" | "good" | "limited" | "failed" | "missing"
    }},
    "validation_summary": "Overall assessment of data reliability"
  }},
  "key_risks": [
    {{
      "risk": "Risk description",
      "source_agents": ["Valid agents only"],
      "severity": "high" | "medium" | "low",
      "mitigation": "Potential mitigation if available"
    }}
  ],
  "key_opportunities": [
    {{
      "opportunity": "Opportunity description",
      "source_agents": ["Valid agents only"],
      "potential": "high" | "medium" | "low"
    }}
  ],
  "balanced_conclusion": "2-3 paragraph conclusion that explicitly addresses data quality limitations and adjusts confidence accordingly. Be transparent about which agents provided useful data and which did not.",
  "methodology_notes": "Note on synthesis methodology and data quality impact",
  "confidence_level": "high" | "medium" | "low" | "very_low"
}}

<ticker>
{ticker}
</ticker>

<validation_summary>
Valid Reports: {valid_count}/{total_agents}
Failed Reports: {failed_count}
Overall Quality: {overall_quality}
</validation_summary>

<timestamp>
{timestamp}
</timestamp>

Return ONLY valid JSON. Do not include any text before or after the JSON object."""
)

# --- CREATE GOVERNOR CHAIN ---


def prepare_governor_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare input for the Governor Agent chain with validation.

    Args:
        input_dict: Contains 'ticker' and 'agent_reports' keys

    Returns:
        Dictionary with formatted inputs and validation results
    """
    ticker = input_dict.get("ticker", "").upper()
    agent_reports = input_dict.get("agent_reports", {})

    # Validate all reports
    validation_results = validate_all_reports(agent_reports)

    # Print validation summary
    print(generate_validation_summary(validation_results))

    # Filter and annotate reports
    filtered_reports = filter_valid_reports(agent_reports, validation_results)

    # Format for LLM
    formatted_reports = format_agent_reports(filtered_reports, validation_results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "ticker": ticker,
        "agent_reports": formatted_reports,
        "timestamp": timestamp,
        "valid_count": validation_results["valid_count"],
        "total_agents": validation_results["total_agents"],
        "failed_count": validation_results["failed_count"],
        "overall_quality": validation_results["overall_quality"],
        "validation_results": validation_results,  # Pass through for report generation
    }


governor_chain = (
    RunnablePassthrough.assign(prepared_input=prepare_governor_input)
    | RunnablePassthrough.assign(
        ticker=lambda x: x["prepared_input"]["ticker"],
        agent_reports=lambda x: x["prepared_input"]["agent_reports"],
        timestamp=lambda x: x["prepared_input"]["timestamp"],
        valid_count=lambda x: x["prepared_input"]["valid_count"],
        total_agents=lambda x: x["prepared_input"]["total_agents"],
        failed_count=lambda x: x["prepared_input"]["failed_count"],
        overall_quality=lambda x: x["prepared_input"]["overall_quality"],
    )
    | governor_prompt_template
    | llm
    | JsonOutputParser()
)

# --- REPORT GENERATION (keeping existing structure, adding validation info) ---


def generate_governor_report(
    ticker: str, synthesis: Dict[str, Any], validation_results: Dict[str, Any]
) -> str:
    """Generate comprehensive investment memo with validation transparency."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
# Investment Memo: ${ticker}
**Prepared by:** Governor Agent (Enhanced with Validation)
**Date:** {timestamp}
**Methodology:** Multi-Agent Synthesis with Holistic AI Principles

---

## Data Quality & Validation Report

**Overall Data Quality:** {validation_results['overall_quality'].upper()}  
**Valid Reports:** {validation_results['valid_count']}/{validation_results['total_agents']}  
**Failed Reports:** {validation_results['failed_count']}

### Agent Coverage Quality:

"""

    coverage = synthesis.get("data_quality_assessment", {}).get("agent_coverage", {})
    for agent_key, quality in coverage.items():
        agent_name = agent_key.replace("_", " ").title()
        emoji_map = {
            "excellent": "🌟",
            "good": "✅",
            "limited": "⚠️",
            "failed": "❌",
            "missing": "❓",
        }
        emoji = emoji_map.get(quality, "❓")
        report += f"- {emoji} **{agent_name}**: {quality.capitalize()}\n"

    report += f"\n**Validation Summary:**  \n{synthesis.get('data_quality_assessment', {}).get('validation_summary', 'No summary available')}\n"

    report += "\n---\n\n## Executive Summary\n\n"
    report += synthesis.get("executive_summary", "No executive summary available.")
    report += f"\n\n**Data Quality Note:**  \n{synthesis.get('data_quality_acknowledgment', 'See validation report above.')}\n"

    report += "\n---\n\n## Critical Insights\n\n"
    report += "Below are the most important findings from VALID specialist agents:\n\n"

    insights = synthesis.get("critical_insights", [])
    for i, insight in enumerate(insights, 1):
        agents = ", ".join(insight.get("source_agents", ["Unknown"]))
        confidence = insight.get("confidence", "unknown").upper()

        report += f"### {i}. {insight.get('insight', 'No insight provided')}\n\n"
        report += f"**Source Agents:** {agents}  \n"
        report += f"**Confidence Level:** {confidence}\n\n"
        report += f"{insight.get('supporting_evidence', 'No supporting evidence provided.')}\n\n"

        quality_note = insight.get("data_quality_note")
        if quality_note:
            report += f"*Data Quality Note: {quality_note}*\n\n"

    if not insights:
        report += "*No critical insights could be reliably identified due to data quality issues.*\n\n"

    # Continue with rest of report structure (consensus, conflicts, etc.)
    report += "---\n\n## Consensus Views\n\n"

    consensus = synthesis.get("consensus_views", [])
    for view in consensus:
        agents = ", ".join(view.get("supporting_agents", ["Unknown"]))
        strength = view.get("strength", "unknown").capitalize()

        report += f"- **{view.get('view', 'No view provided')}**\n"
        report += f"  - *Supporting Agents:* {agents}\n"
        report += f"  - *Consensus Strength:* {strength}\n\n"

    if not consensus:
        report += "*No strong consensus identified across valid agents.*\n\n"

    report += "---\n\n## Conflicting Perspectives\n\n"

    conflicts = synthesis.get("conflicting_perspectives", [])
    for conflict in conflicts:
        report += f"### {conflict.get('topic', 'Unknown Topic')}\n\n"

        persp_a = conflict.get("perspective_a", {})
        persp_b = conflict.get("perspective_b", {})

        report += f"**{persp_a.get('source_agent', 'Unknown Agent')} Perspective:**\n"
        report += f"{persp_a.get('view', 'No view provided')}\n\n"
        report += f"*Evidence:* {persp_a.get('evidence', 'No evidence provided')}\n\n"

        report += f"**{persp_b.get('source_agent', 'Unknown Agent')} Perspective:**\n"
        report += f"{persp_b.get('view', 'No view provided')}\n\n"
        report += f"*Evidence:* {persp_b.get('evidence', 'No evidence provided')}\n\n"

        report += f"**Reconciliation:** {conflict.get('reconciliation', 'No reconciliation provided')}\n\n"

    if not conflicts:
        report += "*No significant conflicts between valid agent perspectives.*\n\n"

    # Risks and Opportunities
    report += "---\n\n## Key Risks\n\n"

    risks = synthesis.get("key_risks", [])
    for risk in risks:
        agents = ", ".join(risk.get("source_agents", ["Unknown"]))
        severity = risk.get("severity", "unknown").upper()

        report += f"### {risk.get('risk', 'Unknown Risk')}\n\n"
        report += f"**Severity:** {severity}  \n"
        report += f"**Source Agents:** {agents}\n\n"

        mitigation = risk.get("mitigation")
        if mitigation and mitigation != "None identified":
            report += f"**Potential Mitigation:** {mitigation}\n\n"

    if not risks:
        report += "*No significant risks identified from valid reports.*\n\n"

    report += "---\n\n## Key Opportunities\n\n"

    opportunities = synthesis.get("key_opportunities", [])
    for opp in opportunities:
        agents = ", ".join(opp.get("source_agents", ["Unknown"]))
        potential = opp.get("potential", "unknown").capitalize()

        report += f"- **{opp.get('opportunity', 'Unknown Opportunity')}**\n"
        report += f"  - *Potential:* {potential}\n"
        report += f"  - *Source Agents:* {agents}\n\n"

    if not opportunities:
        report += "*No significant opportunities identified from valid reports.*\n\n"

    report += "---\n\n## Balanced Conclusion\n\n"
    report += synthesis.get("balanced_conclusion", "No conclusion available.")

    report += f"\n\n**Analysis Confidence:** {synthesis.get('confidence_level', 'medium').upper()}\n"

    report += "\n\n---\n\n## Methodology Notes\n\n"
    report += synthesis.get("methodology_notes", "No methodology notes provided.")

    report += f"""

---

## Disclosure & Data Quality Statement

This investment memo was generated by an AI-powered multi-agent system with **automated 
validation checks**. Each specialist agent's report was validated for:
- Minimum content length and depth
- Presence of quantitative data
- Absence of error messages
- Overall report quality

**Data Quality Impact:**
- Valid Reports: {validation_results['valid_count']}/{validation_results['total_agents']}
- Analysis confidence has been adjusted based on available data quality
- Missing or failed agent reports represent gaps in the analysis

**Important:**
- This is NOT financial advice
- Data quality varies across agents
- Human oversight is essential for investment decisions
- Past performance does not guarantee future results

**Timestamp:** {timestamp}  
**Ticker:** ${ticker}  
**Governor Agent Model:** Gemini Pro Latest (Enhanced with Validation)
"""

    return report.strip()


def save_governor_report(ticker: str, report: str, output_dir: str = "reports") -> str:
    """Save the Governor's investment memo to a file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_investment_memo_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 Investment memo saved to: {filepath}")
    return filepath


# --- MAIN GOVERNOR FUNCTION ---


def run_governor_agent(
    ticker: str, agent_reports: Dict[str, str], save_to_file: bool = True
) -> tuple[str, str]:
    """
    Execute the Governor Agent synthesis with validation.

    Args:
        ticker: Stock ticker symbol
        agent_reports: Dictionary mapping agent names to their report strings
        save_to_file: Whether to save the report to a file

    Returns:
        Tuple of (summary_report, detailed_report)
    """
    print(f"\n{'='*60}")
    print(f"🤖 Governor Agent (Enhanced): Synthesizing ${ticker}")
    print(f"{'='*60}\n")

    if not agent_reports:
        error_msg = "No agent reports provided to Governor Agent"
        print(f"❌ {error_msg}")
        return error_msg, error_msg

    print(f"📊 Received reports from {len(agent_reports)} agents")

    try:
        # Execute the Governor chain (includes validation)
        chain_input = {"ticker": ticker.upper(), "agent_reports": agent_reports}

        # Get validation results from prepared input
        prepared = prepare_governor_input(chain_input)
        validation_results = prepared["validation_results"]

        # Execute synthesis
        synthesis_json = governor_chain.invoke(chain_input)

        # Generate detailed investment memo
        detailed_report = generate_governor_report(
            ticker, synthesis_json, validation_results
        )

        # Save to file if requested
        if save_to_file:
            save_governor_report(ticker, detailed_report)

        # Create summary
        exec_summary = synthesis_json.get("executive_summary", "No summary available.")
        confidence = synthesis_json.get("confidence_level", "unknown")

        summary_report = f"""
**Governor Agent Summary: ${ticker}**

📋 **Executive Summary:**
{exec_summary}

📊 **Data Quality:**
* **Overall Quality:** {validation_results['overall_quality'].upper()}
* **Valid Reports:** {validation_results['valid_count']}/{validation_results['total_agents']}
* **Failed Reports:** {validation_results['failed_count']}
* **Analysis Confidence:** {confidence.upper()}

💡 **Key Takeaway:**
{synthesis_json.get('data_quality_acknowledgment', 'See full report.')}

⚠️ **Validation-Enhanced Analysis**
This synthesis includes automated validation of all agent reports.
Confidence levels are adjusted based on data quality.

---
*Full investment memo with validation transparency saved to file*
*Agent: Governor Agent (Enhanced) | Model: Gemini Pro*
        """

        print("✅ Governor_Agent: Synthesis complete")

        # Return comprehensive dictionary like news_agent
        return {
            "ticker": ticker,
            "agent": "Governor",
            "executive_summary": exec_summary,
            "confidence_level": confidence,
            "validation_results": validation_results,
            "overall_assessment": synthesis_json.get("overall_assessment", ""),
            "investment_thesis": synthesis_json.get("investment_thesis", {}),
            "key_risks": synthesis_json.get("key_risks", []),
            "key_opportunities": synthesis_json.get("key_opportunities", []),
            "recommendation": synthesis_json.get("recommendation", ""),
            "data_quality_acknowledgment": synthesis_json.get(
                "data_quality_acknowledgment", ""
            ),
            "agent_reports_used": list(agent_reports.keys()),
            "summary_report": summary_report.strip(),
            "detailed_report": detailed_report,
        }

    except Exception as e:
        print(f"❌ Governor_Agent: Synthesis failed - {e}")
        import traceback

        traceback.print_exc()

        error_report = f"""
**Governor Agent Report: ${ticker}**

⚠️ **Error:** Synthesis could not be completed.

**Details:** {str(e)}
        """
        return {
            "ticker": ticker,
            "agent": "Governor",
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
        print("Usage: python governor_agent.py <TICKER>\n")

    # Mock agent reports for testing
    mock_reports = {
        "Social Agent": """
**Social Agent Summary: $TSLA**

📊 **Sentiment:** Bullish (+0.65)
**Data Quality:** Good
**Sources:** 8

Retail investors show strong optimism around AI initiatives.
        """,
        "Analyst Agent": """
**Analyst Agent Report: $TSLA**

📈 **Consensus:** Buy
**Target:** $285.50

Recent upgrades from major firms.
        """,
        "News Agent": "Error: Could not fetch news data",  # This will fail validation
        "Chart Agent": "",  # This will fail validation (empty)
        "SEC Agent": "Brief note",  # This will get "limited" quality
    }

    result = run_governor_agent(ticker, mock_reports, save_to_file=True)

    print("\n" + "=" * 60)
    print("GOVERNOR SUMMARY OUTPUT")
    print("=" * 60 + "\n")
    print(result.get("summary_report", ""))
    print("\n" + "=" * 60)
    print("\n📄 Full investment memo saved to 'reports/' directory")
