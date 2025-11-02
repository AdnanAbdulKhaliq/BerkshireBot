"""
Governor Agent - Analyst Swarm Component

This agent receives reports from all specialist agents (SEC, News, Social, Chart, Analyst)
and synthesizes them into a single, balanced investment memo following Holistic AI principles:
- Be unbiased and present multiple perspectives
- Cite sources from each agent
- Present conflicting views fairly
- Identify consensus and disagreements

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
    required_vars = {
        "GEMINI_API_KEY": "Gemini LLM for synthesis"
    }
    
    missing = []
    for var_name, description in required_vars.items():
        if var_name not in os.environ:
            missing.append(f"  ❌ {var_name}: {description}")
    
    if missing:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing)
        raise EnvironmentError(error_msg)
    
    print("✅ Governor_Agent: All environment variables validated")

# Validate at module load time
validate_environment()

# --- SETUP MODEL ---

# Initialize Gemini LLM with moderate temperature for balanced synthesis
llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest",
    temperature=0.3,  # Slightly higher for nuanced synthesis
    api_key=os.environ["GEMINI_API_KEY"]
)

# --- PROMPT TEMPLATE ---

governor_prompt_template = ChatPromptTemplate.from_template(
    """You are the "Governor Agent" - the senior analyst who synthesizes insights from 
a team of specialist analysts into a single, balanced investment memo for ${ticker}.

**HOLISTIC AI PRINCIPLES - YOUR CORE MANDATE:**
1. **Unbiased Analysis**: Present all perspectives fairly, avoid cherry-picking
2. **Source Attribution**: Cite which agent provided each insight
3. **Conflicting Views**: Explicitly highlight where agents disagree
4. **Consensus Building**: Identify areas of agreement across multiple agents
5. **Transparency**: Acknowledge gaps in data or analysis

**YOUR SPECIALIST TEAM REPORTS:**

{agent_reports}

**YOUR TASK:**
Synthesize these reports into a cohesive investment memo that:
1. Identifies the 3-5 most critical insights across ALL agents
2. Highlights consensus views (where 2+ agents agree)
3. Presents conflicting perspectives fairly (where agents disagree)
4. Provides a balanced conclusion that weighs all evidence

**CRITICAL RULES:**
- Never ignore an agent's report - every perspective matters
- When agents conflict, present BOTH views with equal weight
- Cite the source agent for every major claim (e.g., "The Social Agent found...")
- If data is missing or limited, explicitly state this
- Your conclusion should reflect the COLLECTIVE intelligence, not a single view

Return your synthesis in this EXACT JSON format:
{{
  "executive_summary": "3-4 sentence overview synthesizing all perspectives",
  "critical_insights": [
    {{
      "insight": "The key finding",
      "source_agents": ["Social Agent", "Analyst Agent"],
      "supporting_evidence": "Brief evidence summary",
      "confidence": "high" | "medium" | "low"
    }}
  ],
  "consensus_views": [
    {{
      "view": "What multiple agents agree on",
      "supporting_agents": ["Agent 1", "Agent 2", "Agent 3"],
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
      "sec_agent": "present" | "missing",
      "news_agent": "present" | "missing",
      "social_agent": "present" | "missing",
      "chart_agent": "present" | "missing",
      "analyst_agent": "present" | "missing"
    }}
  }},
  "key_risks": [
    {{
      "risk": "Risk description",
      "source_agents": ["Agent 1", "Agent 2"],
      "severity": "high" | "medium" | "low",
      "mitigation": "Potential mitigation if available"
    }}
  ],
  "key_opportunities": [
    {{
      "opportunity": "Opportunity description",
      "source_agents": ["Agent 1", "Agent 2"],
      "potential": "high" | "medium" | "low"
    }}
  ],
  "balanced_conclusion": "2-3 paragraph conclusion that weighs all evidence, presents the investment case from multiple angles, and provides a nuanced recommendation that reflects the collective intelligence of all agents. Do NOT make a definitive buy/sell recommendation - instead, present the case for different investment strategies based on risk tolerance and time horizon.",
  "methodology_notes": "Brief note on how this synthesis was conducted and any limitations"
}}

<ticker>
{ticker}
</ticker>

<timestamp>
{timestamp}
</timestamp>

Return ONLY valid JSON. Do not include any text before or after the JSON object."""
)

# --- AGENT REPORT FORMATTING ---

def format_agent_reports(agent_reports: Dict[str, str]) -> str:
    """
    Format all agent reports into a structured format for the Governor.
    
    Args:
        agent_reports: Dictionary mapping agent names to their report strings
    
    Returns:
        Formatted string with all agent reports
    """
    formatted = ""
    
    agent_order = ["SEC Agent", "News Agent", "Social Agent", "Chart Agent", "Analyst Agent"]
    
    for agent_name in agent_order:
        report = agent_reports.get(agent_name, f"[{agent_name} report not available]")
        
        formatted += f"\n{'='*70}\n"
        formatted += f"{agent_name.upper()} REPORT\n"
        formatted += f"{'='*70}\n\n"
        formatted += report
        formatted += "\n\n"
    
    return formatted

# --- CREATE GOVERNOR CHAIN ---

def prepare_governor_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare input for the Governor Agent chain.
    
    Args:
        input_dict: Contains 'ticker' and 'agent_reports' keys
    
    Returns:
        Dictionary with formatted inputs for the prompt
    """
    ticker = input_dict.get("ticker", "").upper()
    agent_reports = input_dict.get("agent_reports", {})
    
    formatted_reports = format_agent_reports(agent_reports)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "ticker": ticker,
        "agent_reports": formatted_reports,
        "timestamp": timestamp
    }

governor_chain = (
    RunnablePassthrough.assign(prepared_input=prepare_governor_input)
    | RunnablePassthrough.assign(
        ticker=lambda x: x["prepared_input"]["ticker"],
        agent_reports=lambda x: x["prepared_input"]["agent_reports"],
        timestamp=lambda x: x["prepared_input"]["timestamp"]
    )
    | governor_prompt_template
    | llm
    | JsonOutputParser()
)

# --- REPORT GENERATION ---

def generate_governor_report(ticker: str, synthesis: Dict[str, Any]) -> str:
    """
    Generate a comprehensive investment memo from the Governor's synthesis.
    
    Args:
        ticker: Stock ticker symbol
        synthesis: Synthesis JSON from Governor Agent
    
    Returns:
        Formatted markdown investment memo
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Investment Memo: ${ticker}
**Prepared by:** Governor Agent (Analyst Swarm)
**Date:** {timestamp}
**Methodology:** Multi-Agent Synthesis with Holistic AI Principles

---

## Executive Summary

{synthesis.get('executive_summary', 'No executive summary available.')}

---

## Critical Insights

Below are the most important findings synthesized from all specialist agents:

"""
    
    insights = synthesis.get('critical_insights', [])
    for i, insight in enumerate(insights, 1):
        agents = ", ".join(insight.get('source_agents', ['Unknown']))
        confidence = insight.get('confidence', 'unknown').upper()
        
        report += f"""
### {i}. {insight.get('insight', 'No insight provided')}

**Source Agents:** {agents}  
**Confidence Level:** {confidence}

{insight.get('supporting_evidence', 'No supporting evidence provided.')}

"""
    
    if not insights:
        report += "*No critical insights were identified.*\n"
    
    report += "\n---\n\n## Consensus Views\n\n"
    report += "These views are supported by multiple specialist agents:\n\n"
    
    consensus = synthesis.get('consensus_views', [])
    for view in consensus:
        agents = ", ".join(view.get('supporting_agents', ['Unknown']))
        strength = view.get('strength', 'unknown').capitalize()
        
        report += f"- **{view.get('view', 'No view provided')}**\n"
        report += f"  - *Supporting Agents:* {agents}\n"
        report += f"  - *Consensus Strength:* {strength}\n\n"
    
    if not consensus:
        report += "*No strong consensus identified across agents.*\n"
    
    report += "\n---\n\n## Conflicting Perspectives\n\n"
    report += "Where specialist agents disagree, we present both views fairly:\n\n"
    
    conflicts = synthesis.get('conflicting_perspectives', [])
    for conflict in conflicts:
        report += f"### {conflict.get('topic', 'Unknown Topic')}\n\n"
        
        persp_a = conflict.get('perspective_a', {})
        persp_b = conflict.get('perspective_b', {})
        
        report += f"**{persp_a.get('source_agent', 'Unknown Agent')} Perspective:**\n"
        report += f"{persp_a.get('view', 'No view provided')}\n\n"
        report += f"*Evidence:* {persp_a.get('evidence', 'No evidence provided')}\n\n"
        
        report += f"**{persp_b.get('source_agent', 'Unknown Agent')} Perspective:**\n"
        report += f"{persp_b.get('view', 'No view provided')}\n\n"
        report += f"*Evidence:* {persp_b.get('evidence', 'No evidence provided')}\n\n"
        
        report += f"**Reconciliation:** {conflict.get('reconciliation', 'No reconciliation provided')}\n\n"
    
    if not conflicts:
        report += "*No significant conflicts between agent perspectives.*\n"
    
    report += "\n---\n\n## Data Quality Assessment\n\n"
    
    quality = synthesis.get('data_quality_assessment', {})
    report += f"**Overall Quality:** {quality.get('overall_quality', 'unknown').capitalize()}\n\n"
    
    coverage = quality.get('agent_coverage', {})
    report += "**Agent Coverage:**\n"
    for agent, status in coverage.items():
        emoji = "✅" if status == "present" else "❌"
        agent_display = agent.replace('_', ' ').title()
        report += f"- {emoji} {agent_display}: {status.capitalize()}\n"
    
    gaps = quality.get('gaps_identified', [])
    if gaps:
        report += "\n**Identified Gaps:**\n"
        for gap in gaps:
            report += f"- {gap}\n"
    
    report += "\n---\n\n## Key Risks\n\n"
    
    risks = synthesis.get('key_risks', [])
    for risk in risks:
        agents = ", ".join(risk.get('source_agents', ['Unknown']))
        severity = risk.get('severity', 'unknown').upper()
        
        report += f"### {risk.get('risk', 'Unknown Risk')}\n\n"
        report += f"**Severity:** {severity}  \n"
        report += f"**Source Agents:** {agents}\n\n"
        
        mitigation = risk.get('mitigation')
        if mitigation and mitigation != "None identified":
            report += f"**Potential Mitigation:** {mitigation}\n\n"
    
    if not risks:
        report += "*No significant risks identified.*\n"
    
    report += "\n---\n\n## Key Opportunities\n\n"
    
    opportunities = synthesis.get('key_opportunities', [])
    for opp in opportunities:
        agents = ", ".join(opp.get('source_agents', ['Unknown']))
        potential = opp.get('potential', 'unknown').capitalize()
        
        report += f"- **{opp.get('opportunity', 'Unknown Opportunity')}**\n"
        report += f"  - *Potential:* {potential}\n"
        report += f"  - *Source Agents:* {agents}\n\n"
    
    if not opportunities:
        report += "*No significant opportunities identified.*\n"
    
    report += "\n---\n\n## Balanced Conclusion\n\n"
    report += synthesis.get('balanced_conclusion', 'No conclusion available.')
    
    report += "\n\n---\n\n## Methodology Notes\n\n"
    report += synthesis.get('methodology_notes', 'No methodology notes provided.')
    
    report += f"""

---

## Disclosure

This investment memo was generated by an AI-powered multi-agent system (Analyst Swarm) 
that synthesizes insights from five specialist agents: SEC Agent, News Agent, Social Agent, 
Chart Agent, and Analyst Agent.

**Important:**
- This is NOT financial advice
- All perspectives are presented for informational purposes only
- Human oversight and judgment are essential for investment decisions
- Past performance does not guarantee future results

**Timestamp:** {timestamp}  
**Ticker:** ${ticker}  
**Governor Agent Model:** Gemini Pro Latest
"""
    
    return report.strip()

def save_governor_report(ticker: str, report: str, output_dir: str = "reports") -> str:
    """
    Save the Governor's investment memo to a file.
    
    Args:
        ticker: Stock ticker symbol
        report: Formatted report content
        output_dir: Directory to save reports
    
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_investment_memo_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Investment memo saved to: {filepath}")
    return filepath

# --- MAIN GOVERNOR FUNCTION ---

def run_governor_agent(
    ticker: str, 
    agent_reports: Dict[str, str],
    save_to_file: bool = True
) -> tuple[str, str]:
    """
    Execute the Governor Agent synthesis.
    
    Args:
        ticker: Stock ticker symbol
        agent_reports: Dictionary mapping agent names to their report strings
                      Example: {"Social Agent": "...", "Analyst Agent": "..."}
        save_to_file: Whether to save the report to a file
    
    Returns:
        Tuple of (summary_report, detailed_report)
    """
    print(f"\n{'='*60}")
    print(f"🤖 Governor Agent: Synthesizing analysis for ${ticker}")
    print(f"{'='*60}\n")
    
    # Validate we have at least one agent report
    if not agent_reports:
        error_msg = "No agent reports provided to Governor Agent"
        print(f"❌ {error_msg}")
        return error_msg, error_msg
    
    print(f"📊 Received reports from {len(agent_reports)} agents:")
    for agent_name in agent_reports.keys():
        print(f"  ✓ {agent_name}")
    
    try:
        # Execute the Governor chain
        synthesis_json = governor_chain.invoke({
            "ticker": ticker.upper(),
            "agent_reports": agent_reports
        })
        
        # Generate detailed investment memo
        detailed_report = generate_governor_report(ticker, synthesis_json)
        
        # Save to file if requested
        if save_to_file:
            save_governor_report(ticker, detailed_report)
        
        # Create a concise summary for console output
        exec_summary = synthesis_json.get('executive_summary', 'No summary available.')
        quality = synthesis_json.get('data_quality_assessment', {}).get('overall_quality', 'unknown')
        consensus_count = len(synthesis_json.get('consensus_views', []))
        conflict_count = len(synthesis_json.get('conflicting_perspectives', []))
        
        summary_report = f"""
**Governor Agent Summary: ${ticker}**

📋 **Executive Summary:**
{exec_summary}

📊 **Analysis Quality:**
* **Data Quality:** {quality.capitalize()}
* **Consensus Views:** {consensus_count}
* **Conflicting Perspectives:** {conflict_count}
* **Agent Reports Processed:** {len(agent_reports)}

💡 **Key Takeaway:**
{synthesis_json.get('balanced_conclusion', 'See full report.')[:200]}...

---
*Full investment memo saved to file*
*Agent: Governor Agent | Model: Gemini Pro*
        """
        
        print("✅ Governor_Agent: Synthesis complete")
        return summary_report.strip(), detailed_report
        
    except Exception as e:
        print(f"❌ Governor_Agent: Synthesis failed - {e}")
        import traceback
        traceback.print_exc()
        
        error_report = f"""
**Governor Agent Report: ${ticker}**

⚠️ **Error:** Synthesis could not be completed.

**Details:** {str(e)}
        """
        return error_report.strip(), error_report.strip()

# --- CLI ENTRY POINT ---

if __name__ == "__main__":
    import sys
    
    # Example usage with mock reports
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "TSLA"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python governor_agent.py <TICKER>\n")
    
    # Mock agent reports for testing
    mock_reports = {
        "Social Agent": """
**Social-Sentimentalist Agent Summary: $TSLA**

📊 **Quick Overview**
* **Sentiment:** Bullish (+0.65)
* **Data Quality:** Good
* **Sources Analyzed:** 8

📝 **Executive Summary:**
Retail investors show strong optimism around Tesla's AI initiatives and Full Self-Driving progress.
However, concerns about valuation and competition are present.
        """,
        "Analyst Agent": """
**Analyst Agent Report: $TSLA**

📈 **Wall Street Consensus**
* **Overall Rating:** Buy
* **Average Price Target:** $285.50

**Recent Analyst Activity:**
* **Morgan Stanley:** Upgrade (from Hold to Buy)
* **Goldman Sachs:** Maintained Buy rating

📝 **Analysis Summary:**
Professional analysts are increasingly bullish on Tesla's long-term AI and autonomy potential.
        """
    }
    
    # Run the Governor Agent
    summary, detailed = run_governor_agent(ticker, mock_reports, save_to_file=True)
    
    # Display summary
    print("\n" + "="*60)
    print("GOVERNOR SUMMARY OUTPUT")
    print("="*60 + "\n")
    print(summary)
    print("\n" + "="*60)
    print("\n📄 Full investment memo saved to 'reports/' directory")