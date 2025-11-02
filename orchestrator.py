"""
LangGraph Orchestrator - Analyst Swarm Workflow Manager (Enhanced with Retry Logic)

This orchestrator manages the entire analyst swarm workflow with:
- Automatic retry logic for failed agents
- Exponential backoff
- Detailed error tracking
- Graceful degradation

Environment Variables Required:
    - All agent-specific API keys (see individual agent files)
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, Callable, Any, Dict
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Import all specialist agents
try:
    from sec_agent import run_sec_agent
    from news_agent import analyze_company_sentiment
    from social_agent import run_social_agent
    from analyst_agent import run_analyst_agent
    from governor_agent import run_governor_agent
    from risk_assessment_agent import run_risk_assessment_agent
except ImportError as e:
    print(f"⚠️ Warning: Could not import all agents: {e}")
    print("Make sure all agent files are in the same directory")

load_dotenv()

# --- RETRY CONFIGURATION ---

RETRY_CONFIG = {
    "max_retries": 2,
    "base_delay": 2,  # seconds
    "max_delay": 10,  # seconds
    "exponential_base": 2,
    "jitter": True,  # Add randomness to prevent thundering herd
}

# --- STATE DEFINITION (MODIFIED) ---

def merge_lists(left: list | None, right: list | None) -> list:
    """Merge two lists, used for errors and warnings."""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


class AnalystSwarmState(TypedDict):
    """The state object that flows through the entire workflow."""

    # Input (read-only after initialization)
    ticker: str
    timestamp: str

    # Specialist Agent Reports (Split into Summary and Detailed)
    sec_agent_summary: str
    sec_agent_detailed: str
    news_agent_summary: str
    news_agent_detailed: str
    social_agent_summary: str
    social_agent_detailed: str
    chart_agent_summary: str
    chart_agent_detailed: str
    analyst_agent_summary: str
    analyst_agent_detailed: str

    # Specialist Agent Status
    sec_agent_status: str
    news_agent_status: str
    social_agent_status: str
    chart_agent_status: str
    analyst_agent_status: str

    # Retry tracking
    sec_agent_attempts: int
    news_agent_attempts: int
    social_agent_attempts: int
    chart_agent_attempts: int
    analyst_agent_attempts: int

    # Governor Agent Output
    governor_summary: str
    governor_full_memo: str  # This is the "detailed report" from the governor
    governor_status: str
    governor_attempts: int

    # Risk Assessment Output
    risk_summary: str
    risk_full_report: str # This is the "detailed report" from risk
    risk_status: str
    risk_attempts: int

    # Final Status
    workflow_status: str
    errors: Annotated[list, merge_lists]
    warnings: Annotated[list, merge_lists]


# --- RETRY LOGIC DECORATOR ---

def with_retry(
    agent_name: str,
    max_retries: int = RETRY_CONFIG["max_retries"],
    base_delay: float = RETRY_CONFIG["base_delay"],
    max_delay: float = RETRY_CONFIG["max_delay"],
    exponential_base: float = RETRY_CONFIG["exponential_base"],
) -> Callable:
    """Decorator to add retry logic to agent functions."""

    def decorator(agent_func: Callable) -> Callable:
        def wrapper(state: AnalystSwarmState) -> AnalystSwarmState:
            attempts_key = f"{agent_name.lower().replace(' ', '_')}_attempts"
            state.setdefault(attempts_key, 0) # Ensure key exists

            for attempt in range(max_retries + 1):
                try:
                    # Update attempt counter
                    state[attempts_key] = attempt + 1

                    if attempt > 0:
                        delay = min(
                            base_delay * (exponential_base ** (attempt - 1)), max_delay
                        )
                        if RETRY_CONFIG["jitter"]:
                            import random
                            delay = delay * (0.5 + random.random())
                        print(
                            f"⏳ {agent_name}: Retry attempt {attempt + 1}/{max_retries + 1} after {delay:.1f}s delay..."
                        )
                        time.sleep(delay)

                    # Execute the agent function
                    # We pass a copy to avoid partial updates on failure
                    result_state = agent_func(state.copy()) 

                    # Check if successful
                    status_key = f"{agent_name.lower().replace(' ', '_')}_status"
                    if result_state.get(status_key) == "success":
                        if attempt > 0:
                            print(
                                f"✅ {agent_name}: Succeeded on retry attempt {attempt + 1}"
                            )
                            result_state.setdefault("warnings", []).append(
                                f"{agent_name} required {attempt + 1} attempt(s) to succeed"
                            )
                        # Merge successful state back
                        state.update(result_state)
                        return state

                    if attempt == max_retries:
                        print(
                            f"❌ {agent_name}: Failed after {max_retries + 1} attempts (no exception)"
                        )
                        state.update(result_state) # Record the failure state
                        return state

                except Exception as e:
                    print(f"⚠️ {agent_name}: Attempt {attempt + 1} failed - {e}")

                    if attempt == max_retries:
                        print(f"❌ {agent_name}: All retry attempts exhausted")
                        status_key = f"{agent_name.lower().replace(' ', '_')}_status"
                        summary_key = f"{agent_name.lower().replace(' ', '_')}_summary"
                        detailed_key = f"{agent_name.lower().replace(' ', '_')}_detailed"
                        
                        error_msg = f"Error after {max_retries + 1} attempts: {str(e)}"
                        state[status_key] = "failed"
                        state[summary_key] = error_msg
                        state[detailed_key] = error_msg
                        
                        state.setdefault("errors", []).append(
                            f"{agent_name}: {str(e)} (after {max_retries + 1} attempts)"
                        )
                        return state
            
            return state # Should be unreachable, but for safety

        return wrapper

    return decorator


# --- AGENT NODE FUNCTIONS (MODIFIED) ---

@with_retry("SEC Agent")
def sec_agent_node(state: AnalystSwarmState) -> dict:
    """Run the SEC Agent."""
    print(f"\n🔍 Running SEC Agent for ${state['ticker']}...")
    try:
        result = run_sec_agent(state["ticker"], save_to_file=True)
        if "error" in result:
             raise Exception(result["error"])
        
        print("✅ SEC Agent completed")
        return {
            "sec_agent_summary": result.get("summary_report", str(result)),
            "sec_agent_detailed": result.get("detailed_report", "No detailed report available."),
            "sec_agent_status": "success"
        }
    except Exception as e:
        print(f"❌ SEC Agent failed: {e}")
        error_msg = f"Error: {str(e)}"
        return {
            "sec_agent_summary": error_msg,
            "sec_agent_detailed": error_msg,
            "sec_agent_status": "failed",
            "errors": [f"SEC Agent: {str(e)}"]
        }


@with_retry("News Agent")
def news_agent_node(state: AnalystSwarmState) -> dict:
    """Run the News Agent."""
    print(f"\n📰 Running News Agent for ${state['ticker']}...")
    try:
        company_name = state["ticker"]
        result = analyze_company_sentiment(
            company_name=company_name, max_articles=30, lookback_days=7, verbose=True
        )
        if "error" in result:
            raise Exception(result["error"])

        summary = f"""
**News Agent Report: ${state['ticker']}**
📰 **Sentiment Analysis:**
* Overall Score: {result['weighted_sentiment_score']:.2f}
* Bullish Pressure: {result['bullish_pressure']*100:.1f}%
* Bearish Pressure: {result['bearish_pressure']*100:.1f}%
* Articles Analyzed: {result['total_articles_analyzed']}
* High-Impact Articles: {result['high_impact_count']}
{result['dashboard_summary']}
        """
        print("✅ News Agent completed")
        return {
            "news_agent_summary": summary,
            "news_agent_detailed": json.dumps(result, indent=2, default=str),
            "news_agent_status": "success"
        }
    except Exception as e:
        print(f"❌ News Agent failed: {e}")
        error_msg = f"Error: {str(e)}"
        return {
            "news_agent_summary": error_msg,
            "news_agent_detailed": error_msg,
            "news_agent_status": "failed",
            "errors": [f"News Agent: {str(e)}"]
        }


@with_retry("Social Agent")
def social_agent_node(state: AnalystSwarmState) -> dict:
    """Run the Social Sentiment Agent with retry logic."""
    print(f"\n💬 Running Social Agent for ${state['ticker']}...")
    try:
        result = run_social_agent(state["ticker"], save_to_file=True)
        if "error" in result:
             raise Exception(result["error"])

        print("✅ Social Agent completed")
        return {
            "social_agent_summary": result.get("summary_report", str(result)),
            "social_agent_detailed": result.get("detailed_report", "No detailed report available."),
            "social_agent_status": "success"
        }
    except Exception as e:
        print(f"❌ Social Agent failed: {e}")
        error_msg = f"Error: {str(e)}"
        return {
            "social_agent_summary": error_msg,
            "social_agent_detailed": error_msg,
            "social_agent_status": "failed",
            "errors": [f"Social Agent: {str(e)}"]
        }


@with_retry("Chart Agent")
def chart_agent_node(state: AnalystSwarmState) -> dict:
    """Run the Chart/Technical Analysis Agent (placeholder)."""
    print(f"\n📊 Running Chart Agent for ${state['ticker']}...")
    try:
        report = f"""
**Chart Agent Report: ${state['ticker']}**
📈 **Technical Analysis:**
* Price trends and patterns
* Support/resistance levels
* Momentum indicators
*Note: Full Chart agent implementation pending*
        """
        print("✅ Chart Agent completed")
        return {
            "chart_agent_summary": report,
            "chart_agent_detailed": report,  # Summary and detailed are the same for placeholder
            "chart_agent_status": "success"
        }
    except Exception as e:
        print(f"❌ Chart Agent failed: {e}")
        error_msg = f"Error: {str(e)}"
        return {
            "chart_agent_summary": error_msg,
            "chart_agent_detailed": error_msg,
            "chart_agent_status": "failed",
            "errors": [f"Chart Agent: {str(e)}"]
        }


@with_retry("Analyst Agent")
def analyst_agent_node(state: AnalystSwarmState) -> dict:
    """Run the Professional Analyst Ratings Agent with retry logic."""
    print(f"\n📊 Running Analyst Agent for ${state['ticker']}...")
    try:
        result = run_analyst_agent(state["ticker"], save_to_file=True)
        if "error" in result:
             raise Exception(result["error"])

        print("✅ Analyst Agent completed")
        return {
            "analyst_agent_summary": result.get("summary_report", str(result)),
            "analyst_agent_detailed": result.get("detailed_report", "No detailed report available."),
            "analyst_agent_status": "success"
        }
    except Exception as e:
        print(f"❌ Analyst Agent failed: {e}")
        error_msg = f"Error: {str(e)}"
        return {
            "analyst_agent_summary": error_msg,
            "analyst_agent_detailed": error_msg,
            "analyst_agent_status": "failed",
            "errors": [f"Analyst Agent: {str(e)}"]
        }


@with_retry("Governor Agent", max_retries=1)
def governor_agent_node(state: AnalystSwarmState) -> dict:
    """Run the Governor Agent to synthesize all specialist reports."""
    print(f"\n🎯 Running Governor Agent for ${state['ticker']}...")
    try:
        # Collect all agent *summary* reports
        agent_reports = {
            "SEC Agent": state.get("sec_agent_summary", "Not available"),
            "News Agent": state.get("news_agent_summary", "Not available"),
            "Social Agent": state.get("social_agent_summary", "Not available"),
            "Chart Agent": state.get("chart_agent_summary", "Not available"),
            "Analyst Agent": state.get("analyst_agent_summary", "Not available"),
        }

        result = run_governor_agent(state["ticker"], agent_reports, save_to_file=True)
        if "error" in result:
             raise Exception(result["error"])

        print("✅ Governor Agent completed")
        return {
            "governor_summary": result.get("summary_report", str(result)),
            "governor_full_memo": result.get("detailed_report", "No detailed memo available."),
            "governor_status": "success"
        }
    except Exception as e:
        print(f"❌ Governor Agent failed: {e}")
        error_msg = f"Error: {str(e)}"
        return {
            "governor_summary": error_msg,
            "governor_full_memo": error_msg,
            "governor_status": "failed",
            "errors": [f"Governor Agent: {str(e)}"]
        }


@with_retry("Risk Assessment Agent", max_retries=1)
def risk_assessment_node(state: AnalystSwarmState) -> dict:
    """Run the Risk Assessment Agent on the Governor's memo."""
    print(f"\n⚠️ Running Risk Assessment Agent for ${state['ticker']}...")
    try:
        if state.get("governor_status") == "success":
            governor_memo = state.get("governor_full_memo", "")
            if not governor_memo:
                raise Exception("Governor memo is empty, cannot assess risk.")

            result = run_risk_assessment_agent(
                state["ticker"], governor_memo, save_to_file=True
            )
            if "error" in result:
                raise Exception(result["error"])

            print("✅ Risk Assessment Agent completed")
            return {
                "risk_summary": result.get("summary_report", str(result)),
                "risk_full_report": result.get("detailed_report", "No detailed report available."),
                "risk_status": "success"
            }
        else:
            skip_msg = "Skipped: Governor Agent failed or memo not available"
            print("⚠️ Risk Assessment skipped due to Governor failure")
            return {
                "risk_summary": skip_msg,
                "risk_full_report": skip_msg,
                "risk_status": "skipped"
            }
    except Exception as e:
        print(f"❌ Risk Assessment Agent failed: {e}")
        error_msg = f"Error: {str(e)}"
        return {
            "risk_summary": error_msg,
            "risk_full_report": error_msg,
            "risk_status": "failed",
            "errors": [f"Risk Assessment: {str(e)}"]
        }


# --- WORKFLOW CONSTRUCTION ---

def create_analyst_swarm_workflow() -> StateGraph:
    """Create the LangGraph workflow with retry-enabled agents."""
    from langgraph.graph import START
    
    workflow = StateGraph(AnalystSwarmState)
    workflow.add_node("sec_agent", sec_agent_node)
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("social_agent", social_agent_node)
    workflow.add_node("chart_agent", chart_agent_node)
    workflow.add_node("analyst_agent", analyst_agent_node)
    workflow.add_node("governor", governor_agent_node)
    workflow.add_node("risk_assessment", risk_assessment_node)

    # Connect START to all specialist agents (they run in parallel)
    workflow.add_edge(START, "sec_agent")
    workflow.add_edge(START, "news_agent")
    workflow.add_edge(START, "social_agent")
    workflow.add_edge(START, "chart_agent")
    workflow.add_edge(START, "analyst_agent")

    # All specialist agents feed into governor
    workflow.add_edge("sec_agent", "governor")
    workflow.add_edge("news_agent", "governor")
    workflow.add_edge("social_agent", "governor")
    workflow.add_edge("chart_agent", "governor")
    workflow.add_edge("analyst_agent", "governor")
    
    # Governor feeds into risk assessment, which ends
    workflow.add_edge("governor", "risk_assessment")
    workflow.add_edge("risk_assessment", END)

    return workflow


# --- MAIN ORCHESTRATION FUNCTION ---

def run_analyst_swarm(ticker: str, save_state: bool = True) -> AnalystSwarmState:
    """
    Execute the complete Analyst Swarm workflow with retry logic.
    """
    print("\n" + "=" * 70)
    print("🚀 ANALYST SWARM - MULTI-AGENT ANALYSIS SYSTEM (Enhanced)")
    print("=" * 70)
    print(f"📊 Target: ${ticker.upper()}")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"🔄 Retry Logic: Enabled (max {RETRY_CONFIG['max_retries']} retries per agent)"
    )
    print("=" * 70 + "\n")

    # Initialize state (MODIFIED with new fields)
    initial_state = AnalystSwarmState(
        ticker=ticker.upper(),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sec_agent_summary="", sec_agent_detailed="",
        news_agent_summary="", news_agent_detailed="",
        social_agent_summary="", social_agent_detailed="",
        chart_agent_summary="", chart_agent_detailed="",
        analyst_agent_summary="", analyst_agent_detailed="",
        sec_agent_status="pending",
        news_agent_status="pending",
        social_agent_status="pending",
        chart_agent_status="pending",
        analyst_agent_status="pending",
        sec_agent_attempts=0,
        news_agent_attempts=0,
        social_agent_attempts=0,
        chart_agent_attempts=0,
        analyst_agent_attempts=0,
        governor_summary="",
        governor_full_memo="",
        governor_status="pending",
        governor_attempts=0,
        risk_summary="",
        risk_full_report="",
        risk_status="pending",
        risk_attempts=0,
        workflow_status="running",
        errors=[],
        warnings=[],
    )

    try:
        workflow = create_analyst_swarm_workflow()
        app = workflow.compile()
        print("⚙️ Executing workflow with retry logic...\n")
        
        final_state = app.invoke(initial_state)

        if final_state.get("errors"):
            final_state["workflow_status"] = "completed_with_errors"
        else:
            final_state["workflow_status"] = "completed_successfully"

        if save_state:
            save_workflow_state(ticker, final_state)

        print_workflow_summary(final_state)
        return final_state

    except Exception as e:
        print(f"\n❌ WORKFLOW FAILED: {e}")
        import traceback
        traceback.print_exc()
        initial_state["workflow_status"] = "failed"
        initial_state["errors"].append(f"Workflow error: {str(e)}")
        return initial_state


def save_workflow_state(
    ticker: str, state: AnalystSwarmState, output_dir: str = "workflow_states"
):
    """Save the complete workflow state to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_workflow_state_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Create a serializable copy
    state_dict = dict(state)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2)
    print(f"\n💾 Workflow state saved to: {filepath}")


def print_workflow_summary(state: AnalystSwarmState):
    """Print a detailed summary of the workflow execution."""
    print("\n" + "=" * 70)
    print("📋 WORKFLOW EXECUTION SUMMARY (WITH RETRY STATISTICS)")
    print("=" * 70)
    print("\n🤖 Agent Execution Status:")
    agents = [
        ("SEC Agent", state.get("sec_agent_status"), state.get("sec_agent_attempts", 0)),
        ("News Agent", state.get("news_agent_status"), state.get("news_agent_attempts", 0)),
        ("Social Agent", state.get("social_agent_status"), state.get("social_agent_attempts", 0)),
        ("Chart Agent", state.get("chart_agent_status"), state.get("chart_agent_attempts", 0)),
        ("Analyst Agent", state.get("analyst_agent_status"), state.get("analyst_agent_attempts", 0)),
        ("Governor Agent", state.get("governor_status"), state.get("governor_attempts", 0)),
        ("Risk Assessment", state.get("risk_status"), state.get("risk_attempts", 0)),
    ]

    total_attempts = 0
    for agent_name, status, attempts in agents:
        status_emoji = {"success": "✅", "failed": "❌", "pending": "⏳", "skipped": "⏭️"}.get(status, "❓")
        retry_info = (f"({attempts} attempt{'s' if attempts != 1 else ''})" if attempts > 1 else "")
        print(f"  {status_emoji} {agent_name}: {status} {retry_info}")
        total_attempts += attempts

    print(f"\n📊 Retry Statistics:")
    print(f"  Total Execution Attempts: {total_attempts}")
    print(f"  Average Attempts per Agent: {total_attempts / len(agents):.1f}")

    warnings = state.get("warnings", [])
    if warnings:
        print(f"\n⚠️ Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    errors = state.get("errors", [])
    if errors:
        print(f"\n❌ Errors Encountered ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ No errors encountered")

    print(f"\n🎯 Overall Status: {state.get('workflow_status', 'unknown').upper()}")
    print(f"⏱️ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


# --- CLI ENTRY POINT ---

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "TSLA"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python orchestrator.py <TICKER>\n")

    final_state = run_analyst_swarm(ticker, save_state=True)

    print("\n" + "=" * 70)
    print("📄 KEY OUTPUTS")
    print("=" * 70)
    print("\n1. Governor Summary:")
    print(final_state.get("governor_summary", "Not available"))
    print("\n2. Risk Assessment Summary:")
    print(final_state.get("risk_summary", "Not available"))
    print("\n" + "=" * 70)
    print("\n✅ All detailed reports saved to 'reports/' directory")
    print("💾 Workflow state saved to 'workflow_states/' directory")