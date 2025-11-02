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

# --- STATE DEFINITION ---


class AnalystSwarmState(TypedDict):
    """The state object that flows through the entire workflow."""

    # Input
    ticker: str
    timestamp: str

    # Specialist Agent Reports
    sec_agent_report: str
    news_agent_report: str
    social_agent_report: str
    chart_agent_report: str
    analyst_agent_report: str

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
    governor_full_memo: str
    governor_status: str
    governor_attempts: int

    # Risk Assessment Output
    risk_summary: str
    risk_full_report: str
    risk_status: str
    risk_attempts: int

    # Final Status
    workflow_status: str
    errors: list
    warnings: list


# --- RETRY LOGIC DECORATOR ---


def with_retry(
    agent_name: str,
    max_retries: int = RETRY_CONFIG["max_retries"],
    base_delay: float = RETRY_CONFIG["base_delay"],
    max_delay: float = RETRY_CONFIG["max_delay"],
    exponential_base: float = RETRY_CONFIG["exponential_base"],
) -> Callable:
    """
    Decorator to add retry logic to agent functions.

    Args:
        agent_name: Name of the agent for logging
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff

    Returns:
        Decorated function with retry logic
    """

    def decorator(agent_func: Callable) -> Callable:
        def wrapper(state: AnalystSwarmState) -> AnalystSwarmState:
            attempts_key = f"{agent_name.lower().replace(' ', '_')}_attempts"
            current_attempts = state.get(attempts_key, 0)

            for attempt in range(max_retries + 1):
                try:
                    # Update attempt counter
                    state[attempts_key] = attempt + 1

                    if attempt > 0:
                        # Calculate delay with exponential backoff
                        delay = min(
                            base_delay * (exponential_base ** (attempt - 1)), max_delay
                        )

                        # Add jitter (randomness) to prevent thundering herd
                        if RETRY_CONFIG["jitter"]:
                            import random

                            delay = delay * (0.5 + random.random())

                        print(
                            f"⏳ {agent_name}: Retry attempt {attempt + 1}/{max_retries + 1} after {delay:.1f}s delay..."
                        )
                        time.sleep(delay)

                    # Execute the agent function
                    result_state = agent_func(state)

                    # Check if successful
                    status_key = f"{agent_name.lower().replace(' ', '_')}_status"
                    if result_state.get(status_key) == "success":
                        if attempt > 0:
                            print(
                                f"✅ {agent_name}: Succeeded on retry attempt {attempt + 1}"
                            )
                            # Add warning about retry
                            if "warnings" not in result_state:
                                result_state["warnings"] = []
                            result_state["warnings"].append(
                                f"{agent_name} required {attempt + 1} attempt(s) to succeed"
                            )
                        return result_state

                    # If not successful but no exception, treat as failure
                    if attempt == max_retries:
                        print(
                            f"❌ {agent_name}: Failed after {max_retries + 1} attempts (no exception)"
                        )
                        return result_state

                except Exception as e:
                    print(f"⚠️ {agent_name}: Attempt {attempt + 1} failed - {e}")

                    # If this was the last attempt, record error and return
                    if attempt == max_retries:
                        print(f"❌ {agent_name}: All retry attempts exhausted")
                        status_key = f"{agent_name.lower().replace(' ', '_')}_status"
                        report_key = f"{agent_name.lower().replace(' ', '_')}_report"

                        state[status_key] = "failed"
                        state[report_key] = (
                            f"Error after {max_retries + 1} attempts: {str(e)}"
                        )

                        if "errors" not in state:
                            state["errors"] = []
                        state["errors"].append(
                            f"{agent_name}: {str(e)} (after {max_retries + 1} attempts)"
                        )

                        return state

            return state

        return wrapper

    return decorator


# --- AGENT NODE FUNCTIONS WITH RETRY ---


@with_retry("SEC Agent")
def sec_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the SEC Agent."""
    print(f"\n🔍 Running SEC Agent for ${state['ticker']}...")

    try:
        from sec_agent import run_sec_agent

        result = run_sec_agent(state["ticker"], save_to_file=True)

        # Use the summary_report for the state
        report = result.get("summary_report", str(result))

        state["sec_agent_report"] = report
        state["sec_agent_status"] = "success"
        print("✅ SEC Agent completed")

    except Exception as e:
        print(f"❌ SEC Agent failed: {e}")
        state["sec_agent_report"] = f"Error: {str(e)}"
        state["sec_agent_status"] = "failed"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"SEC Agent: {str(e)}")

    return state


@with_retry("News Agent")
def news_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the News Agent."""
    print(f"\n📰 Running News Agent for ${state['ticker']}...")

    try:
        from news_agent import analyze_company_sentiment

        # Get company name from ticker (you might want to add a mapping)
        company_name = state["ticker"]

        result = analyze_company_sentiment(
            company_name=company_name, max_articles=30, lookback_days=7, verbose=True
        )

        if "error" in result:
            raise Exception(result["error"])

        # Format a summary from the result
        report = f"""
**News Agent Report: ${state['ticker']}**

📰 **Sentiment Analysis:**
* Overall Score: {result['weighted_sentiment_score']:.2f}
* Bullish Pressure: {result['bullish_pressure']*100:.1f}%
* Bearish Pressure: {result['bearish_pressure']*100:.1f}%
* Articles Analyzed: {result['total_articles_analyzed']}
* High-Impact Articles: {result['high_impact_count']}

{result['dashboard_summary']}
        """

        state["news_agent_report"] = report
        state["news_agent_status"] = "success"
        print("✅ News Agent completed")

    except Exception as e:
        print(f"❌ News Agent failed: {e}")
        state["news_agent_report"] = f"Error: {str(e)}"
        state["news_agent_status"] = "failed"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"News Agent: {str(e)}")

    return state


@with_retry("Social Agent")
def social_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Social Sentiment Agent with retry logic."""
    print(f"\n💬 Running Social Agent for ${state['ticker']}...")

    try:
        result = run_social_agent(state["ticker"], save_to_file=True)
        state["social_agent_report"] = result.get("summary_report", str(result))
        state["social_agent_status"] = "success"
        print("✅ Social Agent completed")

    except Exception as e:
        print(f"❌ Social Agent failed: {e}")
        state["social_agent_report"] = f"Error: {str(e)}"
        state["social_agent_status"] = "failed"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Social Agent: {str(e)}")

    return state


@with_retry("Chart Agent")
def chart_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
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

        state["chart_agent_report"] = report
        state["chart_agent_status"] = "success"
        print("✅ Chart Agent completed")

    except Exception as e:
        print(f"❌ Chart Agent failed: {e}")
        state["chart_agent_report"] = f"Error: {str(e)}"
        state["chart_agent_status"] = "failed"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Chart Agent: {str(e)}")

    return state


@with_retry("Analyst Agent")
def analyst_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Professional Analyst Ratings Agent with retry logic."""
    print(f"\n📊 Running Analyst Agent for ${state['ticker']}...")

    try:
        result = run_analyst_agent(state["ticker"], save_to_file=True)
        state["analyst_agent_report"] = result.get("summary_report", str(result))
        state["analyst_agent_status"] = "success"
        print("✅ Analyst Agent completed")

    except Exception as e:
        print(f"❌ Analyst Agent failed: {e}")
        state["analyst_agent_report"] = f"Error: {str(e)}"
        state["analyst_agent_status"] = "failed"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Analyst Agent: {str(e)}")

    return state


@with_retry("Governor Agent", max_retries=1)  # Governor gets fewer retries (expensive)
def governor_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Governor Agent to synthesize all specialist reports."""
    print(f"\n🎯 Running Governor Agent for ${state['ticker']}...")

    try:
        # Collect all agent reports
        agent_reports = {
            "SEC Agent": state.get("sec_agent_report", "Not available"),
            "News Agent": state.get("news_agent_report", "Not available"),
            "Social Agent": state.get("social_agent_report", "Not available"),
            "Chart Agent": state.get("chart_agent_report", "Not available"),
            "Analyst Agent": state.get("analyst_agent_report", "Not available"),
        }

        result = run_governor_agent(state["ticker"], agent_reports, save_to_file=True)

        state["governor_summary"] = result.get("summary_report", str(result))
        state["governor_full_memo"] = result.get("detailed_report", str(result))
        state["governor_status"] = "success"
        print("✅ Governor Agent completed")

    except Exception as e:
        print(f"❌ Governor Agent failed: {e}")
        state["governor_summary"] = f"Error: {str(e)}"
        state["governor_full_memo"] = f"Error: {str(e)}"
        state["governor_status"] = "failed"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Governor Agent: {str(e)}")

    return state


@with_retry("Risk Assessment Agent", max_retries=1)
def risk_assessment_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Risk Assessment Agent on the Governor's memo."""
    print(f"\n⚠️ Running Risk Assessment Agent for ${state['ticker']}...")

    try:
        # Only run if Governor was successful
        if state.get("governor_status") == "success":
            result = run_risk_assessment_agent(
                state["ticker"], state["governor_full_memo"], save_to_file=True
            )

            state["risk_summary"] = result.get("summary_report", str(result))
            state["risk_full_report"] = result.get("detailed_report", str(result))
            state["risk_status"] = "success"
            print("✅ Risk Assessment Agent completed")
        else:
            state["risk_summary"] = "Skipped: Governor Agent failed"
            state["risk_full_report"] = "Skipped: Governor Agent failed"
            state["risk_status"] = "skipped"
            print("⚠️ Risk Assessment skipped due to Governor failure")

    except Exception as e:
        print(f"❌ Risk Assessment Agent failed: {e}")
        state["risk_summary"] = f"Error: {str(e)}"
        state["risk_full_report"] = f"Error: {str(e)}"
        state["risk_status"] = "failed"
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Risk Assessment Agent: {str(e)}")

    return state


# --- WORKFLOW CONSTRUCTION ---


def create_analyst_swarm_workflow() -> StateGraph:
    """Create the LangGraph workflow with retry-enabled agents."""

    workflow = StateGraph(AnalystSwarmState)

    # Add all agent nodes (now with retry logic)
    workflow.add_node("sec_agent", sec_agent_node)
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("social_agent", social_agent_node)
    workflow.add_node("chart_agent", chart_agent_node)
    workflow.add_node("analyst_agent", analyst_agent_node)
    workflow.add_node("governor", governor_agent_node)
    workflow.add_node("risk_assessment", risk_assessment_node)

    # Set entry points - all specialist agents run in parallel
    workflow.set_entry_point("sec_agent")
    workflow.set_entry_point("news_agent")
    workflow.set_entry_point("social_agent")
    workflow.set_entry_point("chart_agent")
    workflow.set_entry_point("analyst_agent")

    # All specialist agents converge to Governor
    workflow.add_edge("sec_agent", "governor")
    workflow.add_edge("news_agent", "governor")
    workflow.add_edge("social_agent", "governor")
    workflow.add_edge("chart_agent", "governor")
    workflow.add_edge("analyst_agent", "governor")

    # Governor flows to Risk Assessment
    workflow.add_edge("governor", "risk_assessment")

    # Risk Assessment is the final node
    workflow.add_edge("risk_assessment", END)

    return workflow


# --- MAIN ORCHESTRATION FUNCTION ---


def run_analyst_swarm(ticker: str, save_state: bool = True) -> AnalystSwarmState:
    """
    Execute the complete Analyst Swarm workflow with retry logic.

    Args:
        ticker: Stock ticker symbol
        save_state: Whether to save the final state to a JSON file

    Returns:
        Final state object with all agent reports
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

    # Initialize state
    initial_state = AnalystSwarmState(
        ticker=ticker.upper(),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sec_agent_report="",
        news_agent_report="",
        social_agent_report="",
        chart_agent_report="",
        analyst_agent_report="",
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
        # Create and compile the workflow
        workflow = create_analyst_swarm_workflow()
        app = workflow.compile()

        # Execute the workflow
        print("⚙️ Executing workflow with retry logic...\n")
        final_state = app.invoke(initial_state)

        # Update workflow status
        if final_state.get("errors"):
            final_state["workflow_status"] = "completed_with_errors"
        else:
            final_state["workflow_status"] = "completed_successfully"

        # Save state to file
        if save_state:
            save_workflow_state(ticker, final_state)

        # Print summary
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

    # Convert state to JSON-serializable format
    state_dict = dict(state)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state_dict, f, indent=2)

    print(f"\n💾 Workflow state saved to: {filepath}")


def print_workflow_summary(state: AnalystSwarmState):
    """Print a detailed summary of the workflow execution."""
    print("\n" + "=" * 70)
    print("📋 WORKFLOW EXECUTION SUMMARY (WITH RETRY STATISTICS)")
    print("=" * 70)

    # Agent status overview with retry counts
    print("\n🤖 Agent Execution Status:")
    agents = [
        (
            "SEC Agent",
            state.get("sec_agent_status"),
            state.get("sec_agent_attempts", 0),
        ),
        (
            "News Agent",
            state.get("news_agent_status"),
            state.get("news_agent_attempts", 0),
        ),
        (
            "Social Agent",
            state.get("social_agent_status"),
            state.get("social_agent_attempts", 0),
        ),
        (
            "Chart Agent",
            state.get("chart_agent_status"),
            state.get("chart_agent_attempts", 0),
        ),
        (
            "Analyst Agent",
            state.get("analyst_agent_status"),
            state.get("analyst_agent_attempts", 0),
        ),
        (
            "Governor Agent",
            state.get("governor_status"),
            state.get("governor_attempts", 0),
        ),
        ("Risk Assessment", state.get("risk_status"), state.get("risk_attempts", 0)),
    ]

    total_attempts = 0
    for agent_name, status, attempts in agents:
        status_emoji = {
            "success": "✅",
            "failed": "❌",
            "pending": "⏳",
            "skipped": "⏭️",
        }.get(status, "❓")

        retry_info = (
            f"({attempts} attempt{'s' if attempts != 1 else ''})"
            if attempts > 1
            else ""
        )
        print(f"  {status_emoji} {agent_name}: {status} {retry_info}")
        total_attempts += attempts

    print(f"\n📊 Retry Statistics:")
    print(f"  Total Execution Attempts: {total_attempts}")
    print(f"  Average Attempts per Agent: {total_attempts / len(agents):.1f}")

    # Warning summary
    warnings = state.get("warnings", [])
    if warnings:
        print(f"\n⚠️ Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")

    # Error summary
    errors = state.get("errors", [])
    if errors:
        print(f"\n❌ Errors Encountered ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ No errors encountered")

    # Final status
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

    # Run the complete workflow
    final_state = run_analyst_swarm(ticker, save_state=True)

    # Print key outputs
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
    print(
        f"🔄 Retry logic handled {sum([final_state.get(f'{agent}_attempts', 0) for agent in ['sec_agent', 'news_agent', 'social_agent', 'chart_agent', 'analyst_agent', 'governor', 'risk']])} total attempts"
    )
