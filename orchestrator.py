"""
LangGraph Orchestrator - Analyst Swarm Workflow Manager

This orchestrator manages the entire analyst swarm workflow using LangGraph:
1. User inputs a stock ticker
2. Five specialist agents run in parallel
3. Governor agent synthesizes all reports
4. Risk Assessment agent produces final risk analysis

Environment Variables Required:
    - All agent-specific API keys (see individual agent files)
"""

import os
import json
from datetime import datetime
from typing import TypedDict, Annotated, Sequence
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

# --- STATE DEFINITION ---

class AnalystSwarmState(TypedDict):
    """
    The state object that flows through the entire workflow.
    Each agent adds its findings to this state.
    """
    # Input
    ticker: str
    timestamp: str
    
    # Specialist Agent Reports (raw outputs)
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
    
    # Governor Agent Output
    governor_summary: str
    governor_full_memo: str
    governor_status: str
    
    # Risk Assessment Output
    risk_summary: str
    risk_full_report: str
    risk_status: str
    
    # Final Status
    workflow_status: str
    errors: list


# --- AGENT NODE FUNCTIONS ---

def sec_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the SEC Agent (placeholder - implement when ready)."""
    print(f"\n🔍 Running SEC Agent for ${state['ticker']}...")
    
    try:
        # TODO: Implement SEC agent
        # from sec_agent import run_sec_agent
        # report = run_sec_agent(state['ticker'])
        
        # Placeholder for now
        report = f"""
**SEC Agent Report: ${state['ticker']}**

📄 **10-K Risk Factors Summary:**
* Regulatory compliance risks
* Market competition
* Operational challenges

*Note: Full SEC agent implementation pending*
        """
        
        state['sec_agent_report'] = report
        state['sec_agent_status'] = 'success'
        print("✅ SEC Agent completed")
        
    except Exception as e:
        print(f"❌ SEC Agent failed: {e}")
        state['sec_agent_report'] = f"Error: {str(e)}"
        state['sec_agent_status'] = 'failed'
        state['errors'].append(f"SEC Agent: {str(e)}")
    
    return state


def news_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the News Agent (placeholder - implement when ready)."""
    print(f"\n📰 Running News Agent for ${state['ticker']}...")
    
    try:
        # TODO: Implement News agent
        # from news_agent import run_news_agent
        # report = run_news_agent(state['ticker'])
        
        # Placeholder for now
        report = f"""
**News Agent Report: ${state['ticker']}**

📰 **Recent News Summary:**
* Industry developments
* Company announcements
* Market reactions

*Note: Full News agent implementation pending*
        """
        
        state['news_agent_report'] = report
        state['news_agent_status'] = 'success'
        print("✅ News Agent completed")
        
    except Exception as e:
        print(f"❌ News Agent failed: {e}")
        state['news_agent_report'] = f"Error: {str(e)}"
        state['news_agent_status'] = 'failed'
        state['errors'].append(f"News Agent: {str(e)}")
    
    return state


def social_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Social Sentiment Agent."""
    print(f"\n💬 Running Social Agent for ${state['ticker']}...")
    
    try:
        summary, detailed = run_social_agent(state['ticker'], save_to_file=True)
        state['social_agent_report'] = summary
        state['social_agent_status'] = 'success'
        print("✅ Social Agent completed")
        
    except Exception as e:
        print(f"❌ Social Agent failed: {e}")
        state['social_agent_report'] = f"Error: {str(e)}"
        state['social_agent_status'] = 'failed'
        state['errors'].append(f"Social Agent: {str(e)}")
    
    return state


def chart_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Chart/Technical Analysis Agent (placeholder - implement when ready)."""
    print(f"\n📊 Running Chart Agent for ${state['ticker']}...")
    
    try:
        # TODO: Implement Chart agent
        # from chart_agent import run_chart_agent
        # report = run_chart_agent(state['ticker'])
        
        # Placeholder for now
        report = f"""
**Chart Agent Report: ${state['ticker']}**

📈 **Technical Analysis:**
* Price trends and patterns
* Support/resistance levels
* Momentum indicators

*Note: Full Chart agent implementation pending*
        """
        
        state['chart_agent_report'] = report
        state['chart_agent_status'] = 'success'
        print("✅ Chart Agent completed")
        
    except Exception as e:
        print(f"❌ Chart Agent failed: {e}")
        state['chart_agent_report'] = f"Error: {str(e)}"
        state['chart_agent_status'] = 'failed'
        state['errors'].append(f"Chart Agent: {str(e)}")
    
    return state


def analyst_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Professional Analyst Ratings Agent."""
    print(f"\n📊 Running Analyst Agent for ${state['ticker']}...")
    
    try:
        report = run_analyst_agent(state['ticker'])
        state['analyst_agent_report'] = report
        state['analyst_agent_status'] = 'success'
        print("✅ Analyst Agent completed")
        
    except Exception as e:
        print(f"❌ Analyst Agent failed: {e}")
        state['analyst_agent_report'] = f"Error: {str(e)}"
        state['analyst_agent_status'] = 'failed'
        state['errors'].append(f"Analyst Agent: {str(e)}")
    
    return state


def governor_agent_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Governor Agent to synthesize all specialist reports."""
    print(f"\n🎯 Running Governor Agent for ${state['ticker']}...")
    
    try:
        # Collect all agent reports
        agent_reports = {
            "SEC Agent": state.get('sec_agent_report', 'Not available'),
            "News Agent": state.get('news_agent_report', 'Not available'),
            "Social Agent": state.get('social_agent_report', 'Not available'),
            "Chart Agent": state.get('chart_agent_report', 'Not available'),
            "Analyst Agent": state.get('analyst_agent_report', 'Not available')
        }
        
        summary, full_memo = run_governor_agent(
            state['ticker'], 
            agent_reports, 
            save_to_file=True
        )
        
        state['governor_summary'] = summary
        state['governor_full_memo'] = full_memo
        state['governor_status'] = 'success'
        print("✅ Governor Agent completed")
        
    except Exception as e:
        print(f"❌ Governor Agent failed: {e}")
        state['governor_summary'] = f"Error: {str(e)}"
        state['governor_full_memo'] = f"Error: {str(e)}"
        state['governor_status'] = 'failed'
        state['errors'].append(f"Governor Agent: {str(e)}")
    
    return state


def risk_assessment_node(state: AnalystSwarmState) -> AnalystSwarmState:
    """Run the Risk Assessment Agent on the Governor's memo."""
    print(f"\n⚠️ Running Risk Assessment Agent for ${state['ticker']}...")
    
    try:
        # Only run if Governor was successful
        if state.get('governor_status') == 'success':
            summary, full_report = run_risk_assessment_agent(
                state['ticker'],
                state['governor_full_memo'],
                save_to_file=True
            )
            
            state['risk_summary'] = summary
            state['risk_full_report'] = full_report
            state['risk_status'] = 'success'
            print("✅ Risk Assessment Agent completed")
        else:
            state['risk_summary'] = "Skipped: Governor Agent failed"
            state['risk_full_report'] = "Skipped: Governor Agent failed"
            state['risk_status'] = 'skipped'
            print("⚠️ Risk Assessment skipped due to Governor failure")
        
    except Exception as e:
        print(f"❌ Risk Assessment Agent failed: {e}")
        state['risk_summary'] = f"Error: {str(e)}"
        state['risk_full_report'] = f"Error: {str(e)}"
        state['risk_status'] = 'failed'
        state['errors'].append(f"Risk Assessment Agent: {str(e)}")
    
    return state


# --- WORKFLOW CONSTRUCTION ---

def create_analyst_swarm_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for the Analyst Swarm.
    
    Workflow Structure:
    1. START -> 5 parallel specialist agents
    2. Specialist agents -> Governor agent (join point)
    3. Governor agent -> Risk Assessment agent
    4. Risk Assessment agent -> END
    """
    
    # Create the graph
    workflow = StateGraph(AnalystSwarmState)
    
    # Add all agent nodes
    workflow.add_node("sec_agent", sec_agent_node)
    workflow.add_node("news_agent", news_agent_node)
    workflow.add_node("social_agent", social_agent_node)
    workflow.add_node("chart_agent", chart_agent_node)
    workflow.add_node("analyst_agent", analyst_agent_node)
    workflow.add_node("governor", governor_agent_node)
    workflow.add_node("risk_assessment", risk_assessment_node)
    
    # Set entry point - all specialist agents run in parallel
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
    Execute the complete Analyst Swarm workflow.
    
    Args:
        ticker: Stock ticker symbol (e.g., "TSLA", "AAPL")
        save_state: Whether to save the final state to a JSON file
    
    Returns:
        Final state object with all agent reports
    """
    print("\n" + "="*70)
    print("🚀 ANALYST SWARM - MULTI-AGENT ANALYSIS SYSTEM")
    print("="*70)
    print(f"📊 Target: ${ticker.upper()}")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
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
        governor_summary="",
        governor_full_memo="",
        governor_status="pending",
        risk_summary="",
        risk_full_report="",
        risk_status="pending",
        workflow_status="running",
        errors=[]
    )
    
    try:
        # Create and compile the workflow
        workflow = create_analyst_swarm_workflow()
        app = workflow.compile()
        
        # Execute the workflow
        print("⚙️ Executing workflow...\n")
        final_state = app.invoke(initial_state)
        
        # Update workflow status
        if final_state.get('errors'):
            final_state['workflow_status'] = 'completed_with_errors'
        else:
            final_state['workflow_status'] = 'completed_successfully'
        
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
        
        initial_state['workflow_status'] = 'failed'
        initial_state['errors'].append(f"Workflow error: {str(e)}")
        return initial_state


def save_workflow_state(ticker: str, state: AnalystSwarmState, output_dir: str = "workflow_states"):
    """Save the complete workflow state to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_workflow_state_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert state to JSON-serializable format
    state_dict = dict(state)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2)
    
    print(f"\n💾 Workflow state saved to: {filepath}")


def print_workflow_summary(state: AnalystSwarmState):
    """Print a summary of the workflow execution."""
    print("\n" + "="*70)
    print("📋 WORKFLOW EXECUTION SUMMARY")
    print("="*70)
    
    # Agent status overview
    print("\n🤖 Agent Execution Status:")
    agents = [
        ("SEC Agent", state.get('sec_agent_status')),
        ("News Agent", state.get('news_agent_status')),
        ("Social Agent", state.get('social_agent_status')),
        ("Chart Agent", state.get('chart_agent_status')),
        ("Analyst Agent", state.get('analyst_agent_status')),
        ("Governor Agent", state.get('governor_status')),
        ("Risk Assessment", state.get('risk_status'))
    ]
    
    for agent_name, status in agents:
        status_emoji = {
            'success': '✅',
            'failed': '❌',
            'pending': '⏳',
            'skipped': '⏭️'
        }.get(status, '❓')
        
        print(f"  {status_emoji} {agent_name}: {status}")
    
    # Error summary
    errors = state.get('errors', [])
    if errors:
        print(f"\n⚠️ Errors Encountered ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✅ No errors encountered")
    
    # Final status
    print(f"\n🎯 Overall Status: {state.get('workflow_status', 'unknown').upper()}")
    print(f"⏱️ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


# --- CLI ENTRY POINT ---

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "TSLA"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python orchestrator.py <TICKER>\n")
    
    # Run the complete workflow
    final_state = run_analyst_swarm(ticker, save_state=True)
    
    # Print key outputs
    print("\n" + "="*70)
    print("📄 KEY OUTPUTS")
    print("="*70)
    print("\n1. Governor Summary:")
    print(final_state.get('governor_summary', 'Not available'))
    print("\n2. Risk Assessment Summary:")
    print(final_state.get('risk_summary', 'Not available'))
    print("\n" + "="*70)
    print("\n✅ All detailed reports saved to 'reports/' directory")
    print("💾 Workflow state saved to 'workflow_states/' directory")