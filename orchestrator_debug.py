"""
Orchestrator Debug Tool - Validates Agent Outputs and Workflow Integration

This script helps identify issues with:
1. Agent output format validation
2. Dictionary vs string return handling
3. Report extraction logic
4. Governor/Risk assessment input preparation

Usage:
    python orchestrator_debug.py <TICKER>
    
Example:
    python orchestrator_debug.py AAPL --test-mode
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")

def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")

def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")

def print_info(message: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")


# =============================================================================
# Agent Output Validators
# =============================================================================

def validate_agent_output_structure(agent_name: str, output: Any) -> Dict[str, Any]:
    """
    Validate the structure of an agent's output.
    
    Returns:
        Dict with validation results including:
        - is_valid: bool
        - output_type: str (dict, tuple, str, etc.)
        - has_summary: bool
        - has_detailed: bool
        - issues: List[str]
    """
    validation = {
        "agent_name": agent_name,
        "is_valid": True,
        "output_type": type(output).__name__,
        "has_summary": False,
        "has_detailed": False,
        "has_error": False,
        "issues": [],
        "keys_found": [],
    }
    
    # Check output type
    if isinstance(output, dict):
        validation["keys_found"] = list(output.keys())
        
        # Check for required fields
        if "summary_report" in output:
            validation["has_summary"] = True
        else:
            validation["is_valid"] = False
            validation["issues"].append("Missing 'summary_report' key")
        
        if "detailed_report" in output:
            validation["has_detailed"] = True
        else:
            validation["is_valid"] = False
            validation["issues"].append("Missing 'detailed_report' key")
        
        # Check for error field
        if "error" in output:
            validation["has_error"] = True
            validation["issues"].append(f"Error field present: {output['error']}")
        
        # Check if reports are empty strings
        if validation["has_summary"] and not output["summary_report"].strip():
            validation["is_valid"] = False
            validation["issues"].append("summary_report is empty")
        
        if validation["has_detailed"] and not output["detailed_report"].strip():
            validation["is_valid"] = False
            validation["issues"].append("detailed_report is empty")
    
    elif isinstance(output, tuple):
        validation["issues"].append(f"Output is tuple with {len(output)} elements")
        if len(output) == 2:
            validation["has_summary"] = bool(output[0])
            validation["has_detailed"] = bool(output[1])
            validation["issues"].append("⚠️  Agent returns tuple - orchestrator expects dict!")
        else:
            validation["is_valid"] = False
            validation["issues"].append(f"Unexpected tuple length: {len(output)}")
    
    elif isinstance(output, str):
        validation["is_valid"] = False
        validation["issues"].append("Output is plain string - should be dict with summary_report and detailed_report keys")
    
    else:
        validation["is_valid"] = False
        validation["issues"].append(f"Unexpected output type: {type(output)}")
    
    return validation


def test_agent_individually(agent_name: str, ticker: str) -> Dict[str, Any]:
    """
    Test a single agent and validate its output structure.
    
    Args:
        agent_name: Name of the agent (e.g., "news", "sec", "analyst")
        ticker: Stock ticker to analyze
    
    Returns:
        Dict with test results
    """
    print_info(f"Testing {agent_name.upper()} Agent for ${ticker}...")
    
    try:
        # Import the agent
        if agent_name == "news":
            from news_agent import run_news_agent
            output = run_news_agent(ticker, max_articles=5, lookback_days=7)
        
        elif agent_name == "sec":
            from sec_agent import run_sec_agent
            output = run_sec_agent(ticker, save_to_file=False)
        
        elif agent_name == "analyst":
            from analyst_agent import run_analyst_agent
            output = run_analyst_agent(ticker, save_to_file=False)
        
        elif agent_name == "social":
            from social_agent import run_social_agent
            output = run_social_agent(ticker, save_to_file=False)
        
        else:
            return {
                "success": False,
                "error": f"Unknown agent: {agent_name}"
            }
        
        # Validate the output
        validation = validate_agent_output_structure(agent_name, output)
        
        return {
            "success": True,
            "output": output,
            "validation": validation
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "validation": {
                "agent_name": agent_name,
                "is_valid": False,
                "issues": [f"Exception during execution: {str(e)}"]
            }
        }


def print_validation_report(validation: Dict[str, Any]):
    """Print a formatted validation report."""
    agent_name = validation.get("agent_name", "Unknown Agent")
    is_valid = validation.get("is_valid", False)
    
    if is_valid:
        print_success(f"{agent_name} output structure is VALID")
    else:
        print_error(f"{agent_name} output structure is INVALID")
    
    # Use .get() to avoid KeyErrors on exception
    print(f"  Output Type: {validation.get('output_type', 'N/A (check issues)')}")
    print(f"  Has summary_report: {validation.get('has_summary', 'N/A')}")
    print(f"  Has detailed_report: {validation.get('has_detailed', 'N/A')}")
    print(f"  Has error field: {validation.get('has_error', 'N/A')}")
    
    keys_found = validation.get("keys_found", [])
    if keys_found:
        print(f"  Keys Found: {', '.join(keys_found[:10])}")
        if len(keys_found) > 10:
            print(f"              ... and {len(keys_found) - 10} more")
    
    if validation.get("issues"):
        print(f"\n  Issues Found:")
        for issue in validation["issues"]:
            print(f"    - {issue}")        
# =============================================================================
# Orchestrator Node Validators
# =============================================================================

def test_orchestrator_node_extraction(agent_name: str, mock_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test how the orchestrator extracts reports from agent output.
    
    This simulates what happens in orchestrator.py node functions.
    """
    print_info(f"Testing orchestrator extraction for {agent_name}...")
    
    results = {
        "agent_name": agent_name,
        "extraction_method": None,
        "report_extracted": None,
        "status_set": None,
        "issues": []
    }
    
    # Simulate orchestrator extraction logic
    try:
        # Most nodes use: result.get("summary_report", str(result))
        if isinstance(mock_output, dict):
            results["extraction_method"] = "dict.get('summary_report', str(result))"
            report = mock_output.get("summary_report", str(mock_output))
            results["report_extracted"] = report[:200] + "..." if len(report) > 200 else report
            
            # Check if this would cause issues
            if "summary_report" not in mock_output:
                results["issues"].append("⚠️  'summary_report' key missing - will fallback to str(result)")
                results["issues"].append("    This may result in showing entire dict as string!")
        else:
            results["extraction_method"] = "str(result) - output not a dict"
            results["report_extracted"] = str(mock_output)[:200]
            results["issues"].append("❌ Agent output is not a dict - orchestrator expects dict!")
        
        # Check status setting
        if isinstance(mock_output, dict) and "error" not in mock_output:
            results["status_set"] = "success"
        else:
            results["status_set"] = "failed"
            results["issues"].append("Status will be set to 'failed' due to error field or wrong type")
    
    except Exception as e:
        results["issues"].append(f"Exception during extraction: {str(e)}")
    
    return results


def print_extraction_report(results: Dict[str, Any]):
    """Print extraction test results."""
    agent_name = results["agent_name"]
    
    print(f"\n  Extraction Method: {results['extraction_method']}")
    print(f"  Status Would Be: {results['status_set']}")
    
    if results["report_extracted"]:
        print(f"\n  Extracted Report Preview:")
        print(f"  {results['report_extracted']}")
    
    if results["issues"]:
        print(f"\n  Extraction Issues:")
        for issue in results["issues"]:
            print(f"    {issue}")


# =============================================================================
# Governor/Risk Assessment Input Validators
# =============================================================================

def validate_governor_input_preparation(agent_reports: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate that agent reports are properly formatted for Governor Agent.
    
    The Governor expects: Dict[str, str] mapping agent names to report strings.
    """
    validation = {
        "is_valid": True,
        "issues": [],
        "report_count": len(agent_reports),
        "empty_reports": [],
        "non_string_reports": []
    }
    
    for agent_name, report in agent_reports.items():
        # Check if report is a string
        if not isinstance(report, str):
            validation["is_valid"] = False
            validation["non_string_reports"].append({
                "agent": agent_name,
                "type": type(report).__name__
            })
            validation["issues"].append(f"{agent_name}: Report is {type(report).__name__}, not string")
        
        # Check if report is empty
        if isinstance(report, str) and not report.strip():
            validation["empty_reports"].append(agent_name)
            validation["issues"].append(f"{agent_name}: Report is empty string")
        
        # Check if report is error message
        if isinstance(report, str) and report.lower().startswith("error"):
            validation["issues"].append(f"{agent_name}: Report appears to be error message")
    
    return validation


def print_governor_validation(validation: Dict[str, Any]):
    """Print Governor input validation results."""
    if validation["is_valid"]:
        print_success(f"Governor input is VALID ({validation['report_count']} reports)")
    else:
        print_error(f"Governor input has ISSUES ({validation['report_count']} reports)")
    
    if validation["empty_reports"]:
        print_warning(f"Empty reports from: {', '.join(validation['empty_reports'])}")
    
    if validation["non_string_reports"]:
        print_error("Non-string reports detected:")
        for item in validation["non_string_reports"]:
            print(f"  - {item['agent']}: {item['type']}")
    
    if validation["issues"]:
        print("\n  Issues:")
        for issue in validation["issues"]:
            print(f"    - {issue}")


# =============================================================================
# Complete Workflow Test
# =============================================================================

def run_complete_workflow_test(ticker: str, agents_to_test: list = None):
    """
    Run a complete workflow test simulating the orchestrator.
    
    Args:
        ticker: Stock ticker to analyze
        agents_to_test: List of agent names, or None for all
    """
    if agents_to_test is None:
        agents_to_test = ["news", "analyst", "social", "sec"]
    
    print_section("COMPLETE WORKFLOW TEST")
    print_info(f"Testing workflow for ${ticker} with agents: {', '.join(agents_to_test)}")
    
    # Step 1: Test each agent individually
    print_section("STEP 1: Individual Agent Tests")
    
    agent_results = {}
    for agent_name in agents_to_test:
        result = test_agent_individually(agent_name, ticker)
        agent_results[agent_name] = result
        
        if result["success"]:
            print_validation_report(result["validation"])
            
            # Test orchestrator extraction
            print_section(f"Orchestrator Extraction Test: {agent_name.upper()}")
            extraction = test_orchestrator_node_extraction(agent_name, result["output"])
            print_extraction_report(extraction)
        else:
            print_error(f"Agent execution failed: {result.get('error', 'Unknown error')}")
            if "validation" in result:
                print_validation_report(result["validation"])
        
        print("\n" + "-"*80 + "\n")
    
    # Step 2: Prepare Governor input
    print_section("STEP 2: Governor Input Preparation")
    
    agent_reports = {}
    for agent_name, result in agent_results.items():
        if result["success"] and isinstance(result["output"], dict):
            # Simulate what orchestrator does
            report = result["output"].get("summary_report", str(result["output"]))
            agent_reports[f"{agent_name.capitalize()} Agent"] = report
        else:
            agent_reports[f"{agent_name.capitalize()} Agent"] = f"Error: Agent failed"
    
    governor_validation = validate_governor_input_preparation(agent_reports)
    print_governor_validation(governor_validation)
    
    # Step 3: Summary Report
    print_section("WORKFLOW SUMMARY")
    
    successful = sum(1 for r in agent_results.values() if r["success"])
    valid_outputs = sum(1 for r in agent_results.values() 
                       if r["success"] and r.get("validation", {}).get("is_valid", False))
    
    print(f"Agents Tested: {len(agents_to_test)}")
    print(f"Successful Executions: {successful}/{len(agents_to_test)}")
    print(f"Valid Output Structures: {valid_outputs}/{len(agents_to_test)}")
    
    if successful == len(agents_to_test) and valid_outputs == len(agents_to_test):
        print_success("\n🎉 ALL AGENTS PASSED! Workflow should work correctly.")
    elif successful == len(agents_to_test):
        print_warning("\n⚠️  All agents executed but some have invalid output structures.")
        print("   The orchestrator may have issues extracting reports.")
    else:
        print_error("\n❌ Some agents failed to execute. Check individual results above.")
    
    # Recommendations
    print_section("RECOMMENDATIONS")
    
    for agent_name, result in agent_results.items():
        if not result["success"]:
            print_error(f"{agent_name.upper()} Agent:")
            print(f"  Issue: {result.get('error', 'Unknown error')}")
            print(f"  Fix: Check agent implementation and dependencies")
            print()
        
        elif not result.get("validation", {}).get("is_valid", False):
            print_warning(f"{agent_name.upper()} Agent:")
            validation = result["validation"]
            
            if validation["output_type"] == "tuple":
                print(f"  Issue: Returns tuple instead of dict")
                print(f"  Fix: Change return statement from:")
                print(f"       return summary_report, detailed_report")
                print(f"       to:")
                print(f"       return {{'summary_report': summary_report, 'detailed_report': detailed_report, ...}}")
            
            elif not validation["has_summary"]:
                print(f"  Issue: Missing 'summary_report' key")
                print(f"  Fix: Add 'summary_report' key to return dictionary")
            
            elif not validation["has_detailed"]:
                print(f"  Issue: Missing 'detailed_report' key")
                print(f"  Fix: Add 'detailed_report' key to return dictionary")
            
            print()


# =============================================================================
# Main CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Debug tool for Analyst Swarm orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all agents for AAPL
  python orchestrator_debug.py AAPL
  
  # Test only news and analyst agents
  python orchestrator_debug.py TSLA --agents news analyst
  
  # Quick validation without full execution
  python orchestrator_debug.py NVDA --quick
        """
    )
    
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL, TSLA)")
    parser.add_argument("--agents", nargs="+", choices=["news", "sec", "analyst", "social"],
                       help="Specific agents to test (default: all)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick validation without full agent execution")
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    if args.quick:
        print_section("QUICK VALIDATION MODE")
        print_info("This will check agent files for common issues without full execution")
        
        # Check for return statement patterns
        agents = args.agents or ["news", "sec", "analyst", "social"]
        
        for agent_name in agents:
            filename = f"{agent_name}_agent.py"
            if not os.path.exists(filename):
                print_error(f"{filename} not found")
                continue
            
            print(f"\n{Colors.BOLD}Checking {filename}...{Colors.RESET}")
            
            with open(filename, 'r') as f:
                content = f.read()
            
            # Check for tuple returns in run_ functions
            if f"def run_{agent_name}_agent" in content:
                # Find the function
                start = content.find(f"def run_{agent_name}_agent")
                # Simple check for tuple return
                func_section = content[start:start+5000]
                
                if "return (" in func_section or "return summary" in func_section:
                    print_warning("  Possible tuple return detected")
                    print("  Check if function returns tuple instead of dict")
                
                if 'return {' in func_section and '"summary_report"' in func_section:
                    print_success("  Dictionary return with summary_report found")
                elif 'return {' in func_section:
                    print_warning("  Dictionary return found but check for summary_report key")
                else:
                    print_error("  No clear dictionary return found")
    else:
        run_complete_workflow_test(ticker, args.agents)


if __name__ == "__main__":
    main()