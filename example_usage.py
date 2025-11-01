#!/usr/bin/env python3
"""
Example usage of the SecAgent class for financial analysis.

This demonstrates how to use the refactored SecAgent to analyze
SEC filings for any company.
"""

from SecAgent import SecAgent


def main():
    company_name = "Apple Inc"
    company_description = "A technology company that designs, manufactures, and markets consumer electronics, computer software, and online services."
    
    agent = SecAgent(
        verbose=True,      # Enable detailed logging
        save_final=True,   # Save the final report to a file
        save_trace=True    # Save the execution trace/progress log
    )
    
    report = agent.run(company_name, company_description)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nReport preview (first 500 chars):\n{report[:500]}...")


def example_quiet_mode():
    """Example showing how to run in quiet mode (no logging, no file saves)."""
    agent = SecAgent(verbose=False, save_final=False, save_trace=False)
    report = agent.run("Microsoft Corporation")
    return report


def example_custom_analysis():
    """Example showing how to analyze multiple companies."""
    companies = [
        ("Apple Inc", "Consumer electronics and software"),
        ("Microsoft Corporation", "Software, cloud computing, and hardware"),
        ("Alphabet Inc", "Internet services and technology"),
    ]
    
    reports = {}
    
    for company, description in companies:
        print(f"\n{'='*60}")
        print(f"Analyzing {company}...")
        print(f"{'='*60}\n")
        
        agent = SecAgent(verbose=True, save_final=True, save_trace=True)
        reports[company] = agent.run(company, description)
    
    return reports


if __name__ == "__main__":
    main()

