# SecAgent - Financial Analysis Tool

A class-based SEC filing analyzer that performs autonomous financial risk assessment and multi-year analysis using the SEC API and LangChain.

## Installation

```bash
pip install -U langchain==1.0.3 langchain-google-genai==0.1.* sec-api requests python-dotenv pydantic
```

## Environment Variables

Set up your API keys:

```bash
export SEC_API_KEY="your-sec-api-key"        # from sec-api.io
export GOOGLE_API_KEY="your-google-api-key"  # for Gemini
```

Or create a `.env` file:

```
SEC_API_KEY=your-sec-api-key
GOOGLE_API_KEY=your-google-api-key
```

## Usage

### Basic Usage

```python
from SecAgent import SecAgent

# Create an agent instance
agent = SecAgent(
    verbose=True,      # Enable detailed logging
    save_final=True,   # Save the final report to a file
    save_trace=True    # Save the execution trace/progress log
)

# Run analysis
report = agent.run(
    company="Apple Inc",
    company_description="A technology company that designs consumer electronics"
)

print(report)
```

### Quiet Mode (No Logging or File Saves)

```python
agent = SecAgent(verbose=False, save_final=False, save_trace=False)
report = agent.run("Microsoft Corporation")
```

### Analyze Multiple Companies

```python
companies = [
    ("Apple Inc", "Consumer electronics"),
    ("Microsoft Corporation", "Software and cloud computing"),
]

reports = {}
for company, description in companies:
    agent = SecAgent(verbose=True, save_final=True, save_trace=True)
    reports[company] = agent.run(company, description)
```

## Class Reference

### SecAgent

**Constructor Parameters:**

- `verbose` (bool, default=True): Whether to log detailed progress information
- `save_final` (bool, default=True): Whether to save the final report to a file
- `save_trace` (bool, default=True): Whether to save the execution trace to a file

**Public Methods:**

- `run(company: str, company_description: Optional[str] = None) -> str`
  - Analyzes the past 5 years of 10-K filings for a company
  - **Parameters:**
    - `company`: Company name (e.g., "Apple Inc")
    - `company_description`: Optional description to provide context to the AI
  - **Returns:** Comprehensive multi-year financial and risk analysis report as a string

## Output Files

When enabled, the agent creates the following files:

1. **Final Report** (`sec_report_<company>_<timestamp>.md`):
   - Comprehensive analysis report in Markdown format
   - Includes financial metrics table for all analyzed years
   
2. **Progress Trace** (`sec_agent_progress_<company>_<timestamp>.txt`):
   - Detailed log of all tool calls and intermediate results
   - Useful for debugging and understanding the agent's workflow

## Features

- Analyzes 5 years of 10-K SEC filings
- Extracts 40+ financial metrics using XBRL data
- Analyzes risk factors and their evolution over time
- Provides year-over-year comparative analysis
- Generates comprehensive markdown reports
- Configurable logging and file saving

## Metrics Analyzed

The agent automatically extracts and analyzes:

**Income Statement:**
- Revenue, Net Income, Gross Profit, Operating Income
- EBIT, Cost of Revenue, R&D Expense, SG&A Expense
- Interest Expense, Tax Expense, EPS (Basic & Diluted)

**Cash Flow Statement:**
- Operating/Investing/Financing Cash Flow
- Free Cash Flow, CapEx, D&A
- Dividends Paid, Stock Repurchases

**Balance Sheet:**
- Total Assets/Liabilities, Shareholders' Equity
- Cash & Equivalents, Current Assets/Liabilities
- Long-term Debt, Total Debt, PP&E
- Goodwill, Intangible Assets, and more

## Example Output Structure

```markdown
# SEC Multi-Year Analysis Report: [Company Name]
**Analysis Period:** [dates]
**Filings Analyzed:** 5 10-K filings

## Executive Summary
[Key findings]

## Financial Trends (5-Year Analysis)
[Trend analysis]

## Risk Assessment Evolution
[Risk analysis]

## Management Outlook Trends
[Management discussion]

## Year-over-Year Comparative Analysis
[Comparative analysis]

## Conclusion
[Final assessment]

## Financial Metrics Summary (5-Year)
[Comprehensive financial table]
```

## Advanced Configuration

To add or remove financial metrics, edit the `METRICS_CONFIG` dictionary in `SecAgent.py`. See the inline documentation for details.

## Troubleshooting

**Import errors:**
- Make sure all dependencies are installed
- Check that your Python environment has the required packages

**API errors:**
- Verify your SEC_API_KEY and GOOGLE_API_KEY are set correctly
- Check your API quotas/limits
- Ensure you have an active internet connection

**No financial data:**
- Some companies may not report certain metrics
- XBRL field names can vary between companies
- Check the progress trace file for detailed error messages

