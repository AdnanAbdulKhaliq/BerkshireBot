#!/usr/bin/env python3
"""
SEC Agent - Autonomous Financial Risk Assessment (LangChain v1.0.3)
Rewritten to use **SEC API by D2V (sec-api.io)** instead of direct data.sec.gov calls.

- Uses `sec-api` Python SDK: QueryApi, ExtractorApi, XbrlApi
- API key is read from env var: **SEC_API_KEY**
- LLM: Gemini via langchain-google-genai (GOOGLE_API_KEY)

Install:
  pip install -U langchain==1.0.3 langchain-google-genai==0.1.* sec-api requests

Usage:
  from sec_agent import SecAgent, run_sec_agent

  # Method 1: Using the wrapper function (recommended)
  summary, detailed = run_sec_agent("AAPL", save_to_file=True)

  # Method 2: Using the class directly
  agent = SecAgent(verbose=True, save_final=True, save_trace=True)
  report = agent.run("Apple Inc", "A technology company that designs consumer electronics")

Notes:
- Latest filing discovery uses the **Filing Query API** (sec-api) and is far more reliable than scraping EDGAR endpoints.
- Section extraction uses the **Extractor API** for 10-K/10-Q items (e.g., 1A Risk Factors, 7 MD&A).
- Financial metrics (revenue, net income, margin) come from **XBRL-to-JSON Converter API** instead of regexing raw text.
"""

from __future__ import annotations

import os
import json
import re
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# LangChain v1 APIs
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# SEC API (D2V) SDK
from sec_api import QueryApi, ExtractorApi, XbrlApi

# ============================================================================
# Metrics Configuration - Edit this to add/remove metrics
# ============================================================================
#
# To add a new metric:
#   1. Add a new entry to METRICS_CONFIG with:
#      - A unique key (lowercase, underscore-separated)
#      - display_name: How it appears in the table
#      - xbrl_fields: List of XBRL field names to try (in priority order)
#      - statement: Which statement to look in ("income", "cash_flow", or "balance")
#
# To remove a metric:
#   - Simply delete or comment out the entire entry
#
# Example new metric:
# "total_assets": {
#     "display_name": "Total Assets",
#     "xbrl_fields": ["Assets"],
#     "statement": "balance"
# },
#
# Common XBRL fields:
#   Income: Revenues, OperatingIncomeLoss, GrossProfit, EBIT, EBITDA
#   Cash Flow: PaymentsToAcquirePropertyPlantAndEquipment, OperatingCashFlow
#   Balance: Assets, Liabilities, StockholdersEquity, CashAndCashEquivalents
# ============================================================================

METRICS_CONFIG = {
    "revenue": {
        "display_name": "Revenue",
        "xbrl_fields": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ],
        "statement": "income",
    },
    "net_income": {
        "display_name": "Net Income",
        "xbrl_fields": ["NetIncomeLoss"],
        "statement": "income",
    },
    # "ebitda": {
    #     "display_name": "EBITDA",
    #     "xbrl_fields": ["EBITDA"],
    #     "statement": "income"
    # },
    "capex": {
        "display_name": "CapEx",
        "xbrl_fields": [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsForCapitalImprovements",
        ],
        "statement": "cash_flow",
    },
    # ========== Additional Income Statement Metrics ==========
    "gross_profit": {
        "display_name": "Gross Profit",
        "xbrl_fields": ["GrossProfit"],
        "statement": "income",
    },
    "operating_income": {
        "display_name": "Operating Income",
        "xbrl_fields": ["OperatingIncomeLoss"],
        "statement": "income",
    },
    "ebit": {
        "display_name": "EBIT",
        "xbrl_fields": ["EBIT", "OperatingIncomeLoss"],
        "statement": "income",
    },
    "cost_of_revenue": {
        "display_name": "Cost of Revenue",
        "xbrl_fields": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
        "statement": "income",
    },
    "rd_expense": {
        "display_name": "R&D Expense",
        "xbrl_fields": ["ResearchAndDevelopmentExpense"],
        "statement": "income",
    },
    "sga_expense": {
        "display_name": "SG&A Expense",
        "xbrl_fields": ["SellingGeneralAndAdministrativeExpense"],
        "statement": "income",
    },
    "interest_expense": {
        "display_name": "Interest Expense",
        "xbrl_fields": ["InterestExpense"],
        "statement": "income",
    },
    "tax_expense": {
        "display_name": "Tax Expense",
        "xbrl_fields": ["IncomeTaxExpenseBenefit"],
        "statement": "income",
    },
    "eps_basic": {
        "display_name": "EPS (Basic)",
        "xbrl_fields": ["EarningsPerShareBasic"],
        "statement": "income",
    },
    "eps_diluted": {
        "display_name": "EPS (Diluted)",
        "xbrl_fields": ["EarningsPerShareDiluted"],
        "statement": "income",
    },
    # ========== Cash Flow Statement Metrics ==========
    "operating_cash_flow": {
        "display_name": "Operating Cash Flow",
        "xbrl_fields": ["NetCashProvidedByUsedInOperatingActivities"],
        "statement": "cash_flow",
    },
    "investing_cash_flow": {
        "display_name": "Investing Cash Flow",
        "xbrl_fields": ["NetCashProvidedByUsedInInvestingActivities"],
        "statement": "cash_flow",
    },
    "financing_cash_flow": {
        "display_name": "Financing Cash Flow",
        "xbrl_fields": ["NetCashProvidedByUsedInFinancingActivities"],
        "statement": "cash_flow",
    },
    "free_cash_flow": {
        "display_name": "Free Cash Flow",
        "xbrl_fields": ["FreeCashFlow"],
        "statement": "cash_flow",
    },
    "dividends_paid": {
        "display_name": "Dividends Paid",
        "xbrl_fields": ["PaymentsOfDividends"],
        "statement": "cash_flow",
    },
    "stock_repurchases": {
        "display_name": "Stock Repurchases",
        "xbrl_fields": [
            "PaymentsForRepurchaseOfCommonStock",
            "PaymentsForRepurchaseOfEquity",
        ],
        "statement": "cash_flow",
    },
    "depreciation_amortization": {
        "display_name": "D&A",
        "xbrl_fields": ["DepreciationDepletionAndAmortization", "Depreciation"],
        "statement": "cash_flow",
    },
    # ========== Balance Sheet Metrics ==========
    "total_assets": {
        "display_name": "Total Assets",
        "xbrl_fields": ["Assets"],
        "statement": "balance",
    },
    "total_liabilities": {
        "display_name": "Total Liabilities",
        "xbrl_fields": ["Liabilities"],
        "statement": "balance",
    },
    "stockholders_equity": {
        "display_name": "Shareholders' Equity",
        "xbrl_fields": ["StockholdersEquity"],
        "statement": "balance",
    },
    "cash_and_equivalents": {
        "display_name": "Cash & Equivalents",
        "xbrl_fields": ["CashAndCashEquivalentsAtCarryingValue", "Cash"],
        "statement": "balance",
    },
    "short_term_investments": {
        "display_name": "Short-term Investments",
        "xbrl_fields": ["ShortTermInvestments"],
        "statement": "balance",
    },
    "accounts_receivable": {
        "display_name": "Accounts Receivable",
        "xbrl_fields": ["AccountsReceivableNetCurrent"],
        "statement": "balance",
    },
    "inventory": {
        "display_name": "Inventory",
        "xbrl_fields": ["InventoryNet"],
        "statement": "balance",
    },
    "current_assets": {
        "display_name": "Current Assets",
        "xbrl_fields": ["AssetsCurrent"],
        "statement": "balance",
    },
    "current_liabilities": {
        "display_name": "Current Liabilities",
        "xbrl_fields": ["LiabilitiesCurrent"],
        "statement": "balance",
    },
    "long_term_debt": {
        "display_name": "Long-term Debt",
        "xbrl_fields": ["LongTermDebtNoncurrent", "LongTermDebt"],
        "statement": "balance",
    },
    "total_debt": {
        "display_name": "Total Debt",
        "xbrl_fields": ["DebtCurrent", "LongTermDebt"],
        "statement": "balance",
    },
    "accounts_payable": {
        "display_name": "Accounts Payable",
        "xbrl_fields": ["AccountsPayableCurrent"],
        "statement": "balance",
    },
    "retained_earnings": {
        "display_name": "Retained Earnings",
        "xbrl_fields": ["RetainedEarningsAccumulatedDeficit"],
        "statement": "balance",
    },
    "property_plant_equipment": {
        "display_name": "PP&E (Net)",
        "xbrl_fields": ["PropertyPlantAndEquipmentNet"],
        "statement": "balance",
    },
    "intangible_assets": {
        "display_name": "Intangible Assets",
        "xbrl_fields": ["IntangibleAssetsNetExcludingGoodwill"],
        "statement": "balance",
    },
    "goodwill": {
        "display_name": "Goodwill",
        "xbrl_fields": ["Goodwill"],
        "statement": "balance",
    },
}


def get_metrics_documentation() -> str:
    """Generate documentation about configured metrics."""
    doc = "Configured Financial Metrics:\n"
    for key, config in METRICS_CONFIG.items():
        doc += f"  - {config['display_name']} ({key}): XBRL fields = {config['xbrl_fields']}, Statement = {config['statement']}\n"
    return doc


# ============================================================================
# Pydantic Models for Financial Statistics
# ============================================================================


class FinancialYearStats(BaseModel):
    year: str = Field(..., description="Fiscal year (e.g., '2023')")
    metrics: Dict[str, Optional[float]] = Field(
        default_factory=dict, description="Dictionary of metric_key -> value"
    )

    def get_metric(self, metric_key: str) -> Optional[float]:
        """Get a metric value by key."""
        return self.metrics.get(metric_key)

    def set_metric(self, metric_key: str, value: Optional[float]) -> None:
        """Set a metric value by key."""
        self.metrics[metric_key] = value


class FinancialStats(BaseModel):
    years: List[FinancialYearStats] = Field(
        default_factory=list, description="List of financial stats by year"
    )

    def add_year(self, year_stats: FinancialYearStats) -> None:
        """Add or update stats for a year."""
        for idx, existing in enumerate(self.years):
            if existing.year == year_stats.year:
                self.years[idx] = year_stats
                return
        self.years.append(year_stats)
        self.years.sort(key=lambda x: x.year, reverse=True)

    def to_markdown_table(self) -> str:
        """Convert financial stats to a markdown table using METRICS_CONFIG."""
        if not self.years:
            return "No financial data available."

        metric_keys = list(METRICS_CONFIG.keys())
        header_cols = ["Year"] + [
            METRICS_CONFIG[k]["display_name"] for k in metric_keys
        ]
        header = "| " + " | ".join(header_cols) + " |\n"
        separator = "|" + "|".join(["------" for _ in header_cols]) + "|\n"

        def format_number(val: Optional[float]) -> str:
            if val is None:
                return "N/A"
            if abs(val) >= 1e9:
                return f"${val/1e9:.2f}B"
            elif abs(val) >= 1e6:
                return f"${val/1e6:.2f}M"
            else:
                return f"${val:,.0f}"

        rows = []
        for year_stat in self.years:
            row_values = [year_stat.year] + [
                format_number(year_stat.get_metric(k)) for k in metric_keys
            ]
            row = "| " + " | ".join(row_values) + " |\n"
            rows.append(row)

        return header + separator + "".join(rows)


# ============================================================================
# Configuration
# ============================================================================

USER_AGENT = "SEC-Agent/1.0 (your-email@example.com)"  # not required for sec-api, but used for direct sec.gov fetches if needed

SEC_API_KEY = os.environ.get("SEC_API_KEY", "")  # <- REQUIRED (sec-api.io)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")  # <- for Gemini

# ============================================================================
# Helpers: SEC API Clients
# ============================================================================


def _require_keys() -> None:
    if not SEC_API_KEY:
        raise RuntimeError("ERROR: SEC_API_KEY environment variable not set!")
    if not GOOGLE_API_KEY:
        raise RuntimeError("ERROR: GOOGLE_API_KEY environment variable not set!")


def _sec_clients() -> Tuple[QueryApi, ExtractorApi, XbrlApi]:
    """Init and return SEC API clients."""
    _require_keys()
    return (
        QueryApi(api_key=SEC_API_KEY),
        ExtractorApi(SEC_API_KEY),
        XbrlApi(SEC_API_KEY),
    )


# ============================================================================
# SEC API Functions (D2V)
# ============================================================================


def get_company_cik(company_name: str) -> Dict:
    """
    Resolve a company name to CIK/ticker via **Mapping API**.
    Returns a dict with cik, company_name, tickers (best match first).
    """
    logger.info(f"🔍 Looking up company CIK for: '{company_name}'")
    _require_keys()
    url = f"https://api.sec-api.io/mapping/name/{quote(company_name)}"
    headers = {"Authorization": SEC_API_KEY}
    logger.debug(f"Making API request to: {url}")
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    arr = resp.json() or []
    if not arr:
        logger.warning(f"No company found for: {company_name}")
        return {"error": f"No company found for: {company_name}"}
    # Pick the top result; could be refined by exact match heuristics
    top = arr[0]
    result = {
        "cik": str(top.get("cik", "")),
        "company_name": top.get("name", company_name),
        "tickers": [t for t in [top.get("ticker")] if t],
    }
    logger.info(f"✅ Found CIK: {result['cik']}, Tickers: {result['tickers']}")
    return result


def get_latest_filing(cik: str, form_type: str = "10-K") -> Dict:
    """
    Find the most recent filing of the given form type using **Query API**.
    Returns dict with metadata + the SEC HTML/TXT URLs.
    """
    logger.info(f"📄 Fetching latest {form_type} filing for CIK: {cik}")
    queryApi, *_ = _sec_clients()

    # Query syntax per sec-api docs
    query = f'cik:{int(cik)} AND formType:"{form_type}"'
    params = {
        "query": query,
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    logger.debug(f"Query: {query}")
    res = queryApi.get_filings(params)
    filings = res.get("filings", [])
    if not filings:
        logger.warning(f"No {form_type} filing found for CIK {cik}")
        return {"error": f"No {form_type} filing found for CIK {cik}"}
    f = filings[0]
    htm_url = f.get("linkToFilingDetails")
    txt_url = f.get("linkToTxt")
    filing_date = (f.get("filedAt") or "")[:10]
    logger.info(
        f"✅ Found {form_type} filed on {filing_date}, Accession: {f.get('accessionNo')}"
    )
    return {
        "form_type": f.get("formType"),
        "filing_date": filing_date,
        "url_htm": htm_url,
        "url_txt": txt_url,
        "accession": f.get("accessionNo"),
        "cik": f.get("cik"),
        "ticker": f.get("ticker"),
        "company_name": f.get("companyName"),
    }


def get_multiple_filings(
    cik: str, form_type: str = "10-K", count: int = 5
) -> List[Dict]:
    """
    Find the most recent N filings of the given form type using **Query API**.
    Returns list of dicts with metadata + the SEC HTML/TXT URLs.
    """
    logger.info(f"📄 Fetching last {count} {form_type} filings for CIK: {cik}")
    queryApi, *_ = _sec_clients()

    query = f'cik:{int(cik)} AND formType:"{form_type}"'
    params = {
        "query": query,
        "from": "0",
        "size": str(count),
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    logger.debug(f"Query: {query}")
    res = queryApi.get_filings(params)
    filings = res.get("filings", [])
    if not filings:
        logger.warning(f"No {form_type} filings found for CIK {cik}")
        return []

    results = []
    for f in filings:
        htm_url = f.get("linkToFilingDetails")
        txt_url = f.get("linkToTxt")
        filing_date = (f.get("filedAt") or "")[:10]
        results.append(
            {
                "form_type": f.get("formType"),
                "filing_date": filing_date,
                "url_htm": htm_url,
                "url_txt": txt_url,
                "accession": f.get("accessionNo"),
                "cik": f.get("cik"),
                "ticker": f.get("ticker"),
                "company_name": f.get("companyName"),
            }
        )

    logger.info(
        f"✅ Found {len(results)} {form_type} filings, dates: {[r['filing_date'] for r in results]}"
    )
    return results


def extractor_item_code(form_type: str, section_name: str) -> Optional[str]:
    """Map human section name -> Extractor API item code per form type."""
    sec = section_name.lower()
    if form_type.startswith("10-K") or form_type == "10-K":
        mapping = {
            "risk factors": "1A",
            "management discussion": "7",
            "management’s discussion": "7",
            "management's discussion": "7",
            "financial statements": "8",
        }
    elif form_type.startswith("10-Q") or form_type == "10-Q":
        mapping = {
            "risk factors": "part2item1a",
            "management discussion": "part1item2",
            "management’s discussion": "part1item2",
            "management's discussion": "part1item2",
            "financial statements": "part1item1",
        }
    else:
        mapping = {}
    return mapping.get(sec)


def extract_section_by_url(
    filing_url_htm: str, form_type: str, section_name: str, return_type: str = "text"
) -> str:
    """Use **Extractor API** to pull a specific section from a filing URL."""
    logger.info(f"📖 Extracting section '{section_name}' from {form_type}")
    _, extractorApi, _ = _sec_clients()
    code = extractor_item_code(form_type, section_name)
    if not code:
        logger.warning(f"Section mapping unsupported for {form_type}: '{section_name}'")
        return f"Section mapping unsupported for {form_type}: '{section_name}'."
    try:
        logger.debug(f"Using item code: {code}")
        result = extractorApi.get_section(filing_url_htm, code, return_type)
        result_len = len(result) if result else 0
        logger.info(f"✅ Extracted section (length: {result_len} chars)")
        return result
    except Exception as e:
        logger.error(f"Extractor API error: {e}")
        return f"Extractor API error: {e}"


def xbrl_metrics(
    accession_no: Optional[str] = None, filing_url_htm: Optional[str] = None
) -> Dict:
    """
    Pull structured financials via **XbrlApi** using METRICS_CONFIG. Returns a dict with all configured metrics.
    """
    logger.info(
        f"💰 Extracting XBRL financial metrics (accession: {accession_no or 'None'})"
    )
    *_, xbrlApi = _sec_clients()
    if not accession_no and not filing_url_htm:
        logger.error("xbrl_metrics requires accession_no or filing_url_htm")
        return {"error": "xbrl_metrics requires accession_no or filing_url_htm"}

    try:
        if accession_no:
            logger.debug(f"Using accession number: {accession_no}")
            x = xbrlApi.xbrl_to_json(accession_no=accession_no)
        else:
            logger.debug("Using filing URL")
            x = xbrlApi.xbrl_to_json(htm_url=filing_url_htm)
    except Exception as e:
        logger.error(f"XBRL conversion error: {e}")
        return {"error": f"XBRL conversion error: {e}"}

    def _pick_latest(arr: List[dict]) -> Optional[dict]:
        if not arr:
            return None

        def end_date(v: dict) -> str:
            p = v.get("period", {})
            return p.get("endDate") or p.get("instant") or ""

        return max(arr, key=end_date)

    def _val(v: dict) -> Optional[float]:
        try:
            return float(v.get("value"))
        except Exception:
            return None

    statements = {
        "income": x.get("StatementsOfIncome", {}),
        "cash_flow": x.get("StatementsOfCashFlows", {}),
        "balance": x.get("BalanceSheets", {}),
    }

    result = {}
    period_end = None

    for metric_key, metric_config in METRICS_CONFIG.items():
        statement_type = metric_config["statement"]
        xbrl_fields = metric_config["xbrl_fields"]
        statement = statements.get(statement_type, {})

        latest_value = None
        for field in xbrl_fields:
            data = _pick_latest(statement.get(field, []))
            if data:
                latest_value = _val(data)
                if period_end is None and data.get("period"):
                    period_end = data["period"].get("endDate") or data["period"].get(
                        "instant"
                    )
                break

        result[metric_key] = latest_value
        logger.debug(f"  {metric_key}: {latest_value}")

    result["period_end"] = period_end

    if (
        result.get("revenue")
        and result.get("net_income") is not None
        and result["revenue"] != 0
    ):
        result["profit_margin"] = result["net_income"] / result["revenue"]
    else:
        result["profit_margin"] = None

    logger.info(
        f"✅ XBRL metrics extracted: {', '.join([f'{k}: {v}' for k, v in result.items() if k not in ['period_end', 'profit_margin']])}"
    )
    return result


# ============================================================================
# SEC Agent Class
# ============================================================================


class SecAgent:
    """
    SEC Agent for autonomous financial risk assessment and multi-year analysis.

    Args:
        verbose: Whether to log detailed progress information
        save_final: Whether to save the final report to a file
        save_trace: Whether to save the execution trace to a file
    """

    def __init__(
        self, verbose: bool = True, save_final: bool = True, save_trace: bool = True
    ):
        self.verbose = verbose
        self.save_final = save_final
        self.save_trace = save_trace

        self.current_filings: List[Dict] = []
        self.financial_stats: FinancialStats = FinancialStats()
        self.progress_file_path: Optional[str] = None

        self.logger = logging.getLogger(f"{__name__}.SecAgent")
        if not self.verbose:
            self.logger.setLevel(logging.WARNING)
        else:
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        _require_keys()

    def _log_progress(self, message: str) -> None:
        """Write a progress message to the progress tracking file."""
        if self.save_trace and self.progress_file_path:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.progress_file_path, "a", encoding="utf-8") as f:
                    f.write(f"---\n{timestamp}\n{message}\n")
            except Exception as e:
                if self.verbose:
                    self.logger.error(f"Failed to write to progress file: {e}")

    def _lookup_company_cik(self, company_name: str) -> str:
        """Use D2V **Mapping API** to resolve company name to CIK/ticker.

        Args:
            company_name: e.g. "Apple Inc".
        Returns:
            JSON string: { cik, company_name, tickers[] } or { error }
        """
        if self.verbose:
            self.logger.info(
                f"🔧 TOOL CALLED: lookup_company_cik(company_name='{company_name}')"
            )
        self._log_progress(
            f"Tool Call: lookup_company_cik\nInput: company_name='{company_name}'"
        )

        result = get_company_cik(company_name)
        result_str = json.dumps(result, indent=2)

        if self.verbose:
            self.logger.info(f"🔧 TOOL RESULT: lookup_company_cik -> {result}")
        self._log_progress(f"Tool Result: lookup_company_cik\n{result_str}")

        return result_str

    def _fetch_latest_filings(self, input_str: str) -> str:
        """Use **Query API** to fetch the latest 5 years of filings and cache them.

        Input: "CIK|FORM_TYPE|COUNT" (e.g., "0000320193|10-K|5"). FORM_TYPE defaults to 10-K, COUNT defaults to 5.
        Returns JSON array with filing metadata for each filing.
        """
        if self.verbose:
            self.logger.info(
                f"🔧 TOOL CALLED: fetch_latest_filings(input='{input_str}')"
            )
        self._log_progress(f"Tool Call: fetch_latest_filings\nInput: '{input_str}'")

        try:
            parts = input_str.split("|")
            cik = parts[0].strip()
            form_type = parts[1].strip() if len(parts) > 1 else "10-K"
            count = int(parts[2].strip()) if len(parts) > 2 else 5
            if self.verbose:
                self.logger.info(
                    f"Parsed: CIK={cik}, form_type={form_type}, count={count}"
                )
            filings = get_multiple_filings(cik, form_type, count)
            if filings:
                self.current_filings = filings
                if self.verbose:
                    self.logger.info(f"📝 {len(filings)} filings cached")
                    self.logger.info(
                        f"🔧 TOOL RESULT: fetch_latest_filings -> Found {len(filings)} filings"
                    )
                result_str = json.dumps(
                    {"count": len(filings), "filings": filings}, indent=2
                )
                self._log_progress(
                    f"Tool Result: fetch_latest_filings\nFound {len(filings)} filings\nDates: {[f['filing_date'] for f in filings]}"
                )
                return result_str
            else:
                if self.verbose:
                    self.logger.warning("No filings found")
                result_str = json.dumps(
                    {"error": f"No {form_type} filings found for CIK {cik}"}
                )
                self._log_progress(
                    f"Tool Result: fetch_latest_filings\nNo filings found"
                )
                return result_str
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in fetch_latest_filings: {e}")
            result_str = json.dumps({"error": str(e)})
            self._log_progress(f"Tool Result: fetch_latest_filings\nError: {str(e)}")
            return result_str

    def _extract_filing_section(self, input_str: str) -> str:
        """Extract a section via **Extractor API** from one of the cached filings.

        Input: "SECTION_NAME|FILING_INDEX" (e.g., "Risk Factors|0"). FILING_INDEX defaults to 0 (most recent).
        Supported section names:
          - For 10-K: "Risk Factors" (1A), "Management Discussion" (7), "Financial Statements" (8)
          - For 10-Q: "Risk Factors" (part2item1a), "Management Discussion" (part1item2), "Financial Statements" (part1item1)
        """
        if self.verbose:
            self.logger.info(
                f"🔧 TOOL CALLED: extract_filing_section(input='{input_str}')"
            )
        self._log_progress(f"Tool Call: extract_filing_section\nInput: '{input_str}'")

        if not self.current_filings:
            if self.verbose:
                self.logger.warning(
                    "No filings cached - extract_filing_section called before fetch_latest_filings"
                )
            error_msg = (
                "No filings have been fetched yet. Use fetch_latest_filings first."
            )
            self._log_progress(
                f"Tool Result: extract_filing_section\nError: {error_msg}"
            )
            return error_msg

        parts = input_str.split("|")
        section_name = parts[0].strip()
        filing_idx = int(parts[1].strip()) if len(parts) > 1 else 0

        if filing_idx >= len(self.current_filings):
            if self.verbose:
                self.logger.error(
                    f"Filing index {filing_idx} out of range (have {len(self.current_filings)} filings)"
                )
            error_msg = f"Filing index {filing_idx} out of range. Only {len(self.current_filings)} filings available."
            self._log_progress(
                f"Tool Result: extract_filing_section\nError: {error_msg}"
            )
            return error_msg

        filing = self.current_filings[filing_idx]
        form_type = filing.get("form_type") or filing.get("formType") or "10-K"
        url_htm = filing.get("url_htm")
        filing_date = filing.get("filing_date")

        if not url_htm:
            if self.verbose:
                self.logger.error("Cached filing has no HTML URL")
            error_msg = "Cached filing has no HTML URL."
            self._log_progress(
                f"Tool Result: extract_filing_section\nError: {error_msg}"
            )
            return error_msg

        if self.verbose:
            self.logger.info(
                f"Extracting '{section_name}' from {form_type} filed on {filing_date}"
            )
        result = extract_section_by_url(
            url_htm, form_type, section_name, return_type="text"
        )
        if self.verbose:
            self.logger.info(
                f"🔧 TOOL RESULT: extract_filing_section -> Extracted {len(result)} chars from filing {filing_idx}"
            )

        self._log_progress(
            f"Tool Result: extract_filing_section\nSection: {section_name}\nFiling: {filing_date}\nExtracted: {len(result)} characters"
        )

        return f"[Filing Date: {filing_date}]\n\n{result}"

    def _calculate_financial_metrics(self, _: str = "") -> str:
        """Return revenue, net income, and profit margin using **XBRL API** for all cached filings."""
        if self.verbose:
            self.logger.info(f"🔧 TOOL CALLED: calculate_financial_metrics()")
        self._log_progress(
            f"Tool Call: calculate_financial_metrics\nCalculating metrics for all cached filings"
        )

        if not self.current_filings:
            if self.verbose:
                self.logger.warning(
                    "No filings cached - calculate_financial_metrics called before fetch_latest_filings"
                )
            error_msg = (
                "No filings have been fetched yet. Use fetch_latest_filings first."
            )
            self._log_progress(
                f"Tool Result: calculate_financial_metrics\nError: {error_msg}"
            )
            return error_msg

        all_metrics = []
        for idx, filing in enumerate(self.current_filings):
            accession = filing.get("accession")
            url_htm = filing.get("url_htm")
            filing_date = filing.get("filing_date")
            if self.verbose:
                self.logger.info(
                    f"Calculating metrics for filing {idx} (date: {filing_date}, accession: {accession})"
                )
            metrics = xbrl_metrics(accession_no=accession, filing_url_htm=url_htm)
            metrics["filing_date"] = filing_date
            metrics["filing_index"] = idx
            all_metrics.append(metrics)

        if self.verbose:
            self.logger.info(
                f"🔧 TOOL RESULT: calculate_financial_metrics -> Calculated metrics for {len(all_metrics)} filings"
            )
        result_str = json.dumps(all_metrics, indent=2)
        self._log_progress(
            f"Tool Result: calculate_financial_metrics\nCalculated metrics for {len(all_metrics)} filings\nDates: {[m['filing_date'] for m in all_metrics]}"
        )

        return result_str

    def _populate_year_stats(self, input_str: str) -> str:
        """Populate financial statistics for a specific year into the FinancialStats object.

        Input format is dynamically generated based on METRICS_CONFIG.
        Format: "YEAR|METRIC1_VALUE|METRIC2_VALUE|..."

        The order of metrics follows the order in METRICS_CONFIG.
        All monetary values should be in dollars. Use None or empty string for missing values.
        """
        if self.verbose:
            self.logger.info(
                f"🔧 TOOL CALLED: populate_year_stats(input='{input_str}')"
            )
        self._log_progress(f"Tool Call: populate_year_stats\nInput: '{input_str}'")

        try:
            parts = input_str.split("|")
            if len(parts) < 2:
                metric_keys = list(METRICS_CONFIG.keys())
                expected_format = "YEAR|" + "|".join([k.upper() for k in metric_keys])
                error_result = json.dumps(
                    {"error": f"Invalid input format. Expected: {expected_format}"}
                )
                self._log_progress(
                    f"Tool Result: populate_year_stats\nError: Invalid format"
                )
                return error_result

            year = parts[0].strip()

            def parse_value(val_str: str) -> Optional[float]:
                val_str = val_str.strip()
                if not val_str or val_str.lower() in ["none", "n/a", "null", ""]:
                    return None
                try:
                    return float(val_str)
                except ValueError:
                    return None

            metrics = {}
            metric_keys = list(METRICS_CONFIG.keys())

            for idx, metric_key in enumerate(metric_keys):
                value_idx = idx + 1
                if value_idx < len(parts):
                    metrics[metric_key] = parse_value(parts[value_idx])
                else:
                    metrics[metric_key] = None

            year_stats = FinancialYearStats(year=year, metrics=metrics)

            self.financial_stats.add_year(year_stats)
            if self.verbose:
                self.logger.info(f"✅ Added/updated stats for year {year}: {metrics}")
                self.logger.info(
                    f"🔧 TOOL RESULT: populate_year_stats -> Successfully added stats for {year}"
                )

            result_str = json.dumps(
                {
                    "success": True,
                    "year": year,
                    "message": f"Financial stats for {year} added successfully",
                    "total_years": len(self.financial_stats.years),
                },
                indent=2,
            )

            self._log_progress(
                f"Tool Result: populate_year_stats\nYear: {year}\nMetrics populated: {len([v for v in metrics.values() if v is not None])}/{len(metrics)}"
            )

            return result_str

        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error in populate_year_stats: {e}")
            error_result = json.dumps({"error": str(e)})
            self._log_progress(f"Tool Result: populate_year_stats\nError: {str(e)}")
            return error_result

    def _create_agent(self, company_description: Optional[str] = None):
        """Create the SEC analysis agent using LangChain v1 and sec-api tools."""
        from langchain_core.tools import StructuredTool

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
        )

        # Wrap methods as LangChain tools
        tools = [
            StructuredTool.from_function(
                func=self._lookup_company_cik,
                name="lookup_company_cik",
                description="Use D2V Mapping API to resolve company name to CIK/ticker. Input: company name (e.g., 'Apple Inc'). Returns: JSON with cik, company_name, tickers.",
            ),
            StructuredTool.from_function(
                func=self._fetch_latest_filings,
                name="fetch_latest_filings",
                description="Use Query API to fetch the latest 5 years of filings and cache them. Input: 'CIK|FORM_TYPE|COUNT' (e.g., '0000320193|10-K|5'). FORM_TYPE defaults to 10-K, COUNT defaults to 5. Returns JSON array with filing metadata.",
            ),
            StructuredTool.from_function(
                func=self._extract_filing_section,
                name="extract_filing_section",
                description="Extract a section via Extractor API from one of the cached filings. Input: 'SECTION_NAME|FILING_INDEX' (e.g., 'Risk Factors|0'). FILING_INDEX defaults to 0 (most recent). Supported sections: Risk Factors, Management Discussion, Financial Statements.",
            ),
            StructuredTool.from_function(
                func=self._calculate_financial_metrics,
                name="calculate_financial_metrics",
                description="Return revenue, net income, and profit margin using XBRL API for all cached filings. Input: empty string or any value. Returns JSON array with metrics for each filing.",
            ),
            StructuredTool.from_function(
                func=self._populate_year_stats,
                name="populate_year_stats",
                description=f"Populate financial statistics for a specific year. Input format: 'YEAR|METRIC1|METRIC2|...'. Order: {list(METRICS_CONFIG.keys())}. Use None or empty for missing values. All values in dollars.",
            ),
        ]

        metric_keys = list(METRICS_CONFIG.keys())
        metrics_format = "|".join([k.upper() for k in metric_keys])
        metrics_list = ", ".join(
            [METRICS_CONFIG[k]["display_name"].lower() for k in metric_keys]
        )

        company_context = (
            f"\n\nCompany Context: {company_description}" if company_description else ""
        )

        system_message = f"""You are a financial analyst specializing in SEC filings analysis across multiple years.{company_context}

Use the provided tools which wrap the **SEC API by D2V** to analyze the past 5 years of 10-K filings:
(1) map company name->CIK
(2) fetch the latest 5 years of 10-K filings
(3) extract sections via Extractor API from specific filings
(4) pull XBRL metrics for all filings
(5) populate financial statistics into a structured format

Workflow:
1) lookup_company_cik to get CIK.
2) fetch_latest_filings with 'CIK|10-K|5' to get the past 5 years of 10-K filings.
3) calculate_financial_metrics to get {metrics_list} for all 5 years.
4) For each year's financial data, call populate_year_stats with format: "YEAR|{metrics_format}" 
   (extract the fiscal year from the filing_date or period_end field, e.g., "2023|150000000000|25000000000|35000000000|10000000000").
   Make sure to call this for ALL years of data you receive.
5) For each filing (or at least the most recent ones), extract_filing_section with 'Risk Factors|INDEX' where INDEX is the filing number (0=most recent, 1=previous year, etc.).
6) Write a comprehensive multi-year report in the following format.

IMPORTANT: After calling calculate_financial_metrics, you MUST call populate_year_stats for each year of financial data.
This is critical as the financial table will be automatically appended to your report.

# SEC Multi-Year Analysis Report: [Company Name]
**Analysis Period:** [earliest date] to [most recent date]
**Filings Analyzed:** [number] 10-K filings

## Executive Summary
[2-3 sentences summarizing key trends and findings across the 5-year period]

## Financial Trends (5-Year Analysis)
[Present revenue, net income, and profit margins for each year in a clear format]
[Identify and explain key trends: growth rates, improvements, deteriorations]

## Risk Assessment Evolution
[Analyze how risk factors have evolved over the 5-year period]
[Highlight new risks that emerged and old risks that diminished]
[Identify the top 5-7 most significant current risks]

## Management Outlook Trends
[If available, summarize how management's discussion has evolved]
[Note any strategic shifts or persistent themes]

## Year-over-Year Comparative Analysis
[Compare key metrics and developments between years]
[Identify inflection points or significant changes]

## Conclusion
[Overall assessment of the company's 5-year trajectory]
[Key takeaways for investors and risk assessment]
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=self.verbose, max_iterations=20
        )

        return agent_executor

    def run(self, company: str, company_description: Optional[str] = None) -> str:
        """
        Analyze the past 5 years of 10-K filings for a company.

        Args:
            company: Company name (e.g., "Apple Inc")
            company_description: Optional description of the company to provide context

        Returns:
            str: Comprehensive multi-year financial and risk analysis report
        """
        self.financial_stats = FinancialStats()
        self.current_filings = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.save_trace:
            self.progress_file_path = (
                f"sec_agent_progress_{company.replace(' ', '_')}_{timestamp}.txt"
            )
            with open(self.progress_file_path, "w", encoding="utf-8") as f:
                f.write(f"SEC Agent Progress Tracker\n")
                f.write(f"Company: {company}\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")

        if self.verbose:
            self.logger.info("=" * 80)
            self.logger.info(f"🚀 STARTING 5-YEAR ANALYSIS FOR: {company}")
            if self.save_trace:
                self.logger.info(
                    f"📋 Progress will be tracked in: {self.progress_file_path}"
                )
            self.logger.info("=" * 80)

        agent = self._create_agent(company_description)
        user_message = f"Analyze the past 5 years of SEC 10-K filings for {company} and generate a comprehensive multi-year risk and financial trends report."

        if self.verbose:
            self.logger.info(f"Invoking agent with message: {user_message}")
        self._log_progress(
            f"Agent Invocation\nTask: Analyze 5 years of 10-K filings for {company}"
        )

        result = agent.invoke({"input": user_message})

        if self.verbose:
            self.logger.info("=" * 80)
            self.logger.info(f"✅ 5-YEAR ANALYSIS COMPLETE FOR: {company}")
            self.logger.info("=" * 80)

        self._log_progress(f"Analysis Complete\nCompany: {company}\nStatus: Success")

        # Extract the output from the agent result
        report_content = ""
        try:
            # AgentExecutor returns dict with 'output' key
            if isinstance(result, dict) and "output" in result:
                report_content = result["output"]
            else:
                report_content = str(result)
        except Exception:
            report_content = str(result)

        if self.financial_stats.years:
            if self.verbose:
                self.logger.info(
                    f"📊 Appending financial table with {len(self.financial_stats.years)} years of data"
                )
            financial_table = (
                "\n\n---\n\n## Financial Metrics Summary (5-Year)\n\n"
                + self.financial_stats.to_markdown_table()
            )
            report_content += financial_table
        else:
            if self.verbose:
                self.logger.warning(
                    "⚠️ No financial stats were populated during analysis"
                )

        if self.save_trace and self.progress_file_path:
            with open(self.progress_file_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Financial years tracked: {len(self.financial_stats.years)}\n")
            if self.verbose:
                self.logger.info(f"📋 Progress log saved to: {self.progress_file_path}")

        if self.save_final:
            filename = f"sec_report_{company.replace(' ', '_')}_{timestamp}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            if self.verbose:
                self.logger.info(f"💾 Report saved to: {filename}")

        return report_content


# ============================================================================
# Main Function (matches other agents' pattern)
# ============================================================================


def run_sec_agent(
    ticker: str, company_description: Optional[str] = None, save_to_file: bool = True
) -> tuple[str, str]:
    """
    Execute the SEC Agent analysis following the standard agent pattern.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        company_description: Optional description of the company
        save_to_file: Whether to save the detailed report to a file

    Returns:
        Tuple of (summary_report, detailed_report)
    """
    print(f"\n{'='*60}")
    print(f"🤖 SEC Agent: Analyzing SEC filings for ${ticker}")
    print(f"{'='*60}\n")

    try:
        # Get company name from ticker (simplified - in production use a ticker->name mapping)
        company_name = ticker  # You may want to add a ticker-to-company-name lookup

        if not company_description:
            company_description = f"A company trading under ticker {ticker}"

        # Create SEC Agent instance
        agent = SecAgent(verbose=True, save_final=save_to_file, save_trace=save_to_file)

        print(f"📊 Analyzing 5 years of 10-K filings...")
        print(f"📋 {get_metrics_documentation()}\n")

        # Run the analysis
        detailed_report = agent.run(company_name, company_description)

        # Extract key information for summary
        # Try to extract overall risk level from the report
        risk_level = "Unknown"
        if "HIGH RISK" in detailed_report.upper():
            risk_level = "High Risk"
        elif "MODERATE RISK" in detailed_report.upper():
            risk_level = "Moderate Risk"
        elif "LOW RISK" in detailed_report.upper():
            risk_level = "Low Risk"

        # Count filings analyzed
        filings_count = (
            len(agent.current_filings) if hasattr(agent, "current_filings") else 5
        )

        # Get financial years if available
        years_analyzed = ""
        if agent.financial_stats and agent.financial_stats.years:
            year_strings = [y.year for y in agent.financial_stats.years]
            years_analyzed = f"{min(year_strings)} - {max(year_strings)}"

        # Create summary report
        summary_report = f"""
**SEC Agent Summary: ${ticker}**

📊 **Analysis Scope:**
* **Filings Analyzed:** {filings_count} years of 10-K reports
* **Years Covered:** {years_analyzed if years_analyzed else 'Past 5 years'}
* **Overall Risk Assessment:** {risk_level}

� **Financial Metrics Tracked:**
{get_metrics_documentation()}

💡 **Key Insight:**
Comprehensive 5-year SEC filing analysis completed, including risk factors,
financial trends, and business developments.

✅ **Enhanced with Multi-Year Analysis**
This report synthesizes multiple years of SEC filings to identify trends
and patterns in financial performance and risk disclosures.

---
*Full detailed report with financial metrics saved to file*
*Agent: SEC Agent | Model: Gemini Pro with SEC API*
        """

        print("✅ SEC_Agent: Analysis complete")

        # Extract financial metrics if available
        financial_metrics = {}
        if agent.financial_stats and agent.financial_stats.years:
            # Convert financial stats to a more usable format
            metrics_by_year = []
            for year_stat in agent.financial_stats.years:
                metrics_by_year.append(
                    {"year": year_stat.year, "metrics": year_stat.metrics}
                )
            financial_metrics = {
                "years_data": metrics_by_year,
                "years_list": [y.year for y in agent.financial_stats.years],
            }

        # Return comprehensive dictionary like news_agent
        return {
            "ticker": ticker,
            "agent": "SEC",
            "filings_analyzed": filings_count,
            "years_covered": years_analyzed if years_analyzed else "Past 5 years",
            "overall_risk_level": risk_level,
            "financial_metrics": financial_metrics,
            "metrics_documentation": get_metrics_documentation(),
            "summary_report": summary_report.strip(),
            "detailed_report": detailed_report,
        }

    except Exception as e:
        print(f"❌ SEC_Agent: Analysis failed - {e}")
        import traceback

        traceback.print_exc()

        error_report = f"""
**SEC Agent Report: ${ticker}**

⚠️ **Error:** SEC filing analysis could not be completed.

**Details:** {str(e)}

**Possible Causes:**
- Invalid ticker symbol
- SEC API key not configured
- No recent 10-K filings available
        """
        return {
            "ticker": ticker,
            "agent": "SEC",
            "error": str(e),
            "summary_report": error_report.strip(),
            "detailed_report": error_report.strip(),
        }


# ============================================================================
# CLI / Example Usage
# ============================================================================


def main() -> None:
    """Example usage of the SEC Agent."""
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "AAPL"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python sec_agent.py <TICKER>\n")

    company_description = "A technology company that designs, manufactures, and markets consumer electronics, computer software, and online services."

    result = run_sec_agent(ticker, company_description, save_to_file=True)

    print("\n" + "=" * 60)
    print("SEC AGENT SUMMARY")
    print("=" * 60 + "\n")
    print(result.get("summary_report", ""))
    print("\n" + "=" * 60)
    print("\n📄 Full report saved to current directory")


if __name__ == "__main__":
    main()
