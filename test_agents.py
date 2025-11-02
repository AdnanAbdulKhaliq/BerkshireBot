from news_agent import analyze_company_sentiment  # Commented out - not testing
from social_agent import run_social_agent  # Commented out - not testing
from analyst_agent import run_analyst_agent  # Commented out - not testing
from sec_agent import run_sec_agent
from risk_assessment_agent import run_risk_assessment_agent

# Test News Agent
print("\n" + "=" * 70)
print("Testing News Sentiment Agent")
print("=" * 70)

company_name = "Apple"

news_result = analyze_company_sentiment(
    company_name=company_name,
    max_articles=15,
    lookback_days=7,
    verbose=True,
)

print("\n📊 News Agent Result:")
print(news_result)

# # Test Social Agent
# print("\n" + "=" * 70)
# print("Testing Social Sentiment Agent")
# print("=" * 70)

# ticker = "AAPL"

# social_result = run_social_agent(
#     ticker=ticker, save_to_file=False  # Don't save to file during testing
# )

# print("\n📊 Social Agent Result:")
# print("\n📊 Social Agent Result:")
# print("\n📊 Social Agent Result:")
# print(social_result)

# # Test Analyst Agent
# print("\n" + "=" * 70)
# print("Testing Analyst Agent")
# print("=" * 70)

# analyst_result = run_analyst_agent(
#     ticker=ticker, save_to_file=False  # Don't save to file during testing
# )

# print("\n📊 Analyst Agent Result:")
# print(analyst_result)

# # Test SEC Agent
# print("\n" + "=" * 70)
# print("Testing SEC Agent")
# print("=" * 70)

# company_description = "A technology company that designs, manufactures, and markets consumer electronics, computer software, and online services."

# sec_result = run_sec_agent(
#     ticker=ticker,
#     company_description=company_description,
#     save_to_file=False,  # Don't save to file during testing
# )

# print("\n📊 SEC Agent Result:")
# print("\n--- Agent Info ---")
# print(f"Ticker: {sec_result['ticker']}")
# print(f"Agent: {sec_result['agent']}")
# print(f"Filings Analyzed: {sec_result['filings_analyzed']}")
# print(f"Years Covered: {sec_result['years_covered']}")
# print(f"Risk Level: {sec_result['overall_risk_level']}")

# print("\n--- Summary Report ---")
# print(sec_result["summary_report"])

# print("\n--- Detailed Report (first 1000 chars) ---")
# detailed = sec_result["detailed_report"]
# print(detailed[:1000] + "..." if len(detailed) > 1000 else detailed)

# print("\n--- Financial Metrics Preview ---")
# if sec_result.get("financial_metrics"):
#     metrics = sec_result["financial_metrics"]
#     if metrics.get("years_list"):
#         print(f"Years with data: {metrics['years_list']}")
#         print(f"Number of years: {len(metrics['years_list'])}")

# # Test Risk Assessment Agent
# print("\n" + "=" * 70)
# print("Testing Risk Assessment Agent")
# print("=" * 70)

# investment_memo = """
# Apple Inc. (AAPL) is a leading technology company known for its consumer electronics,
# software, and services. The company has strong brand loyalty, ecosystem lock-in, and
# consistent revenue growth. Key products include iPhone, iPad, Mac, Apple Watch, and
# services like Apple Music and iCloud. The company has a strong balance sheet with
# significant cash reserves and generates substantial free cash flow.
# """

# risk_result = run_risk_assessment_agent(
#     ticker=ticker,
#     investment_memo=investment_memo,
#     save_to_file=False,  # Don't save to file during testing
# )

# print("\n📊 Risk Assessment Agent Result:")
# print(risk_result)
