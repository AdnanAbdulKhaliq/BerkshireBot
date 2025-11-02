"""
Test script for mock Yahoo Finance data integration
"""

print("Testing Monte Carlo Simulation with Mock Data")
print("=" * 60)

from mc_rollout import MC_sims

tickers = ["AAPL", "TSLA", "NVDA"]

for ticker in tickers:
    print(f"\n🎲 Testing {ticker}...")
    try:
        result = MC_sims(ticker, 30, 100, display=False)
        print(f"   ✅ Success!")
        print(f"   📊 Current price: ${result['median'][0]:.2f}")
        print(f"   📈 30-day forecast (median): ${result['median'][-1]:.2f}")
        print(f"   📉 Expected range: ${result['lower'][-1]:.2f} - ${result['upper'][-1]:.2f}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

print("\n" + "=" * 60)
print("\nTesting Analyst Agent with Mock Data")
print("=" * 60)

from analyst_agent import fetch_yahoo_finance_data

for ticker in tickers:
    print(f"\n📊 Testing analyst data for {ticker}...")
    try:
        data = fetch_yahoo_finance_data(ticker)
        if "error" in data:
            print(f"   ❌ Error: {data['error']}")
        else:
            print(f"   ✅ Success!")
            analyst_info = data.get("analyst_info", {})
            if analyst_info:
                print(f"   💰 Current Price: ${analyst_info.get('currentPrice', 'N/A')}")
                print(f"   🎯 Target Mean: ${analyst_info.get('targetMeanPrice', 'N/A')}")
                print(f"   📈 Recommendation: {analyst_info.get('recommendationKey', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

print("\n" + "=" * 60)
print("✅ All tests completed!")
