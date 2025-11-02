import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import re

#example_header = yf.download("AAPL", period="1y", interval="1d")
#print(example_header.head())
#[*********************100%***********************]  1 of 1 completed
#Price            Close        High         Low        Open    Volume
#Ticker            AAPL        AAPL        AAPL        AAPL      AAPL
#Date
#2024-11-01  221.877365  224.306064  219.249596  219.946350  65276700
#2024-11-04  220.981537  221.757922  218.692204  219.966273  44944500

def MC_sims(ticker, t, sims):
    print(f"\nRunning Monte Carlo simulation for {ticker} over {t} trading days with {sims} simulations...:")
    # Download historical data
    data = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)
    prices = data['Close'].values
    prices = [item[0] for item in prices]
    #i.e.
    #new_prices = []
    #for i in prices:
        #new_prices.append(i[0])

    print(min(prices), max(prices))
    print(np.isnan(prices).sum())

    # Sanity check: enough data
    if len(prices) < 2:
        print("Not enough historical data to run simulation.")
        return

    # Daily log returns
    ln_returns = np.diff(np.log(prices))
    print(ln_returns)
    mu_daily = np.mean(ln_returns)
    sigma_daily = np.std(ln_returns)
    S0 = float(prices[-1])  # latest stock price
    print(f"ln range: min={ln_returns.min():.2f}, max={ln_returns.max():.2f}")

    #S&P500 values:
    sp500_data = yf.download("^GSPC", period="1y", interval="1d", auto_adjust=True)
    sp500_prices = sp500_data['Close'].values
    sp500_prices = [item[0] for item in sp500_prices]

    sp500ln_returns = np.diff(np.log(sp500_prices))
    print(sp500ln_returns)
    sp500mu_daily = np.mean(sp500ln_returns)
    sp500sigma_daily = np.std(sp500ln_returns)
    S0_sp500 = float(sp500_prices[-1])  # latest stock price
    print(f"ln range: min={sp500ln_returns.min():.2f}, max={sp500ln_returns.max():.2f}")

    #Calculate Correlation Coefficient (rho)
    #This is the single most important addition for a multi-asset MC
    rho = np.corrcoef(ln_returns, sp500ln_returns)[0, 1]

    # Annualize using fixed trading days
    TRADING_DAYS_PER_YEAR = 250
    #Stock
    mu = mu_daily * TRADING_DAYS_PER_YEAR
    sigma = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    #S&P500
    sp500_mu = sp500mu_daily * TRADING_DAYS_PER_YEAR
    sp500_sigma = sp500sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    #Step size
    dt = 1 / TRADING_DAYS_PER_YEAR  # daily step in years

    # Print basic info
    #Stock
    print(f"\nRetrieved {len(prices)} days of data for {ticker}")
    print(f"Annualised drift (mu): {mu:.4f}")
    print(f"Annualised volatility (sigma): {sigma:.4f}")
    print(f"Current stock price (S0): {S0:.2f}")

    #S&P500
    print(f"\nRetrieved {len(sp500_prices)} days of data for S&P500")
    print(f"Annualised drift (mu): {sp500_mu:.4f}")
    print(f"Annualised volatility (sigma): {sp500_sigma:.4f}")
    print(f"Current stock price (S0): {S0_sp500:.2f}")

    # Monte Carlo simulation
    z = np.random.standard_normal((t, sims))
    
    forecast = np.zeros((t + 1, sims))
    forecast[0] = S0

    for i in range(1, t + 1):
        exp_term = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[i-1]
        forecast[i] = forecast[i-1] * np.exp(exp_term)
        if np.isnan(forecast[i]).any():
            print(f"NaN detected at step {i}, stopping simulation")
            break
    
    sp500_forecast = np.zeros((t + 1, sims))
    sp500_forecast[0] = S0_sp500

    for i in range(1, t+1):
        sp500_exp_term = (sp500_mu - 0.5*sp500_sigma**2) * dt + sp500_sigma * np.sqrt(dt)*z[i-1]
        sp500_forecast[i] = sp500_forecast[i-1] * np.exp(sp500_exp_term)
        if np.isnan(forecast[i]).any():
            print(f"NaN detected at {i}, stopping simulation")
            break

    # Check forecast range
    #Stock
    print(f"Forecast range: min={forecast.min():.2f}, max={forecast.max():.2f}")
    #S&P500
    print(f"S&P500 Forecast range: min={sp500_forecast.min():.2f}, max={sp500_forecast.max():.2f}")

    # Percentiles for visualization
    lower = np.percentile(forecast, 5, axis=1)
    middle = np.percentile(forecast, 50, axis=1)
    upper = np.percentile(forecast, 95, axis=1)

    sp500lower = np.percentile(sp500_forecast, 5, axis=1)
    sp500middle = np.percentile(sp500_forecast, 50, axis=1)
    sp500upper = np.percentile(sp500_forecast, 95, axis=1)

    # Plot using integer days on x-axis
    days = [i for i in range(t + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(days, middle, label=f"Median {ticker} Forecast", color='blue')
    plt.fill_between(days, lower, upper, color='lightgreen', alpha=0.3, label='5th-95th percentile')

    # Sample paths
    colors = plt.cm.viridis(np.linspace(0, 1, 15))
    for i in range(15):
        plt.plot(days, forecast[:, i], color=colors[i], alpha=0.7, linewidth=0.8)
        #plt.plot(days, forecast[:, i], color='red', alpha=0.7, linewidth=0.8, label=f"{ticker} forecast")
        #plt.plot(days, sp500_forecast[:, i], color='green', alpha=0.7, linewidth=0.8, label="S&P500 forecast")

    plt.title(f"Monte Carlo Simulations for {ticker} vs S&P500 - {t} Days Ahead")
    plt.xlabel("Days Ahead")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Summary statistics
    print(f"\nExpected {ticker} price after {t} days: mean = {forecast[-1].mean():.2f}, std = {forecast[-1].std():.2f}")
    print(f"\nExpected S&P500 price after {t} days: mean = {sp500_forecast[-1].mean():.2f}, std = {sp500_forecast[-1].std():.2f}")

    final_price = forecast[-1:]
    expected_fp = final_price.mean()
    print(f"\nFinal price of {ticker} after {t} days: ${expected_fp:.2f}")
    expected_return = (expected_fp/S0 - 1)*100
    annualised_return = ((expected_fp / S0) ** (TRADING_DAYS_PER_YEAR / t) - 1) * 100
    print(f"\nAverage total {ticker} return over {len(forecast)-1} trading days: {expected_return:.2f}%")
    print(f"Average total {ticker} annual return: {annualised_return:.2f}%")

    sp500_final_prices = sp500_forecast[-1, :]
    sp500_expected_fp = sp500_final_prices.mean()
    print(f"\nFinal price of S&P500 after {t} days: ${sp500_expected_fp:.2f}")
    sp500_expected_return = (sp500_expected_fp / S0_sp500 - 1) * 100
    sp500_annualised_return = ((sp500_expected_fp / S0_sp500) ** (TRADING_DAYS_PER_YEAR / t) - 1) * 100
    print(f"\nAverage total S&P500 return over {len(forecast)-1} trading days: {sp500_expected_return:.2f}%")
    print(f"Average total S&P500 annual return: {sp500_annualised_return:.2f}%")

    return forecast, sp500_forecast

if __name__ == "__main__":   # Ensures that the code inside this block only executes when the file is run directly.
    print("""
    ================================
      Monte Carlo Stock Simulator
    ================================
    """)   # Multi-line string literal enclosed in triple quotes (""") allows printing multiple lines at once.

    # --- Ticker input ---
    while True:
        ticker = input("Enter a stock ticker (e.g. AAPL, TSLA, MSFT): ").strip().upper()
        # .strip removes leading and trailing whitespace from a string.
        # .upper() converts all letters in the string to uppercase.
        if ticker:  # Checks if the input is non-empty
            break
        print("Please enter a valid ticker.")

    # --- Forecast horizon input ---
    while True:
        try:
            t = int(input("Enter forecast horizon in trading days (e.g. 60, 100, 180): "))
            # Converts the input string to an integer
            if t <= 0:
                print("Number of trading days can't be negative, please enter a positive integer.")
                continue
            break
        except ValueError:
            # Handles invalid input that cannot be converted to an integer
            print("Invalid input. Please enter a valid integer for forecast horizon.")

    # --- Number of simulations input ---
    while True:
        try:
            sims = int(input("Enter number of MC simulations (e.g. 10000): "))
            # Converts the input string to an integer
            if sims <= 0:
                print("Number of simulations must be positive.")
                continue
            break
        except ValueError:
            # Handles invalid input that cannot be converted to an integer
            print("Invalid input. Defaulting to 10,000 simulations.")
            sims = 10000
            break

    # --- Run simulation ---
    print(f"\nRunning MC simulations for {ticker} over {t} trading days, simulating {sims} times...")
    forecast_output = MC_sims(ticker, t, sims)

# Example usage
# forecast_output = MC_sims("AAPL", 100, 10000)