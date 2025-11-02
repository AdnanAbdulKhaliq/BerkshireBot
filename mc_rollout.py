import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

#example_header = yf.download("AAPL", period="1y", interval="1d")
#print(example_header.head())
#[*********************100%***********************]  1 of 1 completed
#Price            Close        High         Low        Open    Volume
#Ticker            AAPL        AAPL        AAPL        AAPL      AAPL
#Date
#2024-11-01  221.877365  224.306064  219.249596  219.946350  65276700
#2024-11-04  220.981537  221.757922  218.692204  219.966273  44944500

def MC_sims(ticker, t, sims):
    # Download historical data
    data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True)
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

    # Annualize using fixed trading days
    TRADING_DAYS_PER_YEAR = 250
    mu = mu_daily * TRADING_DAYS_PER_YEAR
    sigma = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    dt = 1 / TRADING_DAYS_PER_YEAR  # daily step in years

    # Print basic info
    print(f"\nRetrieved {len(prices)} days of data for {ticker}")
    print(f"Annualised drift (mu): {mu:.4f}")
    print(f"Annualised volatility (sigma): {sigma:.4f}")
    print(f"Current stock price (S0): {S0:.2f}")

    # Monte Carlo simulation
    forecast = np.zeros((t + 1, sims))
    forecast[0] = S0
    z = np.random.standard_normal((t, sims))

    for i in range(1, t + 1):
        exp_term = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[i-1]
        forecast[i] = forecast[i-1] * np.exp(exp_term)
        if np.isnan(forecast[i]).any():
            print(f"NaN detected at step {i}, stopping simulation")
            break

    # Check forecast range
    print(f"Forecast range: min={forecast.min():.2f}, max={forecast.max():.2f}")

    # Percentiles for visualization
    lower = np.percentile(forecast, 5, axis=1)
    middle = np.percentile(forecast, 50, axis=1)
    upper = np.percentile(forecast, 95, axis=1)

    # Plot using integer days on x-axis
    days = [i for i in range(t + 1)]
    plt.figure(figsize=(10, 6))
    plt.plot(days, middle, label="Median Forecast", color='blue')
    plt.fill_between(days, lower, upper, color='pink', alpha=0.3, label='5th-95th percentile')

    # Sample paths
    colors = plt.cm.viridis(np.linspace(0, 1, 15))
    for i in range(15):
        plt.plot(days, forecast[:, i], color=colors[i], alpha=0.7, linewidth=0.8)

    plt.title(f"Monte Carlo Simulations for {ticker} - {t} Days Ahead")
    plt.xlabel("Days Ahead")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()
    ## hello
    ## goodbye


    # Summary statistics
    print(f"\nExpected price after {t} days: mean = {forecast[-1].mean():.2f}, std = {forecast[-1].std():.2f}")

    return forecast

# Example usage
forecast_output = MC_sims("AAPL", 100, 10000)
