import yfinance as yf
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
import numpy as np
import heapq
import plotly.graph_objects as go
import argparse

# Constants
# ROUNDING_BASES is a dictionary defining rounding bases for different price ranges.
# Each key represents the upper limit of a price range, and the corresponding value is the rounding base for that range.
# For example, if the stock price is in the range up to 5000, each tick is 25. If the price is in the range 2000-5000, each tick is 10.
ROUNDING_BASES = {5000: 25, 2000: 10, 500: 5, 200: 2}

def download_stock_data(ticker, period="1y"):
    """Download historical stock data for a given ticker."""
    return yf.download(ticker, period=period)

def find_optimal_clusters(df, saturation_point=0.1):
    """Find the optimal number of clusters using the elbow method."""
    wcss = []
    k_models = []
    price_data = df[['Close']]

    size = min(11, len(df.index))
    for i in range(1, size):
        kmeans = KMeans(
            n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
        )
        kmeans.fit(price_data)
        wcss.append(kmeans.inertia_)
        k_models.append(kmeans)

    # Compare differences in inertias until it's no more than saturation_point
    optimum_k = len(wcss) - 1
    for i in range(0, len(wcss) - 1):
        diff = abs(wcss[i + 1] - wcss[i])
        if diff < saturation_point:
            optimum_k = i
            break

    optimum_clusters = k_models[optimum_k]

    return optimum_clusters

def round_price(price, last_close):
    """Round the price based on the last closing value."""
    base = next((v for k, v in sorted(ROUNDING_BASES.items(), reverse=True) if last_close >= k), 1)
    return base * round(price / base)

def find_local_extrema(data, condition):
    """Find local extrema in the data based on the given condition."""
    indices = argrelextrema(data.head(20).values, condition)[0]
    return [data.iloc[idx]["Close"] for idx in indices]

def get_most_recent(array, price, type, min_difference=25):
    """Get the most recent values in the array based on the given type."""
    new_array = [round(line) for line in array if (line < price and type == "Support") or (line >= price and type == "Resistance")]
    new_array = list(set(new_array))  # Remove duplicates

    if not new_array:
        return []

    temp = heapq.nsmallest(3, new_array, key=lambda x: abs(x - price))

    # Filter levels with a minimum difference
    result = [temp[0]]
    for level in temp[1:]:
        if abs(level - result[-1]) >= min_difference:
            result.append(level)

    return result

def get_support_and_resistance(df):
    """Get support and resistance levels for a given stock ticker."""
    last_close = df["Close"].iloc[0]
    close_clusters = find_optimal_clusters(df)

    close_centers = close_clusters.cluster_centers_
    all_lines = sorted([item for sublist in close_centers.tolist() for item in sublist])

    support = get_most_recent(all_lines, last_close, "Support")
    resistance = get_most_recent(all_lines, last_close, "Resistance")

    # If no support generated, find local minima
    if len(support) == 0:
        min_idx = argrelextrema(df["Close"].head(20).values, np.less_equal)[0]
        for idx in min_idx:
            if last_close > df.iloc[idx]["Close"]:
                support.append(df.iloc[idx]["Close"])

    # If no resistance generated, find local maxima
    if len(resistance) == 0:
        max_idx = argrelextrema(df["Close"].head(20).values, np.greater_equal)[0]
        for idx in max_idx:
            if last_close <= df.iloc[idx]["Close"]:
                resistance.append(df.iloc[idx]["Close"])

    support_list = [round_price(x, last_close) for x in support]
    resistance_list = [round_price(x, last_close) for x in resistance]
            
    # Remove duplicates
    support_list = list(set(support_list))
    resistance_list = list(set(resistance_list))

    # Choose the top 3 based on trading volume
    top_support = heapq.nlargest(3, support_list, key=lambda x: df[df.index == x]["Volume"].values[0] if not df[df.index == x].empty else 0)
    top_resistance = heapq.nlargest(3, resistance_list, key=lambda x: df[df.index == x]["Volume"].values[0] if not df[df.index == x].empty else 0)

    return sorted(top_support), sorted(top_resistance)

def plot_ohlc_graph(df, support, resistance):
    """Plot OHLC graph with support and resistance lines."""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         name='Candlesticks')])

    for level in support:
        fig.add_trace(go.Scatter(x=df.index, y=[level] * len(df), mode='lines', line=dict(color='green'), name='Support'))

    for level in resistance:
        fig.add_trace(go.Scatter(x=df.index, y=[level] * len(df), mode='lines', line=dict(color='red'), name='Resistance'))

    fig.update_layout(title='OHLC Chart with Support and Resistance',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)

    fig.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Identify support and resistance levels in stock price data.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL, AALI.JK)')
    args = parser.parse_args()

    # Download Stock Data
    df = download_stock_data(args.ticker)
    df.sort_index(ascending=False, inplace=True)

    # Get Support and Resistance level
    result = get_support_and_resistance(df)

    #Show Results
    print("Last Close:", df["Close"].iloc[0])
    print("Support Levels:", result[0])
    print("Resistance Levels:", result[1])
    plot_ohlc_graph(df, result[0], result[1])