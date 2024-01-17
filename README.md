# Stock Support and Resistance Identification

## Overview

This repository houses a Python script for automated identification of support and resistance levels in historical stock price data. Leveraging k-means clustering and the elbow method, the script provides a comprehensive solution for traders and analysts to pinpoint key levels of price significance. The script further allows for the retrieval of stock data from Yahoo Finance, optimal clustering determination, and the creation of visually insightful OHLC (Open, High, Low, Close) charts with highlighted support and resistance lines.

## Features

-   Download historical stock data using Yahoo Finance API.
-   Find optimal clusters based on the elbow method.
-   Identify support and resistance levels using k-means clustering and local extrema.
-   Plot OHLC charts with highlighted support and resistance lines.

## How to Use

1. Install pipenv: `pip install pipenv`.
2. Install the required libraries: `pipenv install`.
3. Run the `main.py` script with your preferred stock ticker symbol.

```bash
python main.py AAPL
python main.py AALI.JK
```

## Screenshot
![127 0 0 1_56904_](https://github.com/boysugi20/python-stock-support-resistance/assets/53815726/75e695d9-b288-47bb-8317-38d5b170ab7c)
