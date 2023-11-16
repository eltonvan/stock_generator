from dotenv import load_dotenv
import os
import streamlit as st
from langchain.llms import OpenAI
import yfinance as yf
import json
import re

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=API_KEY)


def get_stock_info(prompt):
    result = llm(prompt)
    return result


def get_stock_prices(stock_list):
    result_with_prices = []
    ticker_pattern = r"\(([A-Z]+)\)"

    for stock in stock_list:
        match = re.search(ticker_pattern, stock)

        if match:
            ticker = match.group(1)  
            
            stock_info = yf.Ticker(ticker)
            try:
                current_price = stock_info.history(period="1d")["Close"].iloc[-1]
                formatted_price = f"Current Price: ${current_price:.2f}"
                result_with_prices.append(f"{stock.strip()}. {formatted_price}")
            except IndexError as e:
                print(e)

    return result_with_prices


prompt = "provide a list of stocks for my portfolio. 10 stocks, not less and not more with name, ticker, and reason for recommendation. \n\n"
stocks = get_stock_info(prompt)

stocks_list = stocks.split("\n")[:-1]
stocks_with_prices = get_stock_prices(stocks_list)


for stock in stocks_with_prices:
    print(stock)
