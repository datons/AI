import yfinance as yf
from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime
import os


def calculate_significant_returns_simple(
    ticker: str, start: str = None, end: str = None, freq: str = "D", n: int = 5
):
    """
    Calculate the returns for a single stock and return the n most extreme values.
    """
    try:
        df_base = yf.download(ticker, start, end)
    except Exception as e:
        raise ValueError(f"Error downloading stock data: {e}")

    if freq not in ["D", "M", "Q", "Y"]:
        raise ValueError("Frequency must be one of 'D', 'M', 'Q', 'Y'")

    if freq == "D":
        returns = df_base["Close"].pct_change().mul(100).dropna()
    else:
        returns = (
            df_base["Close"]
            .to_period(freq)
            .groupby(level=0)
            .last()
            .pct_change()
            .mul(100)
            .dropna()
        )

    returns = returns[ticker]

    extreme_indices = returns.abs().nlargest(n).index
    return returns.loc[extreme_indices]


@tool
def calculate_significant_returns_tool(
    ticker: Annotated[str, "ticker of the stock to calculate returns for"],
    start: Annotated[str, "start date in YYYY-MM-DD format"],
    end: Annotated[str, "end date in YYYY-MM-DD format"],
    freq: Annotated[str, "frequency (e.g., 'D' for daily, 'M' for monthly)"],
    n: Annotated[int, "number of most extreme values to return"] = 5,
):
    """
    Calculate the returns for a single stock and return the n most extreme values.
    """
    try:
        df_base = yf.download(ticker, start, end)
    except Exception as e:
        raise ValueError(f"Error downloading stock data: {e}")

    if freq not in ["D", "M", "Q", "Y"]:
        raise ValueError("Frequency must be one of 'D', 'M', 'Q', 'Y'")

    if freq == "D":
        returns = df_base["Adj Close"].pct_change().mul(100).dropna()
    else:
        returns = (
            df_base["Adj Close"]
            .to_period(freq)
            .groupby(level=0)
            .last()
            .pct_change()
            .mul(100)
            .dropna()
        )

    returns = returns[ticker]

    extreme_indices = returns.abs().nlargest(n).index
    return returns.loc[extreme_indices]


def save_output_to_file(output, folder, middle_name=None):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    if middle_name:
        output_folder = folder / f"outputs/{current_datetime}_{middle_name}"
    else:
        output_folder = folder / f"outputs/{current_datetime}"

    output_folder.mkdir(parents=True, exist_ok=True)
    path = output_folder / "output.md"

    with open(path, "w") as file:
        file.write(output)
