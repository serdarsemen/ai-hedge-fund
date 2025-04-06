from graph.state import AgentState, show_agent_reasoning
from tools.api import (
    get_financial_metrics,
    get_market_cap,
    # search_line_items,
    # get_insider_trades,
    get_company_news,
    get_prices
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from utils.progress import progress
from utils.llm import call_llm
import concurrent.futures
from functools import lru_cache
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class AnalysisCache:
    """Cache for analyzed metrics to avoid recalculation"""
    latest_metrics: Any
    moving_averages: Dict[str, float]
    sentiment_scores: Dict[str, float]

class AdaptiveQuantSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str

def adaptive_quant_agent(state: AgentState):
    """
    Optimized adaptive quantitative agent that combines multiple strategies
    """
    data = state["data"]
    end_date = data["end_date"]
    start_date = data["start_date"]
    tickers = data["tickers"]

    # Initialize batch analysis and adaptive analysis dictionaries
    analysis_data_batch = {}
    analysis_cache = defaultdict(dict)
    adaptive_analysis = {}  # Initialize the dictionary here

    for ticker in tickers:
        progress.update_status("adaptive_quant_agent", ticker, "Starting analysis")

        # Parallel API calls using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                "market_cap": executor.submit(get_market_cap, ticker, end_date),
                "financial": executor.submit(get_financial_metrics, ticker, end_date, "annual", 5),
                "prices": executor.submit(get_prices, ticker, start_date, end_date),
                "news": executor.submit(get_company_news, ticker, end_date, start_date, 50)
            }

            # Gather results
            market_cap = futures["market_cap"].result()
            financial_metrics = futures["financial"].result()
            prices = futures["prices"].result()
            news = futures["news"].result()

            # Pre-calculate commonly used metrics
            if financial_metrics:
                analysis_cache[ticker]["latest_metrics"] = financial_metrics[0]
                analysis_cache[ticker]["moving_averages"] = calculate_moving_averages(prices)
                analysis_cache[ticker]["sentiment_scores"] = preprocess_sentiment(news)

        # Combine multiple strategy scores with dynamic weights
        quality_score = analyze_quality(financial_metrics, analysis_cache[ticker]) * 0.25
        value_score = analyze_value(market_cap, financial_metrics) * 0.25
        momentum_score = analyze_momentum(prices, analysis_cache[ticker]["moving_averages"]) * 0.25
        sentiment_score = analyze_sentiment(news, analysis_cache[ticker]["sentiment_scores"]) * 0.25

        total_score = quality_score + value_score + momentum_score + sentiment_score

        analysis_data_batch[ticker] = {
            "quality_metrics": quality_score,
            "value_metrics": value_score,
            "momentum_metrics": momentum_score,
            "sentiment_metrics": sentiment_score,
            "total_score": total_score
        }

    # Make a single LLM call with batch data
    batch_output = generate_batch_adaptive_output(
        analysis_data_batch,
        state["metadata"]["model_name"],
        state["metadata"]["model_provider"]
    )

    for ticker, output in batch_output.items():
        adaptive_analysis[ticker] = {
            "signal": output.signal,
            "confidence": output.confidence,
            "reasoning": output.reasoning
        }

        progress.update_status("adaptive_quant_agent", ticker, "Done")

    message = HumanMessage(content=json.dumps(adaptive_analysis), name="adaptive_quant_agent")

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(adaptive_analysis, "Adaptive Quant Agent")

    state["data"]["analyst_signals"]["adaptive_quant_agent"] = adaptive_analysis

    return {"messages": [message], "data": state["data"]}

def generate_batch_adaptive_output(
    analysis_data_batch: dict,
    model_name: str,
    model_provider: str,
) -> dict:
    """Process multiple tickers in a batch and return outputs."""
    results = {}

    for ticker, analysis_data in analysis_data_batch.items():
        results[ticker] = generate_adaptive_output(
            ticker,
            analysis_data,
            model_name,
            model_provider
        )

    return results

def generate_adaptive_output(
    ticker: str,
    analysis_data: dict,
    model_name: str,
    model_provider: str,
) -> AdaptiveQuantSignal:
    """Generate the final signal using LLM."""
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an Adaptive Quantitative AI agent that combines multiple investment strategies:
            1. Quality metrics (Munger-style)
            2. Value metrics (Graham-style)
            3. Momentum and trends (Druckenmiller-style)
            4. Market sentiment analysis

            Make decisions based on a holistic view of all metrics while adapting to current market conditions.
            """
        ),
        (
            "human",
            """Based on the following analysis, create an investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return strictly in this JSON format:
            {{
                "signal": "bullish" | "bearish" | "neutral",
                "confidence": float (0-100),
                "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_signal():
        return AdaptiveQuantSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=AdaptiveQuantSignal,
        agent_name="adaptive_quant_agent",
        default_factory=create_default_signal,
    )

def calculate_moving_averages(prices: List[Any]) -> Dict[str, float]:
    """Calculate moving averages using cached individual calculations"""
    if not prices or len(prices) < 50:
        return {"sma_20": 0, "sma_50": 0, "current_price": 0}

    # Convert prices to tuple for caching
    closes = tuple(p.close for p in prices)

    return {
        "sma_20": calculate_moving_average_single(closes, 20),
        "sma_50": calculate_moving_average_single(closes, 50),
        "current_price": closes[-1]
    }

@lru_cache(maxsize=128)
def calculate_moving_average_single(prices_tuple: tuple, window: int) -> float:
    """Calculate a single moving average with caching"""
    if not prices_tuple or len(prices_tuple) < window:
        return 0.0
    return np.mean(prices_tuple[-window:])

def preprocess_sentiment(news: List[Any]) -> Dict[str, float]:
    """Pre-process and cache sentiment analysis"""
    if not news:
        return {"positive_ratio": 0.5, "total_analyzed": 0}

    negative_keywords = frozenset(["lawsuit", "fraud", "investigation", "decline", "bearish", "downgrade"])
    positive_keywords = frozenset(["growth", "upgrade", "innovation", "partnership", "bullish", "beat"])

    positive_count = sum(
        1 for item in news
        if any(keyword in (item.title.lower() if item.title else "") for keyword in positive_keywords)
    )
    negative_count = sum(
        1 for item in news
        if any(keyword in (item.title.lower() if item.title else "") for keyword in negative_keywords)
    )

    total_analyzed = positive_count + negative_count
    return {
        "positive_ratio": positive_count / total_analyzed if total_analyzed > 0 else 0.5,
        "total_analyzed": total_analyzed
    }

def analyze_quality(financial_metrics: List[Any], cache: Dict) -> float:
    """Optimized quality analysis"""
    if not financial_metrics:
        return 0.0

    score = 0.0
    latest = cache["latest_metrics"]

    # Check ROE (Return on Equity)
    if latest.return_on_equity and latest.return_on_equity > 0.15:
        score += 2.5
    elif latest.return_on_equity and latest.return_on_equity > 0.10:
        score += 1.5

    # Check Operating Margin
    if latest.operating_margin and latest.operating_margin > 0.20:
        score += 2.5
    elif latest.operating_margin and latest.operating_margin > 0.15:
        score += 1.5

    return min(score, 10.0)

def analyze_value(market_cap: float, financial_metrics: list) -> float:
    """
    Analyze company valuation using Graham's principles:
    - P/E ratio (price_to_earnings_ratio)
    - P/B ratio (price_to_book_ratio)
    - Debt levels (debt_to_equity)
    - Current ratio (current_ratio)
    """
    if not financial_metrics or not market_cap or len(financial_metrics) == 0:
        return 0.0

    score = 0.0
    latest = financial_metrics[0]

    # P/E Analysis
    if hasattr(latest, 'price_to_earnings_ratio') and latest.price_to_earnings_ratio:
        if latest.price_to_earnings_ratio < 15:
            score += 2.5
        elif latest.price_to_earnings_ratio < 20:
            score += 1.5

    # P/B Analysis
    if hasattr(latest, 'price_to_book_ratio') and latest.price_to_book_ratio:
        if latest.price_to_book_ratio < 1.5:
            score += 2.5
        elif latest.price_to_book_ratio < 3:
            score += 1.5

    # Current Ratio
    if hasattr(latest, 'current_ratio') and latest.current_ratio:
        if latest.current_ratio > 2:
            score += 2.5
        elif latest.current_ratio > 1.5:
            score += 1.5

    # Debt to Equity
    if hasattr(latest, 'debt_to_equity') and latest.debt_to_equity:
        if latest.debt_to_equity < 0.5:
            score += 2.5
        elif latest.debt_to_equity < 1:
            score += 1.5

    return min(score, 10.0)  # Cap at 10

def analyze_momentum(prices: List[Any], moving_averages: Dict[str, float]) -> float:
    """Optimized momentum analysis using pre-calculated metrics"""
    if not prices or len(prices) < 50:
        return 0.0

    score = 0.0
    current_price = moving_averages["current_price"]
    sma_20 = moving_averages["sma_20"]
    sma_50 = moving_averages["sma_50"]

    if current_price > sma_20:
        score += 2.5
    if current_price > sma_50:
        score += 2.5
    if sma_20 > sma_50:
        score += 2.5
    if current_price > prices[-10].close:
        score += 2.5

    return min(score, 10.0)

def analyze_sentiment(news: List[Any], sentiment_scores: Dict[str, float]) -> float:
    """Optimized sentiment analysis using pre-calculated scores"""
    if not news:
        return 5.0

    sentiment_ratio = sentiment_scores["positive_ratio"]
    return min(max(sentiment_ratio * 10, 0.0), 10.0)  # Ensure between 0 and 10







