import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from colorama import Fore, Style, init
import questionary
from agents.portfolio_manager import portfolio_management_agent
from agents.risk_manager import risk_management_agent
from graph.state import AgentState
from utils.display import print_trading_output
from utils.analysts import ANALYST_ORDER, get_analyst_nodes
from utils.progress import progress
from llm.models import LLM_ORDER, get_model_info
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils.visualize import save_graph_as_png
import json

# Load environment variables from .env file
load_dotenv()
init(autoreset=True)

# Cache for workflow instances
workflow_cache = {}

@lru_cache(maxsize=128)
def parse_hedge_fund_response(response: str) -> Optional[Dict]:
    """Parses a JSON string and returns a dictionary with caching."""
    try:
        return json.loads(response)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error parsing response: {e}\nResponse: {repr(response)}")
        return None

def initialize_portfolio(tickers: List[str], initial_cash: float, margin_requirement: float) -> Dict:
    """Initialize portfolio structure efficiently."""
    position_template = {
        "long": 0,
        "short": 0,
        "long_cost_basis": 0.0,
        "short_cost_basis": 0.0,
        "short_margin_used": 0.0,
    }

    realized_template = {
        "long": 0.0,
        "short": 0.0,
    }

    return {
        "cash": initial_cash,
        "margin_requirement": margin_requirement,
        "margin_used": 0.0,
        "positions": {ticker: position_template.copy() for ticker in tickers},
        "realized_gains": {ticker: realized_template.copy() for ticker in tickers}
    }

def run_hedge_fund(
    tickers: List[str],
    start_date: str,
    end_date: str,
    portfolio: Dict,
    show_reasoning: bool = True,
    selected_analysts: Optional[List[str]] = None,
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
) -> Dict:
    """Optimized hedge fund execution."""
    progress.start()

    try:
        # Convert list to tuple for caching
        cache_key = tuple(sorted(selected_analysts)) if selected_analysts else None
        if cache_key not in workflow_cache:
            # Convert back to tuple for create_workflow
            workflow = create_workflow(cache_key)
            workflow_cache[cache_key] = workflow.compile()

        agent = workflow_cache[cache_key]

        final_state = agent.invoke(
            {
                "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": model_provider,
                },
            }
        )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        progress.stop()

def start(state):
    """Initialize the starting state for the workflow."""
    return state

@lru_cache(maxsize=32)
def create_workflow(selected_analysts: Optional[tuple] = None) -> StateGraph:
    """Create workflow with caching support."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    analyst_nodes = get_analyst_nodes()
    # No need to convert to list here since we're already passing a tuple
    selected_list = selected_analysts if selected_analysts else tuple(analyst_nodes.keys())

    # Add selected analyst nodes
    for analyst_key in selected_list:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Add management nodes
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect analysts to risk management
    for analyst_key in selected_list:
        workflow.add_edge(analyst_nodes[analyst_key][0], "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)
    workflow.set_entry_point("start_node")

    return workflow

def parse_args() -> argparse.Namespace:
    """Parse command line arguments efficiently."""
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--margin-requirement", type=float, default=0.0)
    parser.add_argument("--tickers", type=str, default="MSFT,NVDA,AAPL,GOOGL,TSLA")
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--show-agent-graph", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # Select analysts using questionary
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style([
            ("checkbox-selected", "fg:green"),
            ("selected", "fg:green noinherit"),
            ("highlighted", "noinherit"),
            ("pointer", "noinherit"),
        ]),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)

    selected_analysts = choices
    print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

    # Select LLM model
    model_choice = questionary.select(
        "Select your LLM model:",
        choices=[questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER],
        style=questionary.Style([
            ("selected", "fg:green bold"),
            ("pointer", "fg:green bold"),
            ("highlighted", "fg:green"),
            ("answer", "fg:green bold"),
        ])
    ).ask()

    if not model_choice:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)

    model_info = get_model_info(model_choice)
    model_provider = model_info.provider.value if model_info else "Unknown"
    print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # Handle dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Initialize portfolio
    portfolio = initialize_portfolio(tickers, args.initial_cash, args.margin_requirement)

    # Show agent graph if requested
    if args.show_agent_graph:
        file_path = "_".join(selected_analysts) + "_graph.png" if selected_analysts else "graph.png"
        with ThreadPoolExecutor() as executor:
            executor.submit(save_graph_as_png, workflow_cache.get(tuple(sorted(selected_analysts))), file_path)

    # Run hedge fund and display results
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_choice,
        model_provider=model_provider,
    )
    print_trading_output(result)

if __name__ == "__main__":
    main()  # Only call main(), remove the extra print_trading_output(result)
