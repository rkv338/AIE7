from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
import json
import requests
from dice_roller import DiceRoller

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

"""
Add your own tool here, and then use it through Cursor!
"""
@mcp.tool()
def get_metal_price(metal: str, currency: str = "USD", unit: str = "toz") -> str:
    """Get the current spot price for a metal from Metals.dev.

    - metal: e.g. "gold", "silver", "platinum", "palladium"
    - currency: e.g. "USD" (default)
    - unit: e.g. "toz" (troy ounce, default)
    """

    api_key = os.getenv("METALS_DEV_API_KEY")
    if not api_key:
        return "Error: METALS_DEV_API_KEY is not set in the environment."

    endpoint = "https://api.metals.dev/v1/metal/spot"
    params = {
        "api_key": api_key,
        "metal": metal,
        "currency": currency,
        "unit": unit,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=15)
        if response.status_code == 401 or response.status_code == 403:
            return "Error: Unauthorized. Check your METALS_DEV_API_KEY."
        response.raise_for_status()
        data = response.json()

        # Try to produce a friendly summary if possible
        price = None
        # Common shapes: { status: 'success', rate: { price, metal, currency, unit } }
        if isinstance(data, dict):
            rate = data.get("rate") or data.get("data") or {}
            if isinstance(rate, dict):
                price = rate.get("price") or rate.get("spot")

        if price is not None:
            return f"{metal.upper()} {price} {currency}/{unit}"

        # Fallback: return raw JSON if structure is unexpected
        return json.dumps(data)
    except requests.Timeout:
        return "Error: Request to Metals.dev timed out."
    except requests.RequestException as exc:
        return f"Error: Failed to fetch price: {exc}"

if __name__ == "__main__":
    mcp.run(transport="stdio")