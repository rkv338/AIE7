"""
Simple Agent Implementation using LangGraph that communicates with the A2A Agent Node.

This simple agent demonstrates how to create a LangGraph-based agent that can make
API calls to the main Agent Node through the A2A protocol.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, TypedDict, Annotated
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAgentState(TypedDict):
    """State for the Simple Agent LangGraph."""
    messages: Annotated[List, add_messages]
    query: str
    a2a_response: str
    is_complete: bool


class SimpleAgent:
    """
    A Simple Agent that uses LangGraph to make API calls to the A2A Agent Node.
    
    This agent demonstrates the A2A protocol usage by:
    1. Taking user input
    2. Making API calls to the main agent through A2A protocol
    3. Processing and presenting responses
    """
    
    def __init__(self, a2a_base_url: str = "http://localhost:10000"):
        """
        Initialize the Simple Agent.
        
        Args:
            a2a_base_url: Base URL of the A2A agent server
        """
        self.a2a_base_url = a2a_base_url
        self.model = ChatOpenAI(
            model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.7,
        )
        
        # Build the LangGraph
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build the LangGraph for the Simple Agent."""
        graph = StateGraph(SimpleAgentState)
        
        # Add nodes
        graph.add_node("plan_query", self._plan_query_node)
        graph.add_node("call_a2a_agent", self._call_a2a_agent_node)
        graph.add_node("process_response", self._process_response_node)
        
        # Set entry point
        graph.set_entry_point("plan_query")
        
        # Add edges
        graph.add_edge("plan_query", "call_a2a_agent")
        graph.add_edge("call_a2a_agent", "process_response")
        graph.add_edge("process_response", END)
        
        return graph.compile()
    
    def _plan_query_node(self, state: SimpleAgentState) -> Dict[str, Any]:
        """
        Plan and refine the query before sending to the A2A agent.
        This node demonstrates how to prepare queries for the main agent.
        """
        logger.info("Planning query...")
        
        # Get the user's original query
        user_message = state["messages"][-1].content if state["messages"] else state.get("query", "")
        
        # Use LLM to refine the query for better results
        system_prompt = """
        You are a query planner for an AI assistant. Your job is to take a user's query 
        and refine it to be clear, specific, and likely to get good results from an AI agent 
        that has access to web search, academic papers, and document retrieval tools.
        
        Return just the refined query, nothing else.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original query: {user_message}"}
        ]
        
        response = self.model.invoke(messages)
        refined_query = response.content.strip()
        
        logger.info(f"Refined query: {refined_query}")
        
        return {
            "query": refined_query,
            "messages": [AIMessage(content=f"Planning to search for: {refined_query}")]
        }
    
    async def _call_a2a_agent_node(self, state: SimpleAgentState) -> Dict[str, Any]:
        """
        Make an API call to the A2A Agent Node.
        This demonstrates the core A2A protocol communication.
        """
        logger.info("Calling A2A Agent...")
        
        query = state["query"]
        
        try:
            # Create HTTP client with extended timeout
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
                # Initialize A2A Card Resolver
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=self.a2a_base_url,
                )
                
                # Fetch agent card
                agent_card = await resolver.get_agent_card()
                logger.info("Successfully fetched agent card")
                
                # Create A2A client
                client = A2AClient(
                    httpx_client=httpx_client, 
                    agent_card=agent_card
                )
                
                # Prepare message payload
                send_message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            {'kind': 'text', 'text': query}
                        ],
                        'message_id': uuid4().hex,
                    },
                }
                
                # Create request
                request = SendMessageRequest(
                    id=str(uuid4()), 
                    params=MessageSendParams(**send_message_payload)
                )
                
                # Send message and get response
                response = await client.send_message(request)
                
                # Extract response content
                response_content = ""
                if hasattr(response, 'root') and hasattr(response.root, 'result'):
                    result = response.root.result
                    if hasattr(result, 'artifacts') and result.artifacts:
                        for artifact in result.artifacts:
                            if hasattr(artifact, 'content') and artifact.content:
                                for part in artifact.content:
                                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                        response_content += part.root.text + "\n"
                
                if not response_content:
                    response_content = "Received response from A2A agent, but content could not be extracted."
                
                logger.info("Successfully received response from A2A Agent")
                
                return {
                    "a2a_response": response_content.strip(),
                    "messages": [AIMessage(content=f"Received response from A2A Agent: {response_content[:100]}...")]
                }
                
        except Exception as e:
            logger.error(f"Error calling A2A Agent: {e}")
            error_msg = f"Failed to get response from A2A Agent: {str(e)}"
            return {
                "a2a_response": error_msg,
                "messages": [AIMessage(content=error_msg)]
            }
    
    def _process_response_node(self, state: SimpleAgentState) -> Dict[str, Any]:
        """
        Process and enhance the response from the A2A agent.
        This demonstrates post-processing of A2A responses.
        """
        logger.info("Processing A2A response...")
        
        a2a_response = state["a2a_response"]
        original_query = state["query"]
        
        # Use LLM to enhance and format the response
        system_prompt = """
        You are a response processor. You receive responses from an AI agent and your job is to:
        1. Format the response nicely for the user
        2. Add helpful context if needed
        3. Ensure the response directly addresses the user's query
        4. If the response seems incomplete or unhelpful, suggest follow-up actions
        
        Present the final response in a clear, helpful format.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original query: {original_query}\n\nAgent response: {a2a_response}"}
        ]
        
        response = self.model.invoke(messages)
        processed_response = response.content.strip()
        
        logger.info("Response processing complete")
        
        return {
            "is_complete": True,
            "messages": [AIMessage(content=processed_response)]
        }
    
    async def run(self, query: str) -> str:
        """
        Run the Simple Agent with a user query.
        
        Args:
            query: User's query to process
            
        Returns:
            Processed response from the A2A agent
        """
        logger.info(f"Starting Simple Agent with query: {query}")
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "a2a_response": "",
            "is_complete": False
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Return the final response
        if final_state["messages"]:
            return final_state["messages"][-1].content
        else:
            return "No response generated."


async def main():
    
    # Create Simple Agent
    agent = SimpleAgent()
    
    # Example queries to demonstrate different capabilities
    queries = [
        "What are the latest developments in artificial intelligence?",
        "Find recent papers on transformer architectures",
        "What are the current trends in machine learning?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        try:
            response = await agent.run(query)
            print(f"🔍 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Error processing query: {e}", exc_info=True)
        
        print("\n" + "=" * 50)


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
