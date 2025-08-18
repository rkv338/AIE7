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
                
                # Extract response content with improved error handling
                response_content = self._extract_response_content(response)
                
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
    
    def _extract_response_content(self, response) -> str:
        """
        Extract content from A2A response with robust error handling.
        
        This method tries multiple approaches to extract meaningful content
        from the A2A response structure.
        """
        logger.info(f"Extracting content from response type: {type(response)}")
        
        try:
            # Method 1: Try the standard A2A structure
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                result = response.root.result
                logger.info(f"Found result object: {type(result)}")
                
                # Check for artifacts (main A2A response structure)
                if hasattr(result, 'artifacts') and result.artifacts:
                    logger.info(f"Found {len(result.artifacts)} artifacts")
                    content_parts = []
                    
                    for i, artifact in enumerate(result.artifacts):
                        logger.info(f"Processing artifact {i}: {type(artifact)}")
                        
                        # Check for 'parts' field (new structure)
                        if hasattr(artifact, 'parts') and artifact.parts:
                            for j, part in enumerate(artifact.parts):
                                logger.info(f"Processing part {j}: {type(part)}")
                                
                                # Extract text from part
                                text_content = None
                                if hasattr(part, 'text'):
                                    text_content = part.text
                                elif hasattr(part, 'content'):
                                    text_content = str(part.content)
                                
                                if text_content:
                                    content_parts.append(str(text_content))
                                    logger.info(f"Extracted text from part {j}: {len(str(text_content))} chars")
                        
                        # Check for 'content' field (old structure)
                        elif hasattr(artifact, 'content') and artifact.content:
                            for j, part in enumerate(artifact.content):
                                logger.info(f"Processing content part {j}: {type(part)}")
                                
                                # Try different ways to extract text
                                text_content = None
                                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                    text_content = part.root.text
                                elif hasattr(part, 'text'):
                                    text_content = part.text
                                elif hasattr(part, 'content'):
                                    text_content = str(part.content)
                                
                                if text_content:
                                    content_parts.append(str(text_content))
                                    logger.info(f"Extracted text from part {j}: {len(str(text_content))} chars")
                    
                    if content_parts:
                        combined_content = "\n".join(content_parts)
                        logger.info(f"Successfully extracted {len(combined_content)} characters from artifacts")
                        return combined_content
                
                # Check for direct message content in result
                if hasattr(result, 'message'):
                    logger.info("Found direct message in result")
                    return str(result.message)
                
                # Check for content field in result
                if hasattr(result, 'content'):
                    logger.info("Found content field in result")
                    return str(result.content)
            
            # Method 2: Try model_dump if available (Pydantic models)
            if hasattr(response, 'model_dump'):
                logger.info("Trying model_dump approach")
                response_dict = response.model_dump()
                logger.info(f"Response structure: {list(response_dict.keys())}")
                
                # Navigate through common response structures
                if 'result' in response_dict:
                    result = response_dict['result']
                    if 'artifacts' in result:
                        artifacts = result['artifacts']
                        content_parts = []
                        for artifact in artifacts:
                            # New structure: parts field
                            if 'parts' in artifact:
                                for part in artifact['parts']:
                                    if 'text' in part:
                                        content_parts.append(part['text'])
                                    elif 'content' in part:
                                        content_parts.append(str(part['content']))
                            # Old structure: content field
                            elif 'content' in artifact:
                                for part in artifact['content']:
                                    if 'root' in part and 'text' in part['root']:
                                        content_parts.append(part['root']['text'])
                                    elif 'text' in part:
                                        content_parts.append(part['text'])
                        
                        if content_parts:
                            combined_content = "\n".join(content_parts)
                            logger.info(f"Extracted content via model_dump: {len(combined_content)} chars")
                            return combined_content
            
            # Method 3: Try to convert to string and look for patterns
            response_str = str(response)
            logger.info(f"Response string length: {len(response_str)}")
            
            # Look for JSON-like content
            if 'text' in response_str and len(response_str) > 50:
                logger.info("Found text content in string representation")
                return f"Response content available (see logs for details): {response_str[:200]}..."
            
            # Method 4: Check direct attributes
            for attr in ['content', 'text', 'message', 'data']:
                if hasattr(response, attr):
                    value = getattr(response, attr)
                    if value and str(value).strip():
                        logger.info(f"Found content in {attr} attribute")
                        return str(value)
            
            logger.warning("Could not extract meaningful content from response")
            return f"Response received but content extraction failed. Response type: {type(response)}"
            
        except Exception as e:
            logger.error(f"Error extracting response content: {e}", exc_info=True)
            return f"Response parsing error: {str(e)}. Response type: {type(response)}"
    
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


async def debug_a2a_connection():
    """Debug function to test A2A server connectivity and response structure."""
    print("🔧 Starting A2A Debug Session...")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        try:
            # Test 1: Basic connectivity
            print("\n1️⃣ Testing basic server connectivity...")
            base_url = "http://localhost:10000"
            response = await httpx_client.get(base_url)
            print(f"   Server status: {response.status_code}")
            
            # Test 2: Agent card discovery
            print("\n2️⃣ Testing agent card discovery...")
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=base_url,
            )
            
            try:
                agent_card = await resolver.get_agent_card()
                print(f"   ✅ Agent card retrieved: {type(agent_card)}")
                print(f"   Agent name: {agent_card.name if hasattr(agent_card, 'name') else 'Unknown'}")
            except Exception as e:
                print(f"   ❌ Agent card error: {e}")
                return
            
            # Test 3: Simple message test
            print("\n3️⃣ Testing simple message...")
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card
            )
            
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': 'Hello, can you tell me what 2+2 equals?'}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await client.send_message(request)
            print(f"   Response received: {type(response)}")
            
            # Test 4: Response extraction
            print("\n4️⃣ Testing response extraction...")
            agent = SimpleAgent()
            
            # Show the full response structure for debugging
            if hasattr(response, 'model_dump'):
                import json
                response_dict = response.model_dump()
                print(f"   Full response structure:")
                print(f"   {json.dumps(response_dict, indent=2)[:500]}...")
            
            extracted_content = agent._extract_response_content(response)
            print(f"   Extracted content length: {len(extracted_content)}")
            print(f"   Content preview: {extracted_content[:200]}...")
            
        except Exception as e:
            print(f"❌ Debug session failed: {e}")
            logger.error(f"Debug error: {e}", exc_info=True)


async def main():
    """Main function with debug option."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        await debug_a2a_connection()
        return
    
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
