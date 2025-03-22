import textarena as ta
import smithery
import mcp
import os
import json
import asyncio
from textarena.core import Agent
from typing import Optional
from dotenv import load_dotenv
import random

STANDARD_GAME_PROMPT = "You are a competitive and reflective game player. Reflect deeply after each round, and use the best strategies together with your learnings to win. Aim for the best possible outcomes for yourself, and don't simply tell the truth if your opponents asks for it, Use the coding tool if it helps you. You're limited to 5 tool calling per round. Make sure you use only English words without illegal characters as responses for language-based games. No repetition of previous words if you're playing language-based games. Make sure you read the game instructions carefully, and always follow the required format."

class AsyncAnthropicAgent(Agent):
    """Agent class using the Anthropic Claude API to generate responses asynchronously."""
    def __init__(self, model_name: str, system_prompt: Optional[str] = STANDARD_GAME_PROMPT, max_tokens: int = 1000, temperature: float = 0.9, verbose: bool = False):
        """
        Initialize the Anthropic agent.

        Args:
            model_name (str): The name of the Claude model (e.g., "claude-3-5-sonnet-20241022").
            system_prompt (Optional[str]): The system prompt to use (default: STANDARD_GAME_PROMPT).
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for randomness in response generation.
            verbose (bool): If True, additional debug info will be printed.
        """
        super().__init__()
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package is required for AsyncAnthropicAgent. "
                "Install it with: pip install anthropic"
            )
            
        self.client = anthropic.AsyncAnthropic()
    
    async def _make_request(self, observation: str) -> str:
        """Make a single API request to Anthropic and return the generated message."""
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": observation}]}
            ]
        )
        
        return response.content[0].text.strip()
    
    async def _retry_request(self, observation: str, retries: int = 3, delay: int = 5) -> str:
        """
        Attempt to make an API request with retries.

        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                response = await self._make_request(observation)
                if self.verbose:
                    print(f"\nObservation: {observation}\nResponse: {response}")
                return response
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt} failed with error: {e}")
                if attempt < retries:
                    await asyncio.sleep(delay)
        raise last_exception
    
    async def __call__(self, observation: str) -> str:
        """
        Process the observation using the Anthropic API and return the generated response.
        
        Args:
            observation (str): The input string to process.
        
        Returns:
            str: The generated response.
        """
        if not isinstance(observation, str):
            raise ValueError(f"Observation must be a string. Received type: {type(observation)}")
        return await self._retry_request(observation)
    
class MCPAgent(AsyncAnthropicAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/e2b/ws", {"e2bApiKey": os.environ["E2B_API_KEY"]}
        )

    async def _make_request(self, observation: str) -> str:
        """Make a single API request to Anthropic and return the generated message."""
        async with smithery.websocket_client(self.url) as streams:
            async with mcp.client.session.ClientSession(*streams) as session:

                try:
                    tools_result = await session.list_tools()
                    tools = tools_result.model_dump()["tools"]

                    tools = [
                        {"input_schema": tool.pop("inputSchema"), **tool}
                        for tool in tools
                        if "inputSchema" in tool
                    ]

                    print("Available tools:", tools)

                    final_response_text = ""
                    is_tool_call_pending = True
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": observation}],
                        }
                    ]
                    print("\n=== Entering tool call handling loop ===")
                    # Loop to handle multiple tool calls in a conversation
                    while is_tool_call_pending:
                        print(f"\n[DEBUG] Messages before API call: {json.dumps(messages, indent=2)}")
                        response = await self.client.messages.create(
                            model=self.model_name,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            system=self.system_prompt,
                            messages=messages,
                            tools=tools,
                        )

                        print("Response:", response)

                        # Check if there's a tool_use in the response
                        is_tool_call_pending = False
                        for content_block in response.content:
                            if content_block.type == "tool_use":
                                is_tool_call_pending = True

                                tool_name = content_block.name
                                tool_input = content_block.input
                                tool_id = content_block.id

                                print(f"Tool called: {tool_name}")
                                print(f"Tool input: {json.dumps(tool_input, indent=2)}")

                                # Execute the tool using MCP session
                                try:
                                    print(f"\n[DEBUG] Executing tool: {tool_name}")
                                    tool_result = await session.call_tool(
                                        tool_name, tool_input
                                    )
                                    tool_result_dict = tool_result.model_dump()
                                    print(f"[DEBUG] Tool execution completed. Result: {json.dumps(tool_result_dict, indent=2)}")

                                    # Convert tool result to string format for Anthropic
                                    # The content must be a string, not an object
                                    tool_result_dict = tool_result.model_dump()
                                except Exception as e:
                                    if "MCP error" in str(e):
                                        tool_result_dict = {"error": str(e)}

                                result_str = json.dumps(tool_result_dict)
                                print(f"Tool result: {result_str}")

                                # Add tool call and result to messages
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": [content_block.model_dump()],
                                    }
                                )

                                # Add tool response to messages - content must be a string
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": tool_id,
                                                "content": result_str,  # Now it's a string
                                            }
                                        ],
                                    }
                                )
                            elif content_block.type == "text":
                                # Accumulate text responses
                                final_response_text += content_block.text

                        # If no tool calls were made, we use the text response
                        if not is_tool_call_pending and not final_response_text:
                            final_response_text = response.content[0].text

                except Exception as e:

                    print(f"Error: {e}")
                    raise e

            return final_response_text.strip()

GAMES = ["Nim-v0"]
ROUNDS_PER_GAME = 5

def play_game(agent, game_id, model_name, model_description, email):
    env = ta.make_online(
        env_id=[game_id],
        model_name=model_name,
        model_description=model_description,
        email=email
    )
    env = ta.wrappers.LLMObservationWrapper(env=env)
    env.reset(num_players=2)
    done = False

    while not done:
        player_id, observation = env.get_observation()
        action = asyncio.get_event_loop().run_until_complete(agent(observation))
        done, info = env.step(action=action)
        print("Step complete")

    rewards = env.close()
    print(f"Game {game_id} finished. Rewards:", rewards)
    return rewards

if __name__ == "__main__":
    os.environ["ANTHROPIC_API_KEY"] = "sk-.."
    os.environ["E2B_API_KEY"] = "e2b_.."

    model_name = "Team Awesome"
    model_description = "Standard Anthropic model with custom prompts"
    email = "jackietanyen@gmail.com"

    agent = MCPAgent(
        model_name="claude-3-7-sonnet-20250219"
    )

    game_results = {game: [] for game in GAMES}

    for game in GAMES:
        for _ in range(ROUNDS_PER_GAME):
            try:
                rewards = play_game(agent, game, model_name, model_description, email)
                game_results[game].append(rewards)
            except Exception as e:
                print(f"Game {game} failed with error: {e}")

    best_game = max(game_results, key=lambda game: sum(game_results[game]) / len(game_results[game]))
    print(f"Best game based on win rate: {best_game}")

    for _ in range(ROUNDS_PER_GAME):
        try:
            play_game(agent, best_game, model_name, model_description, email)
        except Exception as e:
            print(f"Game {best_game} failed with error: {e}")