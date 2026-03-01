# Building Your First AI Agent with Claude

**Level:** Beginner | **Time:** 45 minutes | **Cost:** $0-5

## What You'll Learn

In this tutorial, you'll build a working AI agent from scratch that can:
- Process user requests autonomously
- Make decisions and take actions
- Learn from feedback
- Optimize its behavior over time

## Prerequisites

- Basic Python knowledge (3.8+)
- An API key from Anthropic or OpenRouter
- A code editor (VS Code recommended)
- Git installed

## Step 1: Set Up Your Project

```bash
# Create directory
mkdir my-first-agent
cd my-first-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install anthropic python-dotenv requests
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your-key-here
```

## Step 2: Understand Agent Architecture

An AI agent has three core components:

```
┌─────────────┐
│   Input     │ (User request, context, available tools)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│   Agent Brain    │ (Claude model with system prompt)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Tool Selection  │ (What action to take next)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Tool Execution │ (Actually do the thing)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Response       │ (Return result to user)
└──────────────────┘
```

## Step 3: Create Your First Agent

Create `agent.py`:

```python
import anthropic
import json
from typing import Any

client = anthropic.Anthropic()

# Define what tools your agent can use
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform math calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression (e.g., '5 + 3')"
                }
            },
            "required": ["expression"]
        }
    }
]

def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Execute a tool and return the result"""
    if tool_name == "get_weather":
        # Simulated weather data
        return f"Weather in {tool_input['location']}: Sunny, 72°F"
    
    elif tool_name == "calculate":
        try:
            result = eval(tool_input['expression'])
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    return "Tool not found"

def run_agent(user_message: str) -> str:
    """Run the agent with a user message"""
    
    messages = [
        {"role": "user", "content": user_message}
    ]
    
    print(f"\n🤖 Agent: Processing request...")
    print(f"📝 User: {user_message}\n")
    
    # Initial request to Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=TOOLS,
        messages=messages
    )
    
    # Agentic loop - keep going until done
    while response.stop_reason == "tool_use":
        # Find the tool use block
        tool_use = None
        for block in response.content:
            if block.type == "tool_use":
                tool_use = block
                break
        
        if not tool_use:
            break
        
        # Execute the tool
        tool_name = tool_use.name
        tool_input = tool_use.input
        
        print(f"🔧 Tool Used: {tool_name}")
        print(f"   Input: {json.dumps(tool_input, indent=2)}")
        
        tool_result = execute_tool(tool_name, tool_input)
        print(f"   Result: {tool_result}\n")
        
        # Add tool result to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(tool_result)
                }
            ]
        })
        
        # Get next response
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages
        )
    
    # Extract final text response
    final_response = ""
    for block in response.content:
        if hasattr(block, 'text'):
            final_response = block.text
            break
    
    print(f"✅ Agent Response:\n{final_response}")
    return final_response

# Test the agent
if __name__ == "__main__":
    # Try different requests
    run_agent("What's the weather in San Francisco?")
    print("\n" + "="*50 + "\n")
    run_agent("Calculate 15% of 200")
```

## Step 4: Run Your Agent

```bash
python agent.py
```

You should see output like:

```
🤖 Agent: Processing request...
📝 User: What's the weather in San Francisco?

🔧 Tool Used: get_weather
   Input: {"location": "San Francisco"}
   Result: Weather in San Francisco: Sunny, 72°F

✅ Agent Response:
The current weather in San Francisco is sunny with a temperature of 72°F. It's a beautiful day!
```

## Step 5: Add More Tools

Extend your agent by adding more tools:

```python
# Add to TOOLS list:
{
    "name": "search",
    "description": "Search the internet",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
}
```

Then implement it in `execute_tool()`:

```python
elif tool_name == "search":
    # Call a real search API here
    return f"Search results for: {tool_input['query']}"
```

## Step 6: Optimize Costs

**Key cost optimization tips:**

1. **Use Claude Haiku** for simple tasks ($0.80/1M tokens)
2. **Cache system prompts** to reuse context
3. **Batch requests** when possible
4. **Monitor token usage** with `response.usage`

Example:

```python
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Estimated cost: ${(response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.015) / 1000}")
```

## Common Pitfalls

❌ **Infinite loops**: Always have a max iteration count
❌ **Hallucinating tools**: Claude might invent tools that don't exist
❌ **Too many tools**: Limit to 5-10 tools per agent
❌ **Poor tool descriptions**: Be specific about what each tool does

## Next Steps

1. **Deploy to production** - Use OpenClaw or Fly.io
2. **Add memory** - Store conversation history
3. **Multi-agent orchestration** - Combine multiple agents
4. **Add error handling** - Graceful failure modes
5. **Monitor performance** - Track success rates and costs

## Resources

- [Anthropic Documentation](https://docs.anthropic.com)
- [OpenClaw Framework](https://docs.openclaw.ai)
- [Agent Patterns Repo](https://github.com/anthropic-ai/anthropic-sdk-python/tree/main/examples)

## Summary

You've built a functional AI agent that:
- ✅ Understands user requests
- ✅ Decides which tools to use
- ✅ Executes tools and processes results
- ✅ Provides intelligent responses

This is the foundation for building complex autonomous systems. From here, you can add memory, multiple agents, long-term planning, and more!

**Total time:** ~45 minutes  
**Cost:** $0-5 API calls  
**Next difficulty:** Advanced multi-agent orchestration
