import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from openai import AzureOpenAI
from typing import Any, Tuple
import logging
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from chatbot import run_chatbot  # Assuming this is available

# Configure logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Azure OpenAI configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY2", )
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT2")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_GPT4O")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_GPT4O")

oai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def extract_payload(tool_result: Any) -> Any:
    if hasattr(tool_result, "structuredContent") and tool_result.structuredContent:
        return tool_result.structuredContent
    elif hasattr(tool_result, "content") and tool_result.content:
        if hasattr(tool_result.content, "text"):
            return tool_result.content.text
        return str(tool_result.content)
    else:
        return str(tool_result)

# Tool schemas for LLM
VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "video",
        "description": "Extract frames from video",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Local path to the video"}
            },
            "required": ["video_path"]
        }
    }
}

AUDIO_TOOL = {
    "type": "function",
    "function": {
        "name": "audio",
        "description": "Process audio from video",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Local path to the video"},
                "video_id": {"type": "string", "description": "Optional video ID"}
            },
            "required": ["video_path"]
        }
    }
}

FRAMES_TOOL = {
    "type": "function",
    "function": {
        "name": "frames",
        "description": "Embed frames from video and upload to Azure AI Search",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
}

GPT_DESC_TOOL = {
    "type": "function",
    "function": {
        "name": "gpt_desc",
        "description": "Generate descriptions using GPT and upload to Azure AI Search",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }
}

SYSTEM_PROMPT = (
    "You have access to tools via MCP: 'video' (extract frames from video), 'frames' (embed frames from video), "
    "'gpt_desc' (generate image descriptions using GPT), 'audio' (process audio from video). "
    "When the user provides a video and asks to process it, first call 'video' to generate video frames, then call 'frames' and 'gpt_desc'. "
    "Call 'audio' for audio processing. Use the provided video_id if available. "
    "When presenting results, list results from frames first, then audio results, then gpt_desc results."
)

def extract_payload(tool_result: Any) -> Any:
    if hasattr(tool_result, "structuredContent") and tool_result.structuredContent:
        return tool_result.structuredContent
    elif hasattr(tool_result, "content") and tool_result.content:
        if hasattr(tool_result.content, "text"):
            return tool_result.content.text
        return str(tool_result.content)
    else:
        return {"status": "failed", "error": f"Invalid tool result: {tool_result}"}

async def run_session(user_text: str) -> Tuple[str, dict]:
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=os.environ.copy()
    )
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    tool_payloads = {}

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            while True:
                logger.info(f"Sending messages to LLM: {json.dumps(messages, indent=2)}")
                try:
                    resp = oai_client.chat.completions.create(
                        model=AZURE_DEPLOYMENT,
                        messages=messages,
                        tools=[VIDEO_TOOL, FRAMES_TOOL, GPT_DESC_TOOL, AUDIO_TOOL],
                        tool_choice="auto"
                    )
                except Exception as e:
                    logger.error(f"LLM API call failed: {e}")
                    return f"LLM API error: {str(e)}", tool_payloads

                choice = resp.choices[0]
                logger.info(f"LLM response finish_reason: {choice.finish_reason}")

                if choice.finish_reason != "tool_calls":
                    final_content = choice.message.content or ""
                    logger.info(f"No tool calls, final response: {final_content}")
                    return final_content, tool_payloads

                tool_calls = choice.message.tool_calls or []
                if not tool_calls:
                    logger.warning("Finish reason is 'tool_calls' but no tool_calls found")
                    return choice.message.content or "", tool_payloads

                messages.append({
                    "role": "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        } for tc in tool_calls
                    ]
                })

                for tool_call in tool_calls:
                    func = tool_call.function
                    name = func.name
                    args_str = func.arguments
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse tool arguments: {args_str}, error: {e}")
                        payload = {"status": "failed", "error": f"Invalid arguments: {e}"}
                        tool_payloads[name] = payload
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": json.dumps(payload)
                        })
                        continue

                    logger.info(f"Calling tool: {name} with args: {args}")
                    try:
                        tool_result = await session.call_tool(name, arguments=args)
                        payload = extract_payload(tool_result)
                        if not isinstance(payload, dict):
                            payload = {"status": "failed", "error": f"Tool {name} returned invalid output: {payload}"}
                    except Exception as e:
                        logger.error(f"Tool {name} failed: {e}")
                        payload = {"status": "failed", "error": str(e)}

                    tool_payloads[name] = payload
                    logger.info(f"Tool {name} result: {payload}")

                    tool_response = json.dumps(payload)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": tool_response
                    })

    return "", tool_payloads

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    video_id = uuid.uuid4().hex
    user_text = f"Process this video: {str(video_path)} with video_id: {video_id}"

    try:
        final_text, tool_payloads = asyncio.run(run_session(user_text))
        print("Processing complete. Final LLM response:")
        print(final_text)
        print(f"Tool payloads: {json.dumps(tool_payloads, indent=2)}")
    except Exception as e:
        print(f"Error during session: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Starting chatbot...")
    run_chatbot(video_id)

if __name__ == "__main__":
    main()