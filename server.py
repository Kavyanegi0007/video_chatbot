import sys
import logging
from mcp.server.fastmcp import FastMCP
from app_tools.audio import process_and_index_audio
from app_tools.vid import extract_frames
from app_tools.frames import frames
from app_tools.gpt_desc import gpt_desc
from pathlib import Path
from typing import List, Dict, Optional, Any

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("video_chatbot")

@mcp.tool(name="video", description="Extract frames from a video")
async def video(video_path: str) -> List[Dict[str, str]]:
    """
    Extract frames from a video and save to frames directory.
    Returns list of frame metadata (frame_id, path).
    """
    output_dir = Path("frames")
    return extract_frames(str(video_path), str(output_dir), fps=0.5, max_frames=20)

@mcp.tool(name="audio", description="Process audio from a video and upload to Azure AI Search")
async def audio(video_path: str, video_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Process audio from a video, transcribe, embed, and index.
    Returns status and document count.
    """
    process_and_index_audio(video_path, video_id)
    return {"status": "processed", "video_path": video_path}

# @mcp.tool(name="frames", description="Embed frames from video and upload to Azure AI Search")
# async def frames_tool() -> Dict[str, Any]:
#     """
#     Embed frames in frames directory and index in Azure AI Search.
#     Returns embedding and indexing status.
#     """
#     return frames()

# @mcp.tool(name="gpt_desc", description="Generate descriptions using GPT and upload to Azure AI Search")
# async def gpt_desc_tool() -> Dict[str, Any]:
#     """
#     Generate GPT descriptions for frames and index in Azure AI Search.
#     Returns description and indexing status.
#     """
#     return gpt_desc()
@mcp.tool(name="frames", description="Embed frames from video and upload to Azure AI Search")
async def frames_tool() -> Dict[str, Any]:
    """
    Embed frames in frames directory and index in Azure AI Search.
    Returns embedding and indexing status.
    """
    try:
        result = frames()
        if not isinstance(result, dict):
            return {"status": "failed", "error": f"Frames processing returned invalid output: {result}"}
        return result
    except Exception as e:
        return {"status": "failed", "error": f"Frames tool failed: {str(e)}"}

@mcp.tool(name="gpt_desc", description="Generate descriptions using GPT and upload to Azure AI Search")
async def gpt_desc_tool() -> Dict[str, Any]:
    """
    Generate GPT descriptions for frames and index in Azure AI Search.
    Returns description and indexing status.
    """
    try:
        result = gpt_desc()
        if not isinstance(result, dict):
            return {"status": "failed", "error": f"GPT description processing returned invalid output: {result}"}
        return result
    except Exception as e:
        return {"status": "failed", "error": f"GPT description tool failed: {str(e)}"}
@mcp.tool(name="health", description="Check server health")
async def health() -> str:
    return "ok"

if __name__ == "__main__":
    mcp.run(transport="stdio")