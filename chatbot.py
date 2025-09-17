import os
import json
import http.client
from pathlib import Path
from typing import List, Dict, Any
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery  # Use VectorizedQuery
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI Client (unchanged)
oai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY2"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_GPT4O", "2024-08-01-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT2"),
)

# Azure AI Search Clients
search_credentials = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY1"))
audio_search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT1"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME1"),
    credential=search_credentials,
    api_version="2024-05-01-preview"
)
frames_search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT1"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME2"),
    credential=search_credentials,
    api_version="2024-05-01-preview"
)
gpt_search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT1"),
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME3"),
    credential=search_credentials,
    api_version="2024-05-01-preview"
)

# Embedding Client (unchanged)
embed_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY1"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING1"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT1")
)

SYSTEM_PROMPT = (
    "You are a video analysis chatbot. Use provided context from video frames, audio transcriptions, and GPT descriptions to answer queries. "
    "Combine information from all sources to provide a comprehensive and concise response. "
    "If no relevant context is found, say so and offer to process a new video."
)

import numpy as np

# Initialize a random projection matrix (1536 → 1024)
# Ideally, save/load this matrix so queries and index use the same mapping
projection_matrix = np.random.normal(size=(1024, 1536)).astype(np.float32)

def project_to_1024(embedding_1536: list[float]) -> list[float]:
    vec = np.array(embedding_1536, dtype=np.float32)
    if len(vec) != 1536:
        raise ValueError(f"Expected 1536-dim vector, got {len(vec)}")
    projected = projection_matrix @ vec  # 1024-dim
    return projected.tolist()

def generate_embedding(text: str) -> List[float]:
    """Generate 1536-dim embedding for text using text-embedding-3-small."""
    if not text or not isinstance(text, str) or text.strip() == "":
        print(f"Error: Invalid input text for embedding: '{text}'")
        return []
    try:
        response = embed_client.embeddings.create(model="text-embedding-3-small", input=text.strip())
        embedding = response.data[0].embedding
        print(f"Generated 1536-dim embedding for '{text}' (len={len(embedding)})")
        return embedding
    except Exception as e:
        print(f"Embedding error for '{text}': {e}")
        return []

def generate_vision_text_embedding(text: str) -> List[float]:
    """Generate 1024-dim embedding for text using Azure AI Vision retrieval:vectorizeText."""
    try:
        headers = {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_AI_VISION_API_KEY")}
        region = os.getenv("AZURE_AI_VISION_REGION")
        host = f"{region}.api.cognitive.microsoft.com"
        url_path = "/computervision/retrieval:vectorizeText?api-version=2024-02-01"

        body = json.dumps({"text": text})
        conn = http.client.HTTPSConnection(host, timeout=10)
        conn.request("POST", url_path, body, {**headers, "Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()

        vector = data.get("vector", [])
        print(f"Generated 1024-dim embedding for frames query '{text}' (len={len(vector)})")
        return vector
    except Exception as e:
        print(f"Vision embedding error for '{text}': {e}")
        return []

def search_context(query: str, video_id: str = None) -> Dict[str, List[Dict]]:
    """Search across all indices for relevant context."""
    results = {"audio": [], "frames": [], "descriptions": []}

    # --- Audio index (1536-dim) ---
    try:
        query_vector = generate_embedding(query)
        if query_vector:
            audio_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=5,
                fields="content_vector"
            )
            audio_results = audio_search_client.search(
                search_text="",
                vector_queries=[audio_query],
                select=["content", "video_id", "start_time", "end_time"],
                filter=f"video_id eq '{video_id}'" if video_id else None,
                top=5
            )
            results["audio"] = [r for r in audio_results]
            print(f"Audio results: {len(results['audio'])} documents")
    except Exception as e:
        print(f"Audio search error: {e}")

    # --- Frames index (1024-dim) ---
    # try:
    #     frames_query_vector = generate_vision_text_embedding(query)
    #     if frames_query_vector:
    #         frames_query = VectorizedQuery(
    #             vector=frames_query_vector,
    #             k_nearest_neighbors=5,
    #             fields="image_vector"
    #         )
    #         frames_results = frames_search_client.search(
    #             search_text="",
    #             vector_queries=[frames_query],
    #             select=["description", "id"],
    #             top=5
    #         )
    #         results["frames"] = [r for r in frames_results]
    #         print(f"Frames results: {len(results['frames'])} documents")
    # except Exception as e:
    #     print(f"Frames search error: {e}")
    # --- Frames index (1024-dim) ---
    try:
        frames_query_vector = project_to_1024(generate_embedding(query))  # project 1536 → 1024
        if frames_query_vector:
            frames_query = VectorizedQuery(
                vector=frames_query_vector,
                k_nearest_neighbors=5,
                fields="image_vector"
            )
            frames_results = frames_search_client.search(
                search_text="",
                vector_queries=[frames_query],
                select=["description", "id"],
                top=5
            )
            results["frames"] = [r for r in frames_results]
            print(f"Frames results: {len(results['frames'])} documents")
    except Exception as e:
        print(f"Frames search error: {e}")

    # --- GPT descriptions index (1536-dim) ---
    try:
        query_vector = generate_embedding(query)
        if query_vector:
            gpt_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=5,
                fields="image_vector"
            )
            gpt_results = gpt_search_client.search(
                search_text="",
                vector_queries=[gpt_query],
                select=["id", "video_id", "frame_no", "path", "description", "scene", "objects", "actions", "attributes", "tags", "ocr_hint"],
                #filter=f"video_id eq '{video_id}'" if video_id else None,
                top=5
            )
            results["descriptions"] = [{
                "id": r.get("id"),
                "video_id": r.get("video_id"),
                "frame_no": r.get("frame_no"),
                "path": r.get("path"),
                "description": r.get("description"),
                "scene": r.get("scene"),
                "objects": r.get("objects", []),
                "actions": r.get("actions", []),
                "attributes": r.get("attributes", []),
                "tags": r.get("tags", []),
                "ocr_hint": r.get("ocr_hint"),
            } for r in gpt_results]
            print(f"GPT results: {len(results['descriptions'])} documents")
    except Exception as e:
        print(f"Descriptions search error: {e}")

    return results

def check_gpt_index_contents():
    try:
        results = gpt_search_client.search(
            search_text="*",
            select=["id", "video_id", "description"],
            top=10
        )
        docs = [r for r in results]
        print(f"Found {len(docs)} documents in gpt_index:")
        for doc in docs:
            print(f"ID: {doc['id']}, Video ID: {doc['video_id']}, Description: {doc['description']}")
        return docs
    except Exception as e:
        print(f"Error checking gpt_index: {e}")
        return []

def generate_response(query: str, context: Dict[str, List[Dict]], video_id: str = None) -> str:
    """Generate a conversational response using GPT-4o."""
    context_text = "Retrieved Context:\n"
    if context["audio"]:
        context_text += "Audio Transcriptions:\n" + "\n".join([f"- {r['content']} (t={r['start_time']}-{r['end_time']}s)" for r in context["audio"]]) + "\n"
    if context["frames"]:
        context_text += "Frame Descriptions:\n" + "\n".join([f"- {r['description']} (id={r['id']})" for r in context["frames"]]) + "\n"
    if context["descriptions"]:
        context_text += "GPT Descriptions:\n" + "\n".join([f"- {r['description']} (Scene: {r['scene']}, Objects: {', '.join(r['objects'])}, Actions: {', '.join(r['actions'])})" for r in context["descriptions"]]) + "\n"
    if not any(context.values()):
        context_text += "No relevant context found.\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\n{context_text}"}
    ]

    try:
        response = oai_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_GPT4O", "gpt-4o-08-06"),
            messages=messages,
            temperature=0.7,
            max_completion_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

def run_chatbot(video_id: str = None):
    """Run the conversational chatbot loop."""
    print("Welcome to the Video Chatbot! Type 'exit' to quit.")
    while True:
        query = input("Your query: ")
        if query.lower() == "exit":
            break
        print("Checking gpt_index contents...")
        check_gpt_index_contents()
        context = search_context(query, video_id)
        response = generate_response(query, context, video_id)
        print(f"Bot: {response}\n")