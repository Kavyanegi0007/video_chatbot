
import os
import json
import base64
import mimetypes
import http.client
import urllib.parse
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters, HnswAlgorithmConfiguration, SimpleField, SearchField,
    SearchFieldDataType, SearchIndex, VectorSearch, VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric, VectorSearchProfile
)
from azure.search.documents.models import VectorQuery
from typing import Any, Dict, List, Optional, Tuple
from openai import AzureOpenAI

load_dotenv()

# ---- Azure OpenAI for GPT-4o (Second Resource) ----
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY2", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT2", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_GPT4O", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_GPT4O", "")

# ---- Azure OpenAI for Embeddings (Original Resource) ----
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT1", "")
openai_key = os.getenv("AZURE_OPENAI_API_KEY1", "")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING1", "2024-12-01-preview")
openai_embeddings_deployment = "text-embedding-3-small"
openai_embeddings_model = "text-embedding-3-small"

# ---- Azure AI Search ----
GPT_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME3", "gpt_index")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY1", "").strip()
SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT1", "")

# Initialize clients
oai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

embed_client = AzureOpenAI(
    api_key=openai_key,
    api_version=openai_api_version,
    azure_endpoint=openai_endpoint
)

# ---- Helper Functions ----
def extract_video_uuid_and_frame(path_str: str):
    """Assumes filenames are like {uuid}-{frame_no}.jpg"""
    name = Path(path_str).stem
    parts = name.split("-")
    if len(parts) >= 2:
        video_uuid = parts[0]
        frame_no = parts[-1]
        return video_uuid, frame_no
    return None, None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

SYSTEM_PROMPT = (
    "You are an expert visual describer. Return ONLY JSON. Be precise and concise."
)
USER_INSTRUCTION = (
    "Describe the image for retrieval/search. Strict JSON only with keys:\n"
    "{\n"
    '  "description": string,\n'
    '  "objects": [string],\n'
    '  "actions": [string],\n'
    '  "scene": string,\n'
    '  "attributes": [string],\n'
    '  "tags": [string],\n'
    '  "ocr_hint": string\n'
    "}"
)

def _to_data_url(p: Path) -> str:
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _image_input(url_or_path: str) -> Dict[str, Any]:
    if str(url_or_path).startswith(("http://", "https://", "data:")):
        return {"type": "image_url", "image_url": {"url": url_or_path}}
    return {"type": "image_url", "image_url": {"url": _to_data_url(Path(url_or_path))}}

def _normalize_desc(d: Dict[str, Any]) -> Dict[str, Any]:
    d.setdefault("description", "")
    d.setdefault("objects", []);      d["objects"] = [str(x) for x in (d["objects"] if isinstance(d["objects"], list) else ([d["objects"]] if d["objects"] else []))]
    d.setdefault("actions", []);      d["actions"] = [str(x) for x in (d["actions"] if isinstance(d["actions"], list) else ([d["actions"]] if d["actions"] else []))]
    d.setdefault("scene", "");        d["scene"] = str(d["scene"])
    d.setdefault("attributes", []);   d["attributes"] = [str(x) for x in (d["attributes"] if isinstance(d["attributes"], list) else ([d["attributes"]] if d["attributes"] else []))]
    d.setdefault("tags", []);         d["tags"] = [str(x) for x in (d["tags"] if isinstance(d["tags"], list) else ([d["tags"]] if d["tags"] else []))]
    d.setdefault("ocr_hint", "");     d["ocr_hint"] = str(d["ocr_hint"])
    return d

def describe_images_in_folder(
    folder: Path,
    out_jsonl: Optional[Path] = None,
    recursive: bool = True,
    deployment: str = None
) -> List[Dict[str, Any]]:
    deployment = deployment or AZURE_DEPLOYMENT
    paths = (folder.rglob("*") if recursive else folder.glob("*"))
    images = [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    results: List[Dict[str, Any]] = []
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        sink = out_jsonl.open("w", encoding="utf-8")
    else:
        sink = None

    for p in images:
        try:
            content = [
                {"type": "text", "text": USER_INSTRUCTION},
                _image_input(str(p)),
            ]
            resp = oai_client.chat.completions.create(
                model=deployment,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_completion_tokens=450,
            )
            raw = resp.choices[0].message.content or ""
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) >= 2 else raw
                if raw.strip().lower().startswith("json"):
                    raw = raw[4:].strip()
            data = _normalize_desc(json.loads(raw))
            video_uuid, frame_no = extract_video_uuid_and_frame(str(p))
            row = {
                "id": f"{video_uuid}-{frame_no}",
                "video_id": video_uuid,
                "frame_no": frame_no,
                "path": str(p),
                **data
            }

            results.append(row)
            if sink:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ describe failed for {p.name}: {e}")
            continue

    if sink:
        sink.close()
        print(f" Wrote {out_jsonl} ({len(results)} items)")
    return results

# Embeddings
import json, time
from typing import Dict, Any, List

EMBED_DEPLOY = openai_embeddings_deployment

def canonical_text_from_frame(j: Dict[str, Any]) -> str:
    return (
        f"desc: {j.get('description','')}\n"
        f"scene: {j.get('scene','')}\n"
        f"objects: {', '.join(map(str, j.get('objects',[])))}\n"
        f"actions: {', '.join(map(str, j.get('actions',[])))}\n"
        f"attributes: {', '.join(map(str, j.get('attributes',[])))}\n"
        f"tags: {', '.join(map(str, j.get('tags',[])))}\n"
        f"ocr: {j.get('ocr_hint','')}"
    )

def embed_json_single(frame_json: Dict[str, Any]) -> List[float]:
    text = canonical_text_from_frame(frame_json)
    r = embed_client.embeddings.create(model=EMBED_DEPLOY, input=text)
    time.sleep(0.2)
    return r.data[0].embedding

# Creating search index
import hashlib

EMBED_DIMS = 1536

def create_or_update_index(index_name: str = GPT_INDEX_NAME):
    if not SERVICE_ENDPOINT or not SEARCH_ADMIN_KEY:
        raise ValueError("Missing AZURE_SEARCH_ENDPOINT1 or AZURE_SEARCH_ADMIN_KEY1")
    
    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)
    ic = SearchIndexClient(endpoint=SERVICE_ENDPOINT.rstrip("/"), credential=cred)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="video_id", type=SearchFieldDataType.String, searchable=True, filterable=True),
        SearchField(name="frame_no", type=SearchFieldDataType.String, searchable=True, filterable=True),
        SearchField(name="path", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="description", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="scene", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True),
        SearchField(name="objects", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True, filterable=True, facetable=True),
        SearchField(name="actions", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True, filterable=True, facetable=True),
        SearchField(name="attributes", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True),
        SearchField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                    searchable=True, filterable=True, facetable=True),
        SearchField(name="ocr_hint", type=SearchFieldDataType.String, searchable=True),
        SearchField(
            name="image_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIMS,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(
            name="myHnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(m=24, ef_construction=200)
        )],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
    )

    idx = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    ic.create_or_update_index(idx)
    return {"index": index_name, "status": "created_or_updated"}

def upload_docs(output_jsonl: Path, index_name: str = GPT_INDEX_NAME, batch_size: int = 500):
    if not output_jsonl.exists():
        raise FileNotFoundError(f"JSONL not found: {output_jsonl}")

    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)
    sc = SearchClient(endpoint=SERVICE_ENDPOINT.rstrip("/"), index_name=index_name, credential=cred)

    batch = []
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["id"] = hashlib.md5(str(row.get("path","")).encode("utf-8")).hexdigest()
            batch.append(row)
            if len(batch) >= batch_size:
                sc.upload_documents(documents=batch)
                batch.clear()

    if batch:
        sc.upload_documents(documents=batch)

    return {"uploaded": "ok"}

# def gpt_desc():
#     FOLDER = Path(os.getenv("IMAGES_DIR", "frames"))
#     OUT = Path(os.getenv("FRAME_DESCRIPTIONS", "output/frame_descriptions.jsonl"))
#     OUT_VEC = Path(os.getenv("FRAME_VECTORS", "output/frame_descriptions_with_vecs.jsonl"))

#     # 1) Describe images
#     results = describe_images_in_folder(FOLDER, out_jsonl=OUT, recursive=True)
#     print(f"✅ Described {len(results)} images -> {OUT}")

#     # 2) Add embeddings
#     OUT_VEC.parent.mkdir(parents=True, exist_ok=True)
#     with OUT_VEC.open("w", encoding="utf-8") as fout:
#         for i, row in enumerate(results, 1):
#             try:
#                 row["image_vector"] = embed_json_single(row)
#                 fout.write(json.dumps(row, ensure_ascii=False) + "\n")
#                 if i % 25 == 0:
#                     print(f"… embedded {i}/{len(results)}")
#             except Exception as e:
#                 print(f"⚠️ embedding failed for {row.get('path','?')}: {e}")
#                 continue

#     # 3) Create/Update index & push docs
#     create_or_update_index(GPT_INDEX_NAME)
#     upload_docs(OUT_VEC, GPT_INDEX_NAME)
#     print(f"✅ Upserted into Azure Search index: {GPT_INDEX_NAME}")
#     print(f"✅ Wrote vectors -> {OUT_VEC}")
def gpt_desc():
    FOLDER = Path(os.getenv("IMAGES_DIR", "frames"))
    OUT = Path(os.getenv("FRAME_DESCRIPTIONS", "output/frame_descriptions.jsonl"))
    OUT_VEC = Path(os.getenv("FRAME_VECTORS", "output/frame_descriptions_with_vecs.jsonl"))

    result = {
        "step": "gpt_desc",
        "status": "failed",
        "error": None,
        "described": 0,
        "embedded": 0,
        "indexed": "not_attempted"
    }

    try:
        # Validate environment variables
        if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT, openai_key, openai_endpoint, SERVICE_ENDPOINT, SEARCH_ADMIN_KEY]):
            raise ValueError("Missing required environment variables for GPT or Search")

        # 1) Describe images
        results = describe_images_in_folder(FOLDER, out_jsonl=OUT, recursive=True)
        result["described"] = len(results)
        print(f"Described {len(results)} images -> {OUT}")

        # 2) Add embeddings
        OUT_VEC.parent.mkdir(parents=True, exist_ok=True)
        embedded_count = 0
        with OUT_VEC.open("w", encoding="utf-8") as fout:
            for i, row in enumerate(results, 1):
                try:
                    row["image_vector"] = embed_json_single(row)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    embedded_count += 1
                    if i % 25 == 0:
                        print(f"… embedded {i}/{len(results)}")
                except Exception as e:
                    print(f"⚠️ embedding failed for {row.get('path','?')}: {e}")
                    continue
        result["embedded"] = embedded_count

        # 3) Create/Update index & push docs
        create_or_update_index(GPT_INDEX_NAME)
        upload_res = upload_docs(OUT_VEC, GPT_INDEX_NAME)
        result["indexed"] = upload_res["uploaded"]
        result["status"] = "success"
        print(f"✅ Upserted into Azure Search index: {GPT_INDEX_NAME}")
        print(f"✅ Wrote vectors -> {OUT_VEC}")
    except Exception as e:
        result["error"] = str(e)
        print(json.dumps(result, indent=2))

    return result

if __name__ == "__main__":
    gpt_desc()
