import os, json, base64, http.client, urllib.parse, requests, argparse
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

load_dotenv()

# --- ENV ---
SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT1", "")
INDEX_NAME       = os.getenv("AZURE_SEARCH_INDEX_NAME2", "frames_index")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY1", "").strip()

AI_VISION_KEY     = os.getenv("AZURE_AI_VISION_API_KEY", "")
AI_VISION_REGION  = os.getenv("AZURE_AI_VISION_REGION", "")
AI_VISION_ENDPOINT= os.getenv("AZURE_AI_VISION_ENDPOINT", "")

DEFAULT_IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "frames"))
DEFAULT_OUTPUT_JSON= Path(os.getenv("OUTPUT_JSON", "output/frameVectors.json"))

# --- Helpers ---
def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def _collect_images(root: Path):
    if root.is_file() and _is_image_file(root):
        yield root; return
    for p in root.rglob("*"):
        if _is_image_file(p):
            yield p

def _normalize_region(r: str) -> str:
    return r.strip().lower().replace(" ", "") if r else r

def _safe_key(s: str) -> str:
    try:
        return base64.urlsafe_b64encode(str(s).encode("utf-8")).decode("ascii")
    except Exception:
        import re
        return re.sub(r"[^A-Za-z0-9_\-=]", "-", str(s))

def get_image_vector(image_path, key, region, endpoint=None):
    headers = {'Ocp-Apim-Subscription-Key': key}
    params = urllib.parse.urlencode({'model-version': '2023-04-15'})

    # Decide host
    if endpoint and str(endpoint).strip():
        parsed = urllib.parse.urlparse(str(endpoint).strip())
        host = parsed.netloc
        if not host:
            raise Exception(f"Invalid AZURE_AI_VISION_ENDPOINT: {endpoint}")
    else:
        host = f"{region}.api.cognitive.microsoft.com"

    # Decide body
    if isinstance(image_path, (str, os.PathLike)) and str(image_path).startswith(("http://","https://")):
        headers['Content-Type'] = 'application/json'
        body = json.dumps({"url": str(image_path)})
    else:
        headers['Content-Type'] = 'application/octet-stream'
        with open(image_path, "rb") as f:
            body = f.read()

    conn = http.client.HTTPSConnection(host, timeout=10)
    url_path = f"/computervision/retrieval:vectorizeImage?api-version=2024-02-01&{params}"
    conn.request("POST", url_path, body, headers)
    resp = conn.getresponse()
    raw = resp.read()
    try:
        data = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        data = {"raw": raw[:200].decode("utf-8", errors="replace") if raw else ""}
    conn.close()

    if resp.status != 200:
        msg = (isinstance(data, dict) and (data.get("message") or data.get("error") or data.get("code") or data.get("raw"))) or ""
        raise Exception(f"Vision API {resp.status} {resp.reason}: {msg}")

    vec = data.get("vector")
    if not isinstance(vec, list):
        raise Exception(f"No 'vector' in response for {image_path}")
    return vec

def embed_folder(images_dir: Path = DEFAULT_IMAGES_DIR, output_json: Path = DEFAULT_OUTPUT_JSON):
    if not AI_VISION_KEY or not AI_VISION_REGION:
        raise RuntimeError("Missing AZURE_AI_VISION_API_KEY or AZURE_AI_VISION_REGION")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    images = list(_collect_images(images_dir))
    if not images:
        return {"count": 0, "docs_path": str(output_json), "docs": []}

    region = _normalize_region(AI_VISION_REGION)
    docs = []
    for img in images:
        vec = get_image_vector(str(img), AI_VISION_KEY, region, AI_VISION_ENDPOINT)
        rel = str(img.relative_to(images_dir)).replace("\\", "/") if img.is_relative_to(images_dir) else img.name
        docs.append({
            "id": rel,  # will be base64-safe’d before upload
            "description": img.stem.replace("_", " ").replace("-", " "),
            "image_vector": vec,
        })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    return {"count": len(docs), "docs_path": str(output_json)}

def create_or_update_index(index_name: str = INDEX_NAME):
    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)
    ic = SearchIndexClient(endpoint=SERVICE_ENDPOINT, credential=cred)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="description", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
        SearchField(
            name="image_vector",
            hidden=False,
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1024,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4, ef_construction=400, ef_search=1000, metric=VectorSearchAlgorithmMetric.COSINE
                ),
            ),
        ],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
    )

    idx = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    ic.create_or_update_index(idx)
    return {"index": index_name, "status": "created_or_updated"}

def upload_docs(output_json: Path = DEFAULT_OUTPUT_JSON, index_name: str = INDEX_NAME):
    with open(output_json, "r", encoding="utf-8") as f:
        docs = json.load(f)

    for d in docs:
        if isinstance(d, dict) and "id" in d:
            d["id"] = _safe_key(d["id"])

    cred = AzureKeyCredential(SEARCH_ADMIN_KEY)
    sc = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=index_name, credential=cred)
    sc.upload_documents(docs)
    return {"uploaded": len(docs)}

# def frames():
#     images_dir = DEFAULT_IMAGES_DIR
#     output_json = DEFAULT_OUTPUT_JSON
#     index_name = INDEX_NAME

#     # 1. Embed all images
#     res = embed_folder(images_dir=images_dir, output_json=output_json)
#     print(json.dumps({"step": "embed", **res}, indent=2))

#     # 2. Create or update index
#     res = create_or_update_index(index_name=index_name)
#     print(json.dumps({"step": "index", **res}, indent=2))

#     # 3. Upload JSON docs
#     res = upload_docs(output_json=output_json, index_name=index_name)
#     print(json.dumps({"step": "upload", **res}, indent=2))
def frames():
    images_dir = DEFAULT_IMAGES_DIR
    output_json = DEFAULT_OUTPUT_JSON
    index_name = INDEX_NAME

    result = {
        "step": "frames",
        "status": "failed",
        "error": None,
        "embed": None,
        "index": None,
        "upload": None
    }

    try:
        # Validate environment variables
        if not all([AI_VISION_KEY, AI_VISION_REGION, SERVICE_ENDPOINT, SEARCH_ADMIN_KEY]):
            raise ValueError("Missing required environment variables: AZURE_AI_VISION_API_KEY, AZURE_AI_VISION_REGION, AZURE_SEARCH_ENDPOINT1, or AZURE_SEARCH_ADMIN_KEY1")

        # 1. Embed all images
        res = embed_folder(images_dir=images_dir, output_json=output_json)
        result["embed"] = res
        print(json.dumps({"step": "embed", **res}, indent=2))

        # 2. Create or update index
        res = create_or_update_index(index_name=index_name)
        result["index"] = res
        print(json.dumps({"step": "index", **res}, indent=2))

        # 3. Upload JSON docs
        res = upload_docs(output_json=output_json, index_name=index_name)
        result["upload"] = res
        print(json.dumps({"step": "upload", **res}, indent=2))

        result["status"] = "success"
    except Exception as e:
        result["error"] = str(e)
        print(json.dumps(result, indent=2))

    return result

if __name__ == "__main__":
    frames()