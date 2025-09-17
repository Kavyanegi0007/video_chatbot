import os
import uuid
import librosa
import soundfile as sf
from openai import OpenAI
from moviepy.video.io.VideoFileClip import VideoFileClip
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Make sure .env is loaded

# Create Azure client
# Audio transcription client (Whisper / GPT-4o transcribe)
audio_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY1"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_AUDIO1"),  # e.g. "2024-05-01-preview"
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT1")
)

# Embeddings client
embedding_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY1"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBEDDING1"),  # e.g. "2023-12-01-preview"
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT1")
)
# Initialize clients
#openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT1")
azure_search_credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY1"))
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME1")
search_index_client = SearchIndexClient(endpoint=azure_search_endpoint, credential=azure_search_credential)
search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=azure_search_credential)

# Embedding and transcription model config
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
import time
from functools import wraps

def timer(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f" Starting '{func.__name__}'...")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"'{func.__name__}' finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")
        return result
    return wrapper

def create_or_update_index():
    """Create Azure AI Search index with vector support using provided HNSW config."""
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSION,
            vector_search_profile_name="myHnswProfile",
        ),
        SearchField(name="video_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchField(name="segment_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="start_time", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SearchField(name="end_time", type=SearchFieldDataType.Double, filterable=True, sortable=True),
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

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="default_semantic_config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")]
                ),
            )
        ]
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
    
    try:
        search_index_client.create_or_update_index(index)
        print(f"Index '{index_name}' created or updated.")
    except Exception as e:
        print(f"Error creating index: {e}")

def extract_audio_from_video(video_path, output_audio_path="extracted_audio.wav"):
    """Extract audio from video file."""
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path)
        video.close()
        print(f"Audio extracted to {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

def split_audio(audio_path, segment_duration=10.0):
    """Split audio into 10-second segments using librosa and save as WAV."""
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    segments = []
    start_time = 0.0

    while start_time < duration:
        end_time = min(start_time + segment_duration, duration)
        if (end_time - start_time) >= 1.0:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]

            segment_path = f"temp_segment_{int(start_time)}.wav"
            sf.write(segment_path, segment, sr, format="wav")

            segments.append({
                "path": segment_path,
                "start_time": start_time,
                "end_time": end_time,
            })
        start_time += segment_duration

    return segments

# def transcribe_segment(segment_path):
#     """Transcribe audio segment using GPT-4o-transcribe."""
#     try:
#         with open(segment_path, "rb") as audio_file:
#             transcription = audio_client.audio.transcriptions.create(
#                 model=TRANSCRIPTION_MODEL,
#                 file=audio_file,
#                 response_format="text"
#             )
#         return transcription.text
#     except Exception as e:
#         print(f"Transcription error for {segment_path}: {e}")
#         return ""
def transcribe_segment(segment_path):
    """Transcribe audio segment using GPT-4o-transcribe."""
    try:
        with open(segment_path, "rb") as audio_file:
            transcription = audio_client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=audio_file
            )
        return transcription.text  # now valid
    except Exception as e:
        print(f"Transcription error for {segment_path}: {e}")
        return ""

def generate_embedding(text):
    """Generate embedding for text chunk."""
    try:
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

import json

# def process_and_index_audio(video_path, video_id=None):
#     """Main pipeline: Extract, split, transcribe, chunk/embed, index."""
#     if video_id is None:
#         video_id = os.path.splitext(os.path.basename(video_path))[0]
    
#     audio_path = extract_audio_from_video(video_path)
#     create_or_update_index()
#     segments = split_audio(audio_path)
#     documents = []
    
#     for idx, segment in enumerate(segments):
#         try:
#             text = transcribe_segment(segment["path"])
#             if not text.strip():
#                 continue
#             embedding = generate_embedding(text)
#             if embedding is None:
#                 continue
#             doc = {
#                 "id": str(uuid.uuid4()),
#                 "content": text,
#                 "content_vector": embedding,
#                 "video_id": video_id,
#                 "segment_id": f"{video_id}_segment_{idx}",
#                 "start_time": segment["start_time"],
#                 "end_time": segment["end_time"],
#             }
#             documents.append(doc)
#         except Exception as e:
#             print(f"Error processing segment {idx}: {e}")
#         finally:
#             if os.path.exists(segment["path"]):
#                 os.remove(segment["path"])
    
#     if os.path.exists(audio_path):
#         os.remove(audio_path)
    
#     if documents:
#         # Save locally before uploading
#         json_file = f"{video_id}_documents.json"
#         with open(json_file, "w", encoding="utf-8") as f:
#             json.dump(documents, f, ensure_ascii=False, indent=2)
#         print(f"Saved {len(documents)} documents locally to {json_file}")

#         try:
#             search_client.merge_or_upload_documents(documents)
#             print(f"Indexed {len(documents)} chunks for video_id: {video_id}")
#         except Exception as e:
#             print(f"Error indexing documents: {e}")
@timer
def process_and_index_audio(video_path, video_id=None):
    """Main pipeline: Extract, split, transcribe, chunk/embed, index."""
    if video_id is None:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    print(f"Processing audio for video_id: {video_id}")
    try:
        audio_path = extract_audio_from_video(video_path)
        print(f"Extracted audio to: {audio_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not created: {audio_path}")
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return {"status": "failed", "error": str(e)}

    create_or_update_index()
    segments = split_audio(audio_path)
    print(f"Split audio into {len(segments)} segments")
    
    documents = []
    for idx, segment in enumerate(segments):
        try:
            text = transcribe_segment(segment["path"])
            if not text.strip():
                print(f"Segment {idx} is empty or silent")
                continue
            print(f"Segment {idx} transcribed: {text[:50]}...")
            embedding = generate_embedding(text)
            if not embedding or len(embedding) != EMBEDDING_DIMENSION:
                print(f"Segment {idx} embedding failed or invalid (length: {len(embedding) if embedding else 0})")
                continue
            doc = {
                "id": str(uuid.uuid4()),
                "content": text,
                "content_vector": embedding,
                "video_id": video_id,
                "segment_id": f"{video_id}_segment_{idx}",
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
            }
            documents.append(doc)
            print(f"Segment {idx} prepared for indexing")
        except Exception as e:
            print(f"Error processing segment {idx}: {e}")
        finally:
            if os.path.exists(segment["path"]):
                os.remove(segment["path"])
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    if documents:
        json_file = f"{video_id}_documents.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(documents)} documents locally to {json_file}")

        try:
            search_client.merge_or_upload_documents(documents)
            print(f"Indexed {len(documents)} chunks for video_id: {video_id}")
            return {"status": "success", "documents_indexed": len(documents)}
        except Exception as e:
            print(f"Error indexing documents: {e}")
            return {"status": "failed", "error": str(e)}
    else:
        print("No documents to index")
        return {"status": "failed", "error": "No valid audio segments processed"}
    
if __name__ == "__main__":
    #video_path = "motivation.mp4"  # Confirm your video file path
    process_and_index_audio(video_path)