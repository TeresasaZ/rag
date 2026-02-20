from qdrant_client import QdrantClient, models
import os, json
from pathlib import Path
from typing import Dict

import logging

logging.basicConfig(
    level=logging.DEBUG
)

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DATA_PATH = os.getenv("DATA_PATH", "")
COLLECTION_NAME = os.getenv('COLLECTION_NAME', "met-museum-artworks")

def prepare_painting_description(painting_obj: Dict) -> Dict:
    painting_data = painting_obj

    intro_statement = f"{painting_data.get('title', '')} by {painting_data.get('artistDisplayName')}"

    artwork_description = f'The description of the artwork is: "{painting_data.get("itemDescription")}'

    gallery_number = f"The artwork is located in gallery number {painting_data.get('GalleryNumber', 'unknown')}." if painting_data.get('GalleryNumber') else ""

    knowledge_text = f"{intro_statement}.{artwork_description}.{gallery_number}"

    return {
        "objectID": painting_data.get("objectID"),
        "knowledge_text": knowledge_text,
        "primary_image": painting_data.get("primaryImage"),
        "galleryMapLocationLink": painting_data.get("galleryLink")
    }

client = QdrantClient(url=QDRANT_URL)

collection_names = [collection.name for collection in client.get_collections().collections]

if COLLECTION_NAME not in collection_names:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "jina-small": models.VectorParams(size = 512, distance=models.Distance.COSINE) # cosine number comparison
        }
    )

    json_path = Path(DATA_PATH)

    with open(json_path, 'r') as f:
        data = json.load(f)

    artwork_obj_lst = []

    for painting in data[:100]:
        prepared_data = prepare_painting_description(painting)

        artwork_obj_lst.append(prepared_data)

    client.upsert(
        collection_name = COLLECTION_NAME,
        points = [
            models.PointStruct(
            id = artwork['objectID'],
            vector = {
                "jina-small": models.Document(
                    text = artwork["knowledge_text"],
                    model="jinaai/jina-embeddings-v2-small-en"
                )
            },
            payload = {
                "artwork_text": artwork["knowledge_text"],
                "artwork_image_url": artwork["primary_image"],
                "artwork_gallery_link": artwork['galleryMapLocationLink']
            }
            )
            for artwork in artwork_obj_lst
        ]
    )
