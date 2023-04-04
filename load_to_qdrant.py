from dotenv import load_dotenv
import os
from supabase import create_client, Client
from qdrant_client import QdrantClient
from qdrant_client.http import models
import time
import json


def main():
    collection_name = "generation_outputs"
    load_dotenv()
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(url, key)
    qdrant = QdrantClient("localhost", port=6333)
    res = qdrant.get_collections()
    has_collection = False
    for collection in res.collections:
        if collection.name == collection_name:
            has_collection = True
            break
    if not has_collection:
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024, distance=models.Distance.COSINE
            ),
        )
    has_more_generation_outputs = True
    last_created_at = "2100-01-01T00:00:00.000000+00:00"
    # load last created at from a txt file
    try:
        with open("last_created_at.txt", "r") as f:
            last_created_at = f.read()
            print(f"Found last created at in txt: {last_created_at}")
    except Exception as e:
        print(f"Last created at file not found, will create one.")
    loaded = 0
    while has_more_generation_outputs:
        try:
            res = (
                supabase.from_("generation_outputs")
                .select(
                    "id,created_at,embedding,image_path,generation:generation_id(prompt:prompt_id(text))"
                )
                .lt("created_at", last_created_at)
                .order("created_at", desc=True)
                .limit(1000)
                .execute()
            )
            data = res.data
            if data is None:
                raise Exception("data is None")
            if len(data) == 0:
                has_more_generation_outputs = False
                break
            flat_data = []
            for row in data:
                if (
                    row["generation"] is None
                    or row["generation"]["prompt"] is None
                    or row["generation"]["prompt"]["text"] is None
                    or row["embedding"] is None
                    or row["image_path"] is None
                    or row["id"] is None
                ):
                    continue
                embedding = json.loads(row["embedding"])
                if type(embedding) != list or len(embedding) != 1024:
                    continue
                flat_data.append(
                    {
                        "id": row["id"],
                        "created_at": row["created_at"],
                        "image_path": row["image_path"],
                        "prompt": row["generation"]["prompt"]["text"],
                        "embedding": embedding,
                    }
                )
            points = []
            for row in flat_data:
                embedding = row["embedding"]
                try:
                    points.append(
                        models.PointStruct(
                            id=row["id"],
                            vector=embedding,
                            payload={
                                "created_at": row["created_at"],
                                "image_path": row["image_path"],
                                "prompt": row["prompt"],
                            },
                        )
                    )
                except Exception as e:
                    print(f"Couldn't create PointStruce, error: {e}")
                    print(f"Skipping this row")
            try:
                qdrant.upsert(
                    collection_name=collection_name,
                    points=points,
                )
            except Exception as e:
                print(f"Qdrant Error: {e}")
                raise e

            last_created_at = flat_data[-1]["created_at"]
            # record last created at to a txt file
            with open("last_created_at.txt", "w") as f:
                f.write(last_created_at)
            loaded += len(flat_data)
            print(f"Total loaded: {loaded}")
            print(f"Last loaded item: {last_created_at}")
        except Exception as e:
            print(f"Error: {e}")
            print("Sleeping for 2 seconds and trying again...")
            time.sleep(2)


if __name__ == "__main__":
    load_dotenv()
    main()
