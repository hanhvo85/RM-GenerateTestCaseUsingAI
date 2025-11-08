import faiss, pickle, os, glob, json
import numpy as np
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY")
print("API_KEY: ",API_KEY)
client = OpenAI(api_key=API_KEY)
if client is None:
    raise ValueError("OpenAI client not provided or failed to initialize.")
else: 
    print("OpenAI client is successfully created")


def get_embeddings(texts, model="text-embedding-3-large"):
    """
    Generate embeddings for one or more texts using OpenAI's embedding model.
    """
    if isinstance(texts, str):
        texts = [texts]

    response = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in response.data], dtype=np.float32)
    
def load_usecases_and_testcases(folder_path):
    all_texts = []
    all_types = []
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files.")

    for path in jsonl_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)

                    # --- Extract usecase ---
                    uc = obj.get("usecase", "")
                    if isinstance(uc, dict):
                        uc = json.dumps(uc, ensure_ascii=False)
                    if uc:
                        all_texts.append(uc)
                        all_types.append("usecase")

                    # --- Extract testcases ---
                    tcs = obj.get("testcases", [])
                    if isinstance(tcs, list):
                        for tc in tcs:
                            if isinstance(tc, dict):
                                tc = json.dumps(tc, ensure_ascii=False)
                            if tc:
                                all_texts.append(tc)
                                all_types.append("testcase")

                except json.JSONDecodeError:
                    print(f"Skipped malformed line in {path}")

    print(f"Loaded {len(all_texts)} total entries "
          f"({all_types.count('usecase')} usecases + {all_types.count('testcase')} testcases).")

    return all_texts, all_types
    

DATASET_PATH = "dataset/dataset-with-description.jsonl"

def compute_save_embeddings(texts, types, save_dir="libs"):
    os.makedirs(save_dir, exist_ok=True)

    # Compute embeddings in batches
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(model="text-embedding-3-large", input=batch)
        embeds = [d.embedding for d in response.data]
        all_embeddings.extend(embeds)

    embeddings = np.array(all_embeddings, dtype=np.float32)
    print("Embeddings shape:", embeddings.shape)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product for cosine similarity
    index.add(embeddings)

    # Save FAISS + metadata
    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))
    with open(os.path.join(save_dir, "index.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "types": types}, f)

    print(f"FAISS index built with {len(texts)} total entries ({len(set(types))} categories).")
    
    
def retrieve_similar(query, top_k=5, index_path="libs/index.faiss", meta_path="libs/index.pkl"):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    texts = meta["texts"]
    types = meta["types"]

    # Embed query
    q_embed = get_embeddings(query)[0].reshape(1, -1).astype('float32')
    faiss.normalize_L2(q_embed)

    # Search
    distances, indices = index.search(q_embed, top_k)
    results = [(texts[i], types[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
    return results

