import os, json, argparse, time, faiss, pickle
from openai import OpenAI
from libs.jsonl import read_jsonl, write_jsonl
from libs.embedding import get_embeddings, load_usecases_and_testcases
from libs.generateTestCase import generate_testcases
from libs.evaluation import calculate_bert_score, get_score


#--------------------------------------------
#Load dataset and create folder of results
#--------------------------------------------
DATASET_PATH = "dataset/dataset-with-description.jsonl"
RESULTS_PATH = "results/GPT4o-results-with-description.jsonl"

if os.path.exists(RESULTS_PATH):

    # Create a backup name with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = RESULTS_PATH.replace(".jsonl", f"-backup-{timestamp}.jsonl")

    # Rename (or move) the existing file
    os.rename(RESULTS_PATH, backup_path)
    
with open(RESULTS_PATH, mode="w", encoding='utf-8') as file:
    file.write("")    
    
dataset = read_jsonl(DATASET_PATH)[:100]


#--------------------------------------------
#Create a client for access OpenAI endpoint
#--------------------------------------------
API_KEY = os.getenv("OPENAI_API_KEY")
print("API_KEY: ",API_KEY)
client = OpenAI(api_key=API_KEY)
if client is None:
    raise ValueError("OpenAI client not provided or failed to initialize.")
else: 
    print("OpenAI client is successfully created")
    

results = read_jsonl(RESULTS_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--embedding", 
            type=str,
            default="False",
            help="To generate test case with embedding or not")
    args = parser.parse_args()
    args.embedding = args.embedding.lower() == "true"
     
 
    for idx, data in enumerate(dataset):
        if len(results) > idx:
            continue

        usecase = data["usecase"]

        if "author" in usecase: del usecase["author"]
        if "id" in usecase: del usecase["id"]

        usecase = json.dumps(usecase, indent=4)
        
        try:
            testcases = generate_testcases(usecase, data["project_description"], client, embedding=args.embedding)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

        p, r, f1 = calculate_bert_score(
            reference=json.dumps(data["testcases"], indent=4),
            candidate=json.dumps(testcases, indent=4),
        )

        results.append({
            "usecase": data["usecase"],
            "testcases": data["testcases"],
            "GPT4omini_testcases": testcases,
            "bert_score": {
                "Precision": p,
                "Recall": r,
                "F1": f1
            }
        })

        write_jsonl(RESULTS_PATH, results)    
        # break
    
    score_summary = get_score(RESULTS_PATH)
    
