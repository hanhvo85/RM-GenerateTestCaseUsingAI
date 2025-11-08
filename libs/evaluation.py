from bert_score import score
from libs.jsonl import *

def calculate_bert_score(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="en", verbose=False)
    return P.mean().item(), R.mean().item(), F1.mean().item()
    
def get_score(RESULTS_PATH):

    results = read_jsonl(RESULTS_PATH)
    precisions, recalls, f1_scores = [], [], []
    for res in results:
        precisions.append(res["bert_score"]["Precision"])
        recalls.append(res["bert_score"]["Recall"])
        f1_scores.append(res["bert_score"]["F1"])

    print(f"Average Precision: {(sum(precisions) / len(precisions)) * 100:0.2f}")
    print(f"Average Recall: {(sum(recalls) / len(recalls)) * 100:0.2f}")
    print(f"Average F1: {(sum(f1_scores) / len(f1_scores)) * 100:0.2f}")
	