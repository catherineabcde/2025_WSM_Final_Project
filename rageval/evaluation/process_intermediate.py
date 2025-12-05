import json
import os
from typing import Dict, List, Any
import argparse

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def calculate_averages(data: List[Dict[str, Any]], metric_list: List[str]) -> Dict[str, float]:
    metric_sums = {metric: 0 for metric in metric_list}
    metric_counts = {metric: 0 for metric in metric_list}
    
    for item in data:
        for metric in metric_list:
            if metric in item:
                metric_sums[metric] += item[metric]
                metric_counts[metric] += 1
    
    averages = {metric: metric_sums[metric] / metric_counts[metric] for metric in metric_list if metric_counts[metric] != 0}
    print(averages)

    # Calculate F1 scores
    if 'Words_Precision' in averages and 'Words_Recall' in averages:
        precision = averages['Words_Precision']
        recall = averages['Words_Recall']
        if precision + recall > 0:
            averages['Words_F1'] = 2 * (precision * recall) / (precision + recall)
        else:
            averages['Words_F1'] = 0.0

    if 'Sentences_Precision' in averages and 'Sentences_Recall' in averages:
        precision = averages['Sentences_Precision']
        recall = averages['Sentences_Recall']
        if precision + recall > 0:
            averages['Sentences_F1'] = 2 * (precision * recall) / (precision + recall)
        else:
            averages['Sentences_F1'] = 0.0
            
    if 'completeness' in averages and 'hallucination' in averages:
        completeness = averages['completeness']
        hallucination = averages['hallucination']
        averages['factual_score'] = completeness - hallucination

    # Calculate generation total score
    if 'factual_score' in averages and 'ROUGELScore' in averages:
        averages['Generation_Total_Score'] = (0.5 * averages['factual_score'] + 0.5 * averages['ROUGELScore'])
    
    # Calculate average of F1 scores
    f1_scores = []
    if 'Words_F1' in averages:
        f1_scores.append(averages['Words_F1'])
    if 'Sentences_F1' in averages:
        f1_scores.append(averages['Sentences_F1'])
        
    if f1_scores:
        averages['Retrieval_Total_Score'] = sum(f1_scores) / len(f1_scores)

    return averages


def process_file(file_path: str, output_file: str, metric_list: List[str]):
    results = {}
    
    data = load_jsonl(file_path)
    averages = calculate_averages(data, metric_list)
    results[file_path] = averages
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--file_path', help='Path to the folder containing the result files')
    argparser.add_argument('--output_file', help='Path to the output file')
    args = argparser.parse_args()
    
    file_path = args.file_path
    output_file = args.output_file
    metric_list = ['Sentences_Precision', 'Sentences_Recall', "Words_Precision", "Words_Recall", 'ROUGELScore', "completeness", "hallucination", "irrelevance"]
    
    process_file(file_path, output_file, metric_list)
    print(f"Results saved to {output_file}")