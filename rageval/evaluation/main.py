import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from metrics import get_metric

def init_worker(evaluator_names, use_openai=True, model='gpt-oss:20b', version='v1'):
    global evaluators
    evaluators = []
    for evaluator_name in evaluator_names:
        if evaluator_name == "keypoint_metrics":
            evaluators.append(get_metric(evaluator_name)(use_openai=use_openai, model=model, version=version))
        else:
            evaluators.append(get_metric(evaluator_name)())

    if not evaluators:
        raise ValueError("No correct evaluators are provided")

def process_item(item, language="zh", idx=0, evaluator_names=None, use_openai=False, model='gpt-oss:20b', version='v1'):
    ground_truth = item["ground_truth"]
    results = None 
    init_worker(evaluator_names, use_openai, model, version)
    for evaluator in evaluators:
        result = evaluator(item, ground_truth, results, language=language)
        if evaluator.name != "keypoint_metrics":
            item[evaluator.name] = result
        else:
            item.update(result)
    return idx, item

def process_jsonl(input_file, output_file, evaluator_names, num_workers, use_openai=False, language="zh", model='gpt-oss:20b', version='v1'):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if output_path.exists():
        # delete existing file
        output_path.unlink()
    processed_ids = set()

    with open(input_path, "r", encoding="utf-8") as f_in:
        items = [json.loads(line) for line in f_in]
    if language == "auto":
        items_to_process = [(item, item["language"]) for item in items if item["query"]["query_id"] not in processed_ids]
    else:
        items_to_process = [(item, language) for item in items if item["query"]["query_id"] not in processed_ids]

    init_worker(evaluator_names, use_openai, model, version)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_item, item, language, idx, evaluator_names, use_openai, model, version) for idx, (item, language) in enumerate(items_to_process)]
        
        with open(output_file, "a", encoding="utf-8") as f_out:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
                idx, item = future.result()
                if item is not None:
                    f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f_out.flush()




def main():
    parser = argparse.ArgumentParser(description="Process JSONL file and evaluate completeness.")
    parser.add_argument("--input_file", help="Path to the input JSONL file")
    parser.add_argument("--output_file", help="Path to the output JSONL file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--language", type=str, help="Language for the metric")

    args = parser.parse_args()
    #evaluator_names = ["rouge-l", "words_precision", "words_recall", "sentences_precision", "sentences_recall", "keypoint_metrics"]
    evaluator_names = ["rouge-l", "words_precision", "words_recall", "sentences_precision", "sentences_recall"]
    process_jsonl(args.input_file, args.output_file, evaluator_names, args.num_workers, True, args.language, "gpt-oss:20b", "v1")

if __name__ == "__main__":
    main()