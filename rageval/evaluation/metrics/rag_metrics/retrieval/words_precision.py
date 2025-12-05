from .utils import split_words
from collections import Counter


class Words_Precision:
    name:str = "Words_Precision"
    def __init__(self, topk = 5):
        self.topk = topk

    def calculate_precision(self, retrieves, ground_truths, language=None) -> float:
        retrieves = retrieves[:self.topk]
        
        if not retrieves or not ground_truths:
            return 0.0
            
        retrieved_words = set(split_words(retrieves, language))
        ground_truth_words = set(split_words(ground_truths, language))

        if not retrieved_words:
            return 0.0

        retrieved_word_counts = Counter(retrieved_words)
        ground_truth_word_counts = Counter(ground_truth_words)

        common_word_counts = retrieved_word_counts & ground_truth_word_counts
        
        precision = sum(common_word_counts.values()) / sum(retrieved_word_counts.values()) if sum(retrieved_word_counts.values()) > 0 else 0.0
        return precision

    def __call__(self, doc, ground_truth, results, language=None) -> float:
        retrieves = [r for r in doc["prediction"].get("references", [])]  # Retrieves from prediction
        ground_truths =doc['ground_truth'].get('references', [])
        return self.calculate_precision(retrieves, ground_truths, language=language)
