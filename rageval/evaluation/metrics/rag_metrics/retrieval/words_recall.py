from .utils import split_words
from collections import Counter

class Words_Recall:
    name:str = "Words_Recall"
    def __init__(self):
        pass

    def calculate_recall(self, retrieves, ground_truths, language=None) -> float:
        if not retrieves or not ground_truths:
            return 0.0
            
        retrieved_words = set(split_words(retrieves, language))
        ground_truth_words = set(split_words(ground_truths, language))

        if not ground_truth_words:
            return 0.0

        retrieved_word_counts = Counter(retrieved_words)
        ground_truth_word_counts = Counter(ground_truth_words)

        common_word_counts = retrieved_word_counts & ground_truth_word_counts
        
        recall = sum(common_word_counts.values()) / sum(ground_truth_word_counts.values()) if sum(ground_truth_word_counts.values()) > 0 else 0.0
        return recall

    def __call__(self, doc, ground_truth, results, language) -> float:
        retrieves = [r for r in doc["prediction"].get("references", [])]  # Retrieves from prediction
        ground_truths =doc['ground_truth'].get('references', [])
        return self.calculate_recall(retrieves, ground_truths, language=language)