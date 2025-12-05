from .rag_metrics.generation.rouge_l import ROUGELScore
from .rag_metrics.retrieval.eir_precision import EIR_Precision
from .rag_metrics.retrieval.eir_recall import EIR_Recall
from .rag_metrics.generation.keypoint_metrics import KEYPOINT_METRICS
from .rag_metrics.retrieval.words_precision import Words_Precision
from .rag_metrics.retrieval.words_recall import Words_Recall

METRICS_REGISTRY = {
    "rouge-l": ROUGELScore,
    "words_precision": Words_Precision,
    "words_recall": Words_Recall,
    "sentences_precision": EIR_Precision,
    "sentences_recall": EIR_Recall,
    "keypoint_metrics": KEYPOINT_METRICS,
}


def get_metric(metric_name):
    return METRICS_REGISTRY[metric_name]
