import evaluate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AutomaticEvaluator:
    def __init__(self, lang: str = "es"):
        self.lang = lang
        self.cosine_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.bertscore = evaluate.load("bertscore")

    def compute_bleu(self, pred: str, ref: str) -> dict:
        return self.bleu.compute(predictions=[pred], references=[[ref]])

    def compute_rouge(self, pred: str, ref: str) -> dict:
        return self.rouge.compute(predictions=[pred], references=[ref])

    def compute_bertscore(self, pred: str, ref: str) -> dict:
        result = self.bertscore.compute(predictions=[pred], references=[ref], lang=self.lang)
        return {
            "precision": result["precision"][0],
            "recall": result["recall"][0],
            "f1": result["f1"][0]
        }

    def compute_cosine(self, pred: str, ref: str) -> float:
        emb = self.cosine_model.encode([pred, ref])
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    def compute_all(self, pred: str, ref: str) -> dict:
        return {
            "bleu": self.compute_bleu(pred, ref),
            "rouge": self.compute_rouge(pred, ref),
            "bertscore": self.compute_bertscore(pred, ref),
            "cosine": self.compute_cosine(pred, ref)
        }