from src.evaluators.model_evaluator_interface import IModelEvaluator

# ----- Config
MODEL_NAME = "NeuMF"

HYPERPARAMETER_LABELS = {
    "learning_rate": "lr",
    "dropout_prob": "dropout",
    "mf_embedding_size": "mf_emb",
    "mlp_embedding_size": "mlp_emb",
}
# -----

class NeuMFEvaluator(IModelEvaluator):
    """Train, cross-validate and tune RecBole's NeuMF."""

    MODEL_NAME = MODEL_NAME
    HYPERPARAMETER_LABELS = HYPERPARAMETER_LABELS
