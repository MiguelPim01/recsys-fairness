from src.evaluators.model_evaluator_interface import IModelEvaluator
from src.models.multivae import MultiVAE

# ----- Config
MODEL_NAME = "MultiVAE"

HYPERPARAMETER_LABELS = {
    "learning_rate": "lr",
    "dropout_prob": "dropout",
    "mlp_hidden_size": "mlp",
    "latent_dimension": "latent",
    "anneal_cap": "anneal",
    "total_anneal_steps": "anneal_steps",
}
# -----


class MultiVAEEvaluator(IModelEvaluator):
    """Train, cross-validate and tune the rating-aware MultiVAE."""

    MODEL_NAME = MODEL_NAME
    MODEL_CLASS = MultiVAE
    HYPERPARAMETER_LABELS = HYPERPARAMETER_LABELS
