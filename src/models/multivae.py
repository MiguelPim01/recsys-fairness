"""Rating-aware MultiVAE model used by the thesis experiments."""

import torch
import torch.nn.functional as F
from recbole.model.general_recommender.multivae import MultiVAE as RecBoleMultiVAE


class MultiVAE(RecBoleMultiVAE):
    """
    Adapt RecBole's MultiVAE to reconstruct ratings in the ``[1, 5]`` range.

    RecBole's implementation treats every observed interaction as a value of one and
    optimizes a multinomial reconstruction objective. The thesis datasets contain
    rating magnitudes, so this variant keeps those values in the user history and
    reconstructs the observed entries with mean squared error. The VAE architecture,
    reparameterization and KL annealing schedule remain unchanged.

    The model must be initialized with the training dataset. Passing the unsplit
    dataset would expose validation and test ratings through the input history.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.RATING = config["RATING_FIELD"]
        history_ids, history_values, _ = dataset.history_item_matrix(value_field=self.RATING)
        
        self.history_item_id = history_ids.to(self.device)
        self.history_item_value = history_values.to(self.device)

        # ``update`` controls KL annealing and must survive checkpoint restoration.
        self.other_parameter_name = ["update"]

    @staticmethod
    def to_rating(logits: torch.Tensor) -> torch.Tensor:
        """
        Map decoder logits continuously to the rating scale.
        """
        return 1.0 + 4.0 * torch.sigmoid(logits)

    def calculate_loss(self, interaction):
        """
        Return masked rating reconstruction loss plus annealed KL loss.
        """
        users = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(users)
        observed = rating_matrix > 0
        active_users = observed.any(dim=1)

        if not torch.any(active_users):
            raise RuntimeError("MultiVAE received a batch without training ratings")

        logits, mu, logvar = self.forward(rating_matrix)
        predictions = self.to_rating(logits)

        reconstruction_loss = F.mse_loss(
            predictions[observed], rating_matrix[observed]
        )

        active_mu = mu[active_users]
        active_logvar = logvar[active_users]
        kl_loss = -0.5 * torch.mean(
            torch.sum(
                1
                + active_logvar
                - active_mu.pow(2)
                - active_logvar.exp(),
                dim=1,
            )
        )

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        return reconstruction_loss + anneal * kl_loss

    def predict(self, interaction):
        """
        Predict ratings for the provided user-item pairs.
        """
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        
        rating_matrix = self.get_rating_matrix(users)
        logits, _, _ = self.forward(rating_matrix)
        predictions = self.to_rating(logits)
        
        rows = torch.arange(items.shape[0], device=predictions.device)
        
        return predictions[rows, items]

    def full_sort_predict(self, interaction):
        """
        Predict ratings for every item and return RecBole's flattened layout.
        """
        users = interaction[self.USER_ID]
        
        rating_matrix = self.get_rating_matrix(users)
        logits, _, _ = self.forward(rating_matrix)
        
        return self.to_rating(logits).reshape(-1)
