import logging
import sys
from pathlib import Path
from typing import Any

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed

# ----- Variables
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# -----

# ----- Config
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
# -----

class NeuMFEvaluator:
    """Train and evaluate RecBole's NeuMF on an existing atomic dataset."""

    def __init__(self, dataset_dir: Path | str = REPOSITORY_ROOT / "data/sample/lastfm", config_path: Path | str = REPOSITORY_ROOT / "config/models/NeuMF.yaml"):
        self.dataset_dir = Path(dataset_dir)
        self.config_path = Path(config_path)

    def evaluate(self) -> dict[str, Any]:
        config = Config(
            model="NeuMF",
            dataset=self.dataset_dir.name,
            config_file_list=[str(self.config_path)],
            config_dict={"data_path": str(self.dataset_dir.parent.resolve())},
        )
        
        init_seed(config["seed"], config["reproducibility"])

        print("\n======== Loading dataset ========")
        dataset = create_dataset(config)
        print(dataset)

        print("\n======== Train, val and test split ========")
        train_data, valid_data, test_data = data_preparation(config, dataset)

        init_seed(config["seed"], config["reproducibility"])
        
        model = get_model(config["model"])(config, train_data.dataset).to(
            config["device"]
        )
        print("\n======== Model ========")
        print(model)

        trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
        trainer = trainer_class(config, model)
        
        print("\n======== Training ========")
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=False,
            show_progress=config["show_progress"],
        )

        print("\n======== Test evaluation ========")
        test_result = trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=config["show_progress"],
        )

        results = {
            "best_valid_score": best_valid_score,
            "best_valid_result": best_valid_result,
            "test_result": test_result,
        }
        
        print("\n======== Results ========")
        print(f"Best validation score: {best_valid_score}")
        print(f"Validation metrics: {best_valid_result}")
        print(f"Test metrics: {test_result}")
        
        return results

    @staticmethod
    def _configure_console_logging():
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s - %(levelname)s] %(message)s",
            stream=sys.stdout,
            force=True,
        )
