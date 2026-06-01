"""
Para utilizar os datasets com o RecBole corretamente, precisamos convertê-los para o formato atomic da biblioteca. 
Para o caso do Movie Lens 1M, precisamos de três arquivos resultantes com suas respectivas colunas:

1. ml-1m.inter
  a. user_id:token
  b. item_id:token
  c. rating:float
  d. timestamp:float
2. ml-1m.user
  a. user_id:token
  b. age:int
  c. gender:token
  d. occupation:token
  e. zip_code:token
3. ml-1m.item
  a. item_id:token
  b. movie_title:token_seq
  c. release_year:token
  d. genre:token_seq

The config file must have:
  data_path: ~/xxx/yyy/
  dataset: ml-1m
"""

from pathlib import Path
from datetime import datetime
from logging import getLogger

from recbole.config import Config
from torch.utils.tensorboard import SummaryWriter
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer


if __name__ == '__main__':
    config_file = str(Path(__file__).with_name('ml_1m.yaml'))

    config = Config(
        model='ItemKNN',
        dataset='ml-1m',
        config_file_list=[config_file]
    )

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = get_model(config['model'])(config, train_data._dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=False,
        show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config['show_progress']
    )

    print(test_result)
    
    run_name = datetime.now().strftime("ItemKNN_ml1m_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"log_tensorboard/custom_test_results/{run_name}")

    for metric_name, metric_value in test_result.items():
        writer.add_scalar(f"test/{metric_name}", metric_value, 0)

    writer.add_text("experiment/model", "ItemKNN", 0)
    writer.add_text("experiment/dataset", "MovieLens 1M manually converted to RecBole atomic format", 0)
    writer.add_text("experiment/config", str(config), 0)

    writer.close()