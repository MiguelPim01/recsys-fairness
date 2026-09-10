.PHONY: run_experiments clean

USER_LIMIT ?= 1000
ITEM_LIMIT ?= 1000
FOLD_WORKERS ?= 1
USE_RESTAURANTS_USERS_ONLY ?= false

YELP_TRANSFORM_ARGUMENTS = $(if $(filter true,$(USE_RESTAURANTS_USERS_ONLY)),--use-restaurants-users-only)

run_experiments:
	@./scripts/transform_datasets.sh all $(YELP_TRANSFORM_ARGUMENTS)

	@./scripts/sample_datasets.sh all --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT)

	@./scripts/evaluate_models.sh --model all --dataset all --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT) --cross-validation --hyperparameter-search --folds 5 --fold-workers $(FOLD_WORKERS)

clean:
	@echo "Cleaning processed and sampled data..."
	rm -rf data/processed/* data/sample/*
