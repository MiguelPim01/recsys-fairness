.PHONY: run_experiments analyze_experiment clean

USER_LIMIT ?= 1000
ITEM_LIMIT ?= 1000
FOLD_WORKERS ?= 1
USE_RESTAURANTS_USERS_ONLY ?= false
SEED ?= 42
EXPERIMENT ?=

PYTHON_COMMAND = $(if $(wildcard .venv/bin/python),.venv/bin/python,uv run python)

YELP_TRANSFORM_ARGUMENTS = $(if $(filter true,$(USE_RESTAURANTS_USERS_ONLY)),--use-restaurants-users-only)

run_experiments:
	@set -eu; \
	experiment="$$($(PYTHON_COMMAND) -m src.utils.experiments create --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT) --seed $(SEED))"; \
	trap '$(PYTHON_COMMAND) -m src.utils.experiments status --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT) --experiment "'"$$experiment"'" --status failed' 0; \
	echo "Starting experiment $$experiment for $(USER_LIMIT)_$(ITEM_LIMIT)"; \
	./scripts/transform_datasets.sh all $(YELP_TRANSFORM_ARGUMENTS); \
	./scripts/sample_datasets.sh all --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT) --experiment "$$experiment"; \
	$(PYTHON_COMMAND) -m src.utils.experiments prepare --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT) --experiment "$$experiment" --dataset all --folds 5; \
	./scripts/evaluate_models.sh --model all --dataset all --experiment "$$experiment" --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT) --cross-validation --hyperparameter-search --folds 5 --fold-workers $(FOLD_WORKERS); \
	./scripts/analyze_fairness.sh --model all --dataset all --experiment "$$experiment" --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT); \
	$(PYTHON_COMMAND) -m src.utils.experiments status --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT) --experiment "$$experiment" --status complete; \
	trap - 0; \
	echo "Completed experiment $$experiment"

analyze_experiment:
	@test -n "$(EXPERIMENT)" || { echo "EXPERIMENT=seed_k is required" >&2; exit 2; }
	@./scripts/analyze_fairness.sh --model all --dataset all --experiment "$(EXPERIMENT)" --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT)

clean:
	@echo "Cleaning processed data; versioned samples are preserved..."
	rm -rf data/processed/*
