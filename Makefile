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
	@$(PYTHON_COMMAND) -m src.utils.campaign run \
		--seeds "$(SEED)" \
		--user-limit $(USER_LIMIT) \
		--item-limit $(ITEM_LIMIT) \
		--folds 5 \
		--fold-workers $(FOLD_WORKERS) \
		$(YELP_TRANSFORM_ARGUMENTS)

analyze_experiment:
	@test -n "$(EXPERIMENT)" || { echo "EXPERIMENT=seed_k is required" >&2; exit 2; }
	@./scripts/analyze_fairness.sh --model all --dataset all --experiment "$(EXPERIMENT)" --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT)

clean:
	@echo "Cleaning processed data; versioned samples are preserved..."
	rm -rf data/processed/*
