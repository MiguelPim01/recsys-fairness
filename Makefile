.PHONY: run_experiments clean

USER_LIMIT ?= 1000
ITEM_LIMIT ?= 1000
USE_RESTAURANTS_USERS_ONLY ?= false

YELP_TRANSFORM_ARGUMENTS = $(if $(filter true,$(USE_RESTAURANTS_USERS_ONLY)),--use-restaurants-users-only)

define default_heading
	@if [ -t 1 ] && [ -z "$${NO_COLOR+x}" ]; then printf '\033[1m%s\033[0m\n' "$(1)"; else printf '%s\n' "$(1)"; fi
endef

define yellow_heading
	@if [ -t 1 ] && [ -z "$${NO_COLOR+x}" ]; then printf '\033[1;93m%s\033[0m\n' "$(1)"; else printf '%s\n' "$(1)"; fi
endef

define yellow_command
	@if [ -t 1 ] && [ -z "$${NO_COLOR+x}" ]; then printf '\033[93m%s\033[0m\n' "$(1)"; else printf '%s\n' "$(1)"; fi
	@$(1)
endef

run_experiments:
	$(call default_heading,Transforming datasets)
	./scripts/transform_datasets.sh all $(YELP_TRANSFORM_ARGUMENTS)

	$(call default_heading,Sampling datasets)
	./scripts/sample_datasets.sh all --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT)
	
	$(call yellow_heading,Running experiments)
	$(call yellow_command,./scripts/evaluate_models.sh --model all --dataset all --user-limit $(USER_LIMIT) --item-limit $(ITEM_LIMIT) --cross-validation --hyperparameter-search --folds 5)

clean:
	@echo "Cleaning processed and sampled data..."
	rm -rf data/processed/* data/sample/*
