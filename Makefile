.PHONY: run_experiments clean

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
	./scripts/transform_datasets.sh all

	$(call default_heading,Sampling datasets)
	./scripts/sample_datasets.sh all
	
	$(call yellow_heading,Running experiments)
	$(call yellow_command,./scripts/evaluate_models.sh --model all --dataset all --cross-validation --hyperparameter-search --folds 5)

clean:
	@echo "Cleaning processed and sampled data..."
	rm -rf data/processed/* data/sample/*
