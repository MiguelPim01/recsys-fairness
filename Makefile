.PHONY: run_experiments clean

run_experiments:
	@echo "Transforming datasets"
	./scripts/transform_datasets.sh all

	@echo "Sampling datasets"
	./scripts/sample_datasets.sh all
	
	@echo "Running experiments"
	./scripts/evaluate_models.sh --model all --dataset all --cross-validation --hyperparameter-search --folds 5

clean:
	@echo "Cleaning processed and sampled data..."
	rm -rf data/processed/* data/sample/*
