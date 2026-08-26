from src.evaluators.NeuMF_evaluator import NeuMFEvaluator
from src.sampler.lastfm_sampler import LastFMSampler


def main() -> None:
    sampler = LastFMSampler()
    print("======== Creating LastFM sample ========")
    
    statistics = sampler.create_sample()
    
    print(f"Source: {sampler.source_dir.resolve()}")
    print(f"Output: {sampler.output_dir.resolve()}")
    print(f"Users: {statistics['users']}")
    print(f"Items: {statistics['items']}")
    print(f"Interactions: {statistics['interactions']}")

    evaluator = NeuMFEvaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
