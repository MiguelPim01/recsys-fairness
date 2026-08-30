from src.splitters.cross_val_interface import ICrossValidationSplitter

# ----- Config
DATASET_NAME = "yelp"
MANIFEST_FILENAME = "yelp.cv_manifest.json"
# -----


class YelpCrossValidationSplitter(ICrossValidationSplitter):
    """Create reusable per-user folds and an isolated final test set."""

    DATASET_NAME = DATASET_NAME
    MANIFEST_FILENAME = MANIFEST_FILENAME
