from src.sampler.cross_val_interface import ICrossValidationSplitter

# ----- Config
DATASET_NAME = "lastfm"
MANIFEST_FILENAME = "lastfm.cv_manifest.json"
# -----

class LastFMCrossValidationSplitter(ICrossValidationSplitter):
    """Create reusable per-user folds and an isolated final test set."""

    DATASET_NAME = DATASET_NAME
    MANIFEST_FILENAME = MANIFEST_FILENAME
