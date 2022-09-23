#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

TRAINING_DATA_PACKAGE_NAME = "speakerbox/training-data"
TRAINED_MODEL_PACKAGE_NAME = "speakerbox/model"
S3_BUCKET = "s3://evamaxfield-uw-equitensors-speakerbox"
TRAINING_DATA_DIR = Path("training-data")
TRAINING_DATA_DIRS_FOR_UPLOAD = [TRAINING_DATA_DIR / "diarized"]
PREPARED_DATASET_DIR = "prepared-speakerbox-dataset"
TRAINED_MODEL_NAME = "trained-speakerbox"
ANNOTATED_DATA_DIR = Path("annotated-dataset")


class InstanceModelHashes:
    """
    Hashes for highest accuracy models for each CDP Instance.
    """

    Seattle = "15148ff0e8cb56a455ce6b41f4bb744c966baa9b1d99c0925b25cf200aa7b9a6"
