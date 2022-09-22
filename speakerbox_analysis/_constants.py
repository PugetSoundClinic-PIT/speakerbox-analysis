#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

TRAINING_DATA_PACKAGE_NAME = "speakerbox/training-data"
TRAINED_MODEL_PACKAGE_NAME = "speakerbox/model"
S3_BUCKET = "s3://evamaxfield-uw-equitensors-speakerbox"
TRAINING_DATA_DIR = Path(__file__).parent / "training-data"
TRAINING_DATA_DIRS_FOR_UPLOAD = [TRAINING_DATA_DIR / "diarized"]
PREPARED_DATASET_DIR = Path(__file__).parent / "prepared-speakerbox-dataset"
TRAINED_MODEL_NAME = "trained-speakerbox"


class InstanceModelHashes:
    """
    Hashes for highest accuracy models for each CDP Instance.
    """

    Seattle = "453d51cc7006d2ba26640ba91eed67a5f8a9315d7c25d95f81072edb20054054"
