#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import DatasetDict
from quilt3 import Package, list_package_versions
from speakerbox import eval_model, train

from .. import _constants as constants
from .._types import PathLike

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


def pull(
    trained_model_package_name: str = constants.TRAINED_MODEL_PACKAGE_NAME,
    s3_bucket_uri: str = constants.S3_BUCKET,
    package_dir_with_model: str = constants.TRAINED_MODEL_NAME,
    top_hash: Optional[str] = None,
    dest: PathLike = "./",
) -> None:
    """
    Pull down a single model.

    Parameters
    ----------
    trained_model_package_name: str
        The quilt package name which stores the trained speakerbox model.
        Default: "speakerbox/model"
    s3_bucket_uri: str
        The S3 bucket URI which stores the trained model package.
        Default: "s3://evamaxfield-uw-equitensors-speakerbox"
    package_dir_with_model: str
        The quilt package directory which contains the model files.
        Default: "trained-speakerbox"
    top_hash: Optional[str]
        Specific model version to pull.
        Default: None (latest)
    dest: PathLike
        Location to store the model.
        Default: current directory
    """
    package = Package.browse(
        trained_model_package_name,
        s3_bucket_uri,
        top_hash=top_hash,
    )
    package[package_dir_with_model].fetch(dest)


def list_n(
    n: int = 10,
    trained_model_package_name: str = constants.TRAINED_MODEL_PACKAGE_NAME,
    s3_bucket_uri: str = constants.S3_BUCKET,
) -> None:
    """
    List all stored models.

    Parameters
    ----------
    n: int
        Number of models to check
        Default: 10
    trained_model_package_name: str
        The quilt package name which stores the trained speakerbox model.
        Default: "speakerbox/model"
    s3_bucket_uri: str
        The S3 bucket URI which stores the trained model package.
        Default: "s3://evamaxfield-uw-equitensors-speakerbox"
    """
    # Get package versions
    lines = []
    versions = list(list_package_versions(trained_model_package_name, s3_bucket_uri))
    checked = 0
    for _, version in versions[::-1]:
        p = Package.browse(
            trained_model_package_name,
            s3_bucket_uri,
            top_hash=version,
        )
        for line in p.manifest:
            message = line["message"]
            lines.append(f"hash: {version} -- message: '{message}'")
            break

        checked += 1
        if checked == n:
            break

    single_print = "\n".join(lines)
    log.info(f"Models:\n{single_print}")


def train_and_eval(
    dataset_dir: PathLike = constants.PREPARED_DATASET_DIR,
    model_name: str = constants.TRAINED_MODEL_NAME,
    trained_model_package_name: str = constants.TRAINED_MODEL_PACKAGE_NAME,
    s3_bucket_uri: str = constants.S3_BUCKET,
) -> str:
    """
    Train and evaluate a new speakerbox model.

    Parameters
    ----------
    dataset_dir: PathLike
        Directory name for where the prepared dataset is stored.
        Default: prepared-speakerbox-dataset/
    model: str
        Name for the trained model.
        Default: trained-speakerbox
    trained_model_package_name: str
        The quilt package name which stores the trained speakerbox model.
        Default: "speakerbox/model"
    s3_bucket_uri: str
        The S3 bucket URI which stores the trained model package.
        Default: "s3://evamaxfield-uw-equitensors-speakerbox"

    Returns
    -------
    top_hash: str
        The generated package top hash. Includes both the model and eval results.
    """
    # Record training start time
    training_start_dt = datetime.utcnow().replace(microsecond=0).isoformat()

    # Load dataset
    dataset = DatasetDict.load_from_disk(dataset_dir)

    # Train
    model_storage_path = train(dataset, model_name=model_name)

    # Create reusable model storage function
    def store_model_dir(message: str) -> str:
        package = Package()
        package.set_dir(model_name, model_name)

        # Log contents
        dir_contents = list(Path(model_name).glob("*"))
        log.info(f"Uploading directory contents: {dir_contents}")

        # Upload
        pushed = package.push(
            trained_model_package_name,
            s3_bucket_uri,
            message=message,
            force=True,
        )
        return pushed.top_hash

    # Remove checkpoints and runs subdirs
    shutil.rmtree(model_storage_path / "runs")
    for checkpoint_dir in model_storage_path.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)

    # Store model to S3
    top_hash = store_model_dir(
        message=f"{training_start_dt} -- initial storage before eval",
    )
    log.info(f"Completed initial storage of model. Result hash: {top_hash}")

    # Eval
    accuracy, precision, recall, loss = eval_model(
        dataset["valid"],
        model_name=model_name,
    )
    eval_results_str = (
        f"eval acc: {accuracy:.5f}, pre: {precision:.5f}, "
        f"rec: {recall:.5f}, loss: {loss:.5f}"
    )
    log.info(eval_results_str)

    # Store eval results too
    top_hash = store_model_dir(message=f"{training_start_dt} -- {eval_results_str}")
    log.info(f"Completed storage of model eval results. Result hash: {top_hash}")
    return top_hash
