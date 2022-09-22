#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Iterable, Optional

import git
import pandas as pd
from quilt3 import Package
from speakerbox import preprocess
from speakerbox.datasets import seattle_2021_proto

from .. import _constants as constants
from .._types import PathLike

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


def upload_for_model_training(
    training_data_dirs_for_upload: Iterable[
        PathLike
    ] = constants.TRAINING_DATA_DIRS_FOR_UPLOAD,
    package_name: str = constants.TRAINING_DATA_PACKAGE_NAME,
    s3_bucket_uri: str = constants.S3_BUCKET,
    dry_run: bool = False,
    force: bool = False,
) -> str:
    """
    Upload data required for training a new model to S3.

    Parameters
    ----------
    training_data_dirs_for_upload: List[PathLike]
        List of directory paths containing data required for training.
        Default: ["./training-data/diarized/]
    package_name: str
        Name to give to the training data quilt package.
        Default: "speakerbox/training-data"
    s3_bucket_uri: str
        URI of the S3 bucket to push to.
        Default: "s3://evamaxfield-uw-equitensors-speakerbox"
    dry_run: bool
        Conduct dry run of the package generation. Will create a JSON manifest file
        of that package instead of uploading.
        Default: False (commit push)
    force: bool
        Should the current repo status be ignored and allow a dirty git tree.
        Default: False (do not allow dirty git tree)

    Returns
    -------
    top_hash: str
        The generated package top hash.

    Raises
    ------
    ValueError
        Git tree is dirty and force was not specified.
    """
    # Report with directory will be used for upload
    log.info(f"Using contents of directories: {training_data_dirs_for_upload}")

    # Create quilt package
    package = Package()
    for training_data_dir in training_data_dirs_for_upload:
        training_data_dir_p = Path(training_data_dir)
        package.set_dir(training_data_dir_p.name, training_data_dir_p)

    # Report package contents
    log.info(f"Package contents: {package}")

    # Check for dry run
    if dry_run:
        # Attempt to build the package
        top_hash = package.build(package_name)

        # Get resolved save path
        manifest_save_path = Path("upload-manifest.jsonl").resolve()
        with open(manifest_save_path, "w") as manifest_write:
            package.dump(manifest_write)

        # Report where manifest was saved
        log.info(f"Dry run generated manifest stored to: {manifest_save_path}")
        log.info(f"Completed package dry run. Result hash: {top_hash}")
        return top_hash

    # Check repo status
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty():
        if not force:
            raise ValueError(
                "Repo has uncommitted files and force was not specified. "
                "Commit your files before continuing."
            )
        else:
            log.warning(
                "Repo has uncommitted files but force was specified. "
                "I hope you know what you're doing..."
            )

    # Get current git commit
    commit = repo.head.object.hexsha

    # Upload
    pushed = package.push(
        package_name,
        s3_bucket_uri,
        message=f"From commit: {commit}",
    )
    log.info(f"Completed package push. Result hash: {pushed.top_hash}")
    return pushed.top_hash


def prepare_for_model_training(
    training_data_dir: PathLike = constants.TRAINING_DATA_DIR,
    uploaded_data_package_name: str = constants.TRAINING_DATA_PACKAGE_NAME,
    s3_bucket_uri: str = constants.S3_BUCKET,
    prepared_dataset_storage_dir: PathLike = constants.PREPARED_DATASET_DIR,
    top_hash: Optional[str] = None,
    equalize: bool = False,
) -> Path:
    """
    Pull and prepare the dataset for training a new model.

    Parameters
    ----------
    training_data_dir: PathLike
        The directory in which all training data will be placed.
        Default: "./training-data"
    uploaded_data_package_name: str
        The quilt package name for any preprocessed data to pull down
        (usually already labeled diarization chunks).
        Default: "speakerbox/training-data"
    s3_bucket_uri: str
        The S3 bucket URI to pull data any preprocessed data from.
        Default: "s3://evamaxfield-uw-equitensors-speakerbox"
    prepared_dataset_storage_dir: PathLike
        Directory name for where the prepared dataset should be stored.
        Default: prepared-speakerbox-dataset/
    top_hash: Optional[str]
        A specific version of the S3 stored data to retrieve.
        Default: None (use latest)
    equalize: bool
        Should the prepared dataset be equalized by label counts.
        Default: False (do not equalize)

    Returns
    -------
    dataset_path: Path
        Path to the prepared and serialized dataset.
    """
    log.info("Pulling data to prepare for model training...")
    # Setup storage dir
    training_data_storage_dir = Path(training_data_dir).resolve()
    training_data_storage_dir.mkdir(exist_ok=True)

    # Pull / prep original Seattle data
    log.info("Unpacking the Seattle 2021 prototype annotation dataset...")
    seattle_2021_proto_dir = training_data_storage_dir / "seattle-2021-proto"
    seattle_2021_proto_dir = seattle_2021_proto.unpack(
        dest=seattle_2021_proto_dir,
        clean=True,
    )
    seattle_2021_ds_items = seattle_2021_proto.pull_all_files(
        annotations_dir=seattle_2021_proto_dir / "annotations",
        transcript_output_dir=seattle_2021_proto_dir / "unlabeled_transcripts",
        audio_output_dir=seattle_2021_proto_dir / "audio",
    )

    # Expand annotated gecko data
    seattle_2021_ds = preprocess.expand_gecko_annotations_to_dataset(
        seattle_2021_ds_items,
        audio_output_dir=training_data_storage_dir / "chunked-audio-from-gecko",
        overwrite=True,
    )

    # Pull diarized data
    package = Package.browse(
        uploaded_data_package_name,
        s3_bucket_uri,
        top_hash=top_hash,
    )

    # Download
    log.info("Pulling pre-diarized and labelled audio...")
    package.fetch(training_data_storage_dir)

    # Expand diarized data
    diarized_ds = preprocess.expand_labeled_diarized_audio_dir_to_dataset(
        [
            "training-data/diarized/01e7f8bb1c03/",
            "training-data/diarized/2cdf68ae3c2c/",
            "training-data/diarized/6d6702d7b820/",
            "training-data/diarized/9f55f22d8e61/",
            "training-data/diarized/9f581faa5ece/",
        ],
        audio_output_dir=training_data_storage_dir / "chunked-audio-from-diarized",
        overwrite=True,
    )

    # Combine into single
    combined_ds = pd.concat([seattle_2021_ds, diarized_ds], ignore_index=True)

    # Generate train test validate splits
    dataset, _ = preprocess.prepare_dataset(
        combined_ds,
        equalize_data_within_splits=equalize,
    )

    # Store to disk
    dataset.save_to_disk(prepared_dataset_storage_dir)
    return Path(prepared_dataset_storage_dir)
