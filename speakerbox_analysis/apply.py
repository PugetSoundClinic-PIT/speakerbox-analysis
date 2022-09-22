#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

import pandas as pd
from cdp_backend.annotation.speaker_labels import annotate
from cdp_data import datasets, instances
from tqdm import tqdm

from . import _constants as constants
from ._types import (
    PathLike,
    _TranscriptApplicationError,
    _TranscriptApplicationReturn,
    _TranscriptMeta,
)
from .model import pull_model

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


def _pull_or_use_model(model_top_hash: str, model_storage_path: str) -> None:
    # Pull model
    if Path(model_storage_path).exists():
        log.info(f"Using existing model found in directory: '{model_storage_path}'")
    else:
        log.info(
            f"Pulling and using model from hash: '{model_top_hash}' "
            f"(storing to: '{model_storage_path}')"
        )
        pull_model(
            top_hash=model_top_hash,
            dest=model_storage_path,
        )


def apply_model_to_single_transcript(
    transcript: PathLike,
    audio: PathLike,
    dest: Optional[PathLike] = None,
    model_top_hash: str = (
        "453d51cc7006d2ba26640ba91eed67a5f8a9315d7c25d95f81072edb20054054"
    ),
    model_storage_path: str = "trained-speakerbox",
    transcript_meta: Optional[_TranscriptMeta] = None,
    remote_storage_dir: Optional[str] = None,
    fs_kwargs: Dict[str, Any] = {},
) -> Union[Path, _TranscriptApplicationReturn, _TranscriptApplicationError]:
    """
    Apply a trained Speakerbox model to a single transcript.

    Parameters
    ----------
    transcript: PathLike
        The path to the transcript file to annotate.
    audio: PathLike
        The path to the audio file to use for classification.
    dest: Optional[PathLike]
        Optional local storage destination
    model_top_hash: str
        The model top hash to pull from remote store and use for annotation.
        Default: 453d51... (highest accuracy model for Seattle to date)
    transcript_meta: Optional[_TranscriptMeta]
        Optional metadata to hand back during return. Used in parallel application.
    remote_storage_dir: Optional[str]
        An optional remote storage dir to store the annotated transcripts to.
        Should be in the form of '{bucket}/{dir}'. The file will be stored with a
        random uuid.
    fs_kwargs: Dict[str, Any]
        Extra arguments to pass to the created file system connection.

    Returns
    -------
    Union[Path, _TranscriptApplicationReturn, _TranscriptApplicationError]
        If transcript_meta was not provided and no errors arose during application,
        only the Path is returned.

        If transcript_meta was provided and no errors arose during application,
        a _TranscriptApplicationReturn is returned that passed back the annotated
        path and the metadata.

        If any error occurs during application, a _TranscriptApplicationError is
        returned.
    """
    # Pull or use model
    _pull_or_use_model(
        model_top_hash=model_top_hash,
        model_storage_path=model_storage_path,
    )

    # Configure destination file
    transcript = Path(transcript)
    if dest is None:
        transcript_name_no_suffix = transcript.with_suffix("").name
        dest_name = f"{transcript_name_no_suffix}-annotated.json"
        dest = transcript.parent / dest_name

    # Dest should always be a path
    dest = Path(dest)

    # Annotate and store
    try:
        annotated_transcript = annotate(
            transcript=transcript,
            audio=audio,
            model=model_storage_path,
        )
    except Exception as e:
        return _TranscriptApplicationError(
            transcript=str(transcript),
            error=str(e),
        )

    # Store and return
    with open(dest, "w") as open_f:
        open_f.write(annotated_transcript.to_json(indent=4))

    # Optionally store to S3
    if remote_storage_dir:
        import s3fs

        fs = s3fs.S3FileSystem(**fs_kwargs)

        # Make remote path
        remote_path = f"{remote_storage_dir}/{uuid4()}.json"
        fs.put_file(str(dest), remote_path)
        log.info(f"Stored '{dest}' to '{remote_path}'")

        # Attach remote path to meta
        if transcript_meta is not None:
            transcript_meta.stored_annotated_transcript_uri = f"s3://{remote_path}"

    # Return simple path (likely single application)
    if transcript_meta is None:
        return dest

    # Return application return (likely batch / parallel apply)
    return _TranscriptApplicationReturn(
        annotated_transcript_path=str(dest),
        transcript_meta=transcript_meta,
    )


def apply_model_across_cdp_dataset(
    instance: str = instances.CDPInstances.Seattle,
    start_datetime: Optional[Union[str, datetime]] = None,
    end_datetime: Optional[Union[str, datetime]] = None,
    model_top_hash: str = constants.InstanceModelHashes.Seattle,
    model_storage_path: str = constants.TRAINED_MODEL_NAME,
    remote_storage_dir: Optional[str] = None,
    fs_kwargs: Dict[str, Any] = {},
) -> str:
    """
    Apply a trained Speakerbox model across a large CDP session dataset.

    Parameters
    ----------
    instance: str
        The CDP instance infrastructure slug.
        Default: cdp_data.CDPInstances.Seattle
    start_datetime: Optional[Union[str, datetime]]
        The start datetime in ISO format (i.e. "2021-01-01")
        Default: None (earliest data available from instance)
    end_datetime: Optional[Union[str, datetime]]
        The end datetime in ISO format (i.e. "2021-02-01")
        Default: None (latest data available from instance)
    model_top_hash: str
        The model top hash to pull from remote store and use for annotation.
        Default: 453d51... (highest accuracy model for Seattle to date)
    remote_storage_dir: Optional[str]
        An optional remote storage dir to store the annotated transcripts to.
        Should be in the form of '{bucket}/{dir}'. A directory with the datetime
        of when this function was called will be appended to the path as well.
        For example, when provided the following: 'my-bucket/application-results/'
        the annotated files will ultimately be placed in the directory:
        'my-bucket/application-results/2022-06-29T11:14:42/'.
        Each file will be given a random uuid.
    fs_kwargs: Dict[str, Any]
        Extra arguments to pass to the created file system connection.

    Returns
    -------
    str
        The path to the results parquet file.

    Notes
    -----
    When attempting to use remote storage, be sure to set your `AWS_PROFILE`
    environment variable.
    """
    if remote_storage_dir:
        # Clean up storage dir tail
        if remote_storage_dir[-1] == "/":
            remote_storage_dir = remote_storage_dir[:-1]

        # Store in directory with datetime of run
        dt = datetime.utcnow().replace(microsecond=0).isoformat()
        remote_storage_dir = f"{remote_storage_dir}/{dt}"
        log.info(f"Will store annotated transcripts to: '{remote_storage_dir}'")

    # Get session dataset to apply against
    ds = datasets.get_session_dataset(
        infrastructure_slug=getattr(instances.CDPInstances, instance),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        store_transcript=True,
        store_audio=True,
    )

    # Pull or use model
    # Do this now to avoid parallel download problems
    _pull_or_use_model(
        model_top_hash=model_top_hash,
        model_storage_path=model_storage_path,
    )

    log.info("Annotating transcripts...")
    annotation_returns = []
    for _, row in tqdm(ds.iterrows(), desc="Transcripts annotated"):
        annotation_returns.append(
            apply_model_to_single_transcript(
                transcript=row.transcript_path,
                audio=row.audio_path,
                dest=None,
                model_top_hash=model_top_hash,
                model_storage_path=model_storage_path,
                transcript_meta=_TranscriptMeta(
                    event_id=row.event.id,
                    session_id=row.id,
                    session_datetime=row.session_datetime,
                ),
                remote_storage_dir=remote_storage_dir,
                fs_kwargs=fs_kwargs,
            )
        )

    # Filter any errors
    errors = pd.DataFrame(
        [
            e.to_dict()
            for e in annotation_returns
            if isinstance(e, _TranscriptApplicationError)
        ]
    )
    results = pd.DataFrame(
        [
            {
                "annotated_transcript_path": r.annotated_transcript_path,
                **r.transcript_meta.to_dict(),
            }
            for r in annotation_returns
            if isinstance(r, _TranscriptApplicationReturn)
        ]
    )

    # Log info
    log.info(f"Annotated {len(results)} transcripts; {len(errors)} errored")

    # Store errors to CSV for easy viewing
    # Store results to parquet for fast load
    errors.to_csv("errors.csv", index=False)
    results_save_path = f"results--start_{start_datetime}--end_{end_datetime}.parquet"
    results.to_parquet(results_save_path)

    # Store errors and results to remote storage
    if remote_storage_dir:
        import s3fs

        fs = s3fs.S3FileSystem(**fs_kwargs)

        # Shove errors and results to remote
        fs.put_file("errors.csv", f"{remote_storage_dir}/errors.csv")
        fs.put_file(results_save_path, f"{remote_storage_dir}/{results_save_path}")

    # Return path to parquet file of results
    return results_save_path
