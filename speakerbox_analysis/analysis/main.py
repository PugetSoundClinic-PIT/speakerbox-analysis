#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import dask.dataframe as dd
import pandas as pd
from cdp_backend.pipeline.transcript_model import Sentence, Transcript
from dataclasses_json import DataClassJsonMixin

from .._types import PathLike

###############################################################################


@dataclass
class SpeakingTimeProgress(DataClassJsonMixin):
    time: float
    sum_: float
    mean_: float
    median_: float
    max_: float


@dataclass
class SpeakingTime(DataClassJsonMixin):
    name: Optional[str]
    sum_: float
    mean_: float
    median_: float
    max_: float
    timeseries_as_seconds: List[SpeakingTimeProgress]
    timeseries_as_progress: List[SpeakingTimeProgress]


def _speaking_times_single(transcript_path: Path) -> pd.DataFrame:
    # For a single transcript get list of speaker names
    # for each speaker create stats
    # Also get sum of all speakers

    # Open transcript
    with open(transcript_path, "r") as open_f:
        transcript = Transcript.from_json(open_f.read())

    # Agg sentences
    speaker_sentences: Dict[Optional[str], List[Sentence]] = {}
    for sentence in transcript.sentences:
        speaker_sentences[sentence.speaker_name] = sentence

    # Calculations
    for speaker, sentences in speaker_sentences.items():
        pass


###############################################################################


def run(results_dir: PathLike, analysis_results_dir: PathLike = "weird") -> Path:
    # Read all result files
    results_dir = Path(results_dir)
    results = (
        dd.read_parquet(f"{results_dir}/*.parquet").compute().reset_index(drop=True)
    )

    # Create new column with shortened transcript paths
    results["local_transcript_path"] = results["stored_annotated_transcript_uri"].apply(
        lambda uri: results_dir / uri.rsplit("/")[-1]
    )

    # Calc speaking time of different speakers

    print(results)
    return results
