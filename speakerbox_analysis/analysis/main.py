#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import dask.dataframe as dd
import numpy as np
import pandas as pd
from cdp_backend.pipeline.transcript_model import Sentence, Transcript
from dataclasses_json import DataClassJsonMixin

from .._types import PathLike

###############################################################################


@dataclass
class SpeakingTimeProgress(DataClassJsonMixin):
    progress: int
    sum_: float
    mean_: float
    median_: float
    max_: float
    std_: float
    count_: int


@dataclass
class SpeakingTime(DataClassJsonMixin):
    name: Optional[str]
    sum_: float
    mean_: float
    median_: float
    max_: float
    std_: float
    count_: int


@dataclass
class SpeakingTimesReturn(DataClassJsonMixin):
    stats: pd.DataFrame
    sparse_timeseries: pd.DataFrame
    filled_timeseries: pd.DataFrame


def _speaking_times_single(transcript_path: Path) -> SpeakingTimesReturn:
    # For a single transcript get list of speaker names
    # for each speaker create stats
    # Also get sum of all speakers

    # Open transcript
    with open(transcript_path, "r") as open_f:
        transcript = Transcript.from_json(open_f.read())

    # Calc total meeting duration
    meeting_duration = (
        transcript.sentences[-1].end_time - transcript.sentences[0].start_time
    )

    # Agg sentences
    speaker_sentences: Dict[Optional[str], List[Sentence]] = {}
    for sentence in transcript.sentences:
        # Create new speaker with sentence list or append
        if sentence.speaker_name not in speaker_sentences:
            speaker_sentences[sentence.speaker_name] = [sentence]
        else:
            speaker_sentences[sentence.speaker_name].append(sentence)

    # Construct speaking timeseries windows
    window_duration = meeting_duration / 100
    sentence_timeseries_windows: List[List[Sentence]] = []
    current_window_sentences = []
    current_window_start_time = 0.0
    current_window_end_time = current_window_start_time + window_duration
    for sentence in transcript.sentences:
        # Include the sentence in the current window if a majority of the
        # segment is in the window
        overlap = max(
            0,
            min(current_window_end_time, sentence.end_time)
            - max(current_window_start_time, sentence.start_time),
        )
        sentence_duration = sentence.end_time - sentence.start_time

        if overlap / sentence_duration >= 0.5:
            current_window_sentences.append(sentence)

        # If the overlap percent isn't high enough we know we need
        # to move to the next window
        else:
            sentence_timeseries_windows.append(current_window_sentences)
            current_window_sentences = [sentence]
            current_window_start_time += window_duration
            current_window_end_time = current_window_start_time + window_duration

    # Append the last group
    sentence_timeseries_windows.append(current_window_sentences)

    # Precompute speaker timeseries
    speaker_timeseries: Dict[Optional[str], List[SpeakingTimeProgress]] = {
        speaker: [] for speaker in speaker_sentences
    }
    for i, window_sentences in enumerate(sentence_timeseries_windows):
        # Create a speaker to sentence list LUT for this window
        window_speaker_sentences: Dict[Optional[str], List[Sentence]] = {}
        for sentence in window_sentences:
            # Create new speaker with sentence list or append
            if sentence.speaker_name not in window_speaker_sentences:
                window_speaker_sentences[sentence.speaker_name] = [sentence]
            else:
                window_speaker_sentences[sentence.speaker_name].append(sentence)

        # Get speaker duration time vector for this window
        for (
            speaker,
            single_speaker_window_sentences,
        ) in window_speaker_sentences.items():
            vec = np.array(
                [
                    (s.end_time - s.start_time)
                    for s in single_speaker_window_sentences
                    if s.speaker_name == speaker
                ]
            )

            # Calc stats if count is greater than 0
            count_ = len(vec)
            if count_ >= 0:
                speaker_timeseries[speaker].append(
                    SpeakingTimeProgress(
                        progress=i,
                        sum_=vec.sum(),
                        mean_=vec.mean(),
                        median_=np.median(vec),
                        max_=vec.max(),
                        std_=vec.std(),
                        count_=count_,
                    )
                )

    # Convert timeseries to DataFrames
    sparse_timeseries_frames = []
    filled_timeseries_frames = []
    for speaker, timeseries in speaker_timeseries.items():
        # Generate blank frame to fill in data
        filled_timeseries = pd.DataFrame(
            {
                "progress": range(100),
                "sum_": 0,
                "mean_": 0,
                "median_": 0,
                "max_": 0,
                "std_": 0,
                "count_": 0,
            }
        )

        # Read data into frame
        sparse_timeseries = pd.DataFrame([stp.to_dict() for stp in timeseries])

        # Fill in data to the full frame
        filled_timeseries.index = filled_timeseries.progress
        sparse_timeseries.index = sparse_timeseries.progress
        for attr in ["sum_", "mean_", "median_", "max_", "std_", "count_"]:
            filled_timeseries[attr] = sparse_timeseries[attr]

        # Fill the NaNs with 0
        filled_timeseries = filled_timeseries.fillna(0)

        # Add the speaker name column
        sparse_timeseries["speaker"] = str(speaker)
        filled_timeseries["speaker"] = str(speaker)

        # Append to all records
        sparse_timeseries_frames.append(sparse_timeseries)
        filled_timeseries_frames.append(filled_timeseries)

    # Calculations
    speaker_stats: List[SpeakingTime] = []
    for speaker, sentences in speaker_sentences.items():
        # Get speaker duration time vectors
        vec = np.array([(s.end_time - s.start_time) for s in sentences])
        speaker_stats.append(
            SpeakingTime(
                name=speaker,
                sum_=vec.sum(),
                mean_=vec.mean(),
                median_=np.median(vec),
                max_=vec.max(),
                std_=vec.std(),
                count_=len(vec),
            )
        )

    return SpeakingTimesReturn(
        stats=pd.DataFrame([stats.to_dict() for stats in speaker_stats]),
        sparse_timeseries=pd.concat(sparse_timeseries_frames, ignore_index=True),
        filled_timeseries=pd.concat(filled_timeseries_frames, ignore_index=True),
    )


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
