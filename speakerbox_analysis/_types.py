#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from dataclasses_json import DataClassJsonMixin

###############################################################################

PathLike = Union[str, Path]

###############################################################################


@dataclass
class _TranscriptMeta(DataClassJsonMixin):
    event_id: str
    session_id: str
    session_datetime: datetime
    stored_annotated_transcript_uri: Optional[str] = None


@dataclass
class _TranscriptApplicationReturn(DataClassJsonMixin):
    annotated_transcript_path: str
    transcript_meta: _TranscriptMeta


@dataclass
class _TranscriptApplicationError(DataClassJsonMixin):
    transcript: str
    error: str
