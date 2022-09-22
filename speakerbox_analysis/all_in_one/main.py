#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Optional

from .. import _constants as constants
from .._types import PathLike
from ..data import prepare_for_model_training
from ..model import train_and_eval

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


def prepare_dataset_and_train_and_eval_model(
    prepared_dataset_storage_dir: PathLike = constants.PREPARED_DATASET_DIR,
    top_hash: Optional[str] = None,
    equalize: bool = False,
    model_name: str = constants.TRAINED_MODEL_NAME,
) -> None:
    """
    Runs prepare_dataset and train_and_eval one after the other.

    Parameters are passed down to the appropriate functions.

    See Also
    --------
    speakerbox_analysis.data.prepare_for_model_training
        The function to prepare the dataset to be ready for training.
    speakerbox_analysis.model.train_and_eval
        The function to train and evaluate a model.
    """
    prepare_for_model_training(
        prepared_dataset_storage_dir=prepared_dataset_storage_dir,
        top_hash=top_hash,
        equalize=equalize,
    )

    train_and_eval(
        dataset_dir=prepared_dataset_storage_dir,
        model_name=model_name,
    )
