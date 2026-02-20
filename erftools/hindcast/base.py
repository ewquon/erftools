"""Abstract base class for ERA5/GFS hindcast preprocessing workflows."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from erftools.utils.projection import create_lcc_mapping

from .config import HindcastConfig

logger = logging.getLogger(__name__)


class HindcastBase(ABC):
    """Abstract base class for hindcast preprocessing workflows.

    Subclasses must implement :meth:`download` and :meth:`process`.
    The :meth:`run` method provides a template that calls both in order.

    Parameters
    ----------
    config_file:
        Path to the key:value input file (e.g. ``input_for_Laura``).
    **kwargs:
        Additional keyword arguments passed to subclasses.
    """

    def __init__(self, config_file: str, **kwargs: Any) -> None:
        self.config_file: str = config_file
        self.config: HindcastConfig = HindcastConfig.from_file(config_file)
        self.config.validate()
        self.lambert_conformal: str = create_lcc_mapping(self.config.area)
        logger.info("Loaded config from %s", config_file)

    def _setup_output_dirs(self, *dirs: str) -> None:
        """Create one or more output directories."""
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @abstractmethod
    def download(self) -> Any:
        """Download raw hindcast data.

        Returns
        -------
        Any
            Implementation-specific download result passed to :meth:`process`.
        """

    @abstractmethod
    def process(self, download_result: Any) -> None:
        """Process downloaded data and write ERF input files.

        Parameters
        ----------
        download_result:
            The value returned by :meth:`download`.
        """

    def run(self) -> None:
        """Template method: download then process."""
        logger.info("Starting %s run", self.__class__.__name__)
        result = self.download()
        self.process(result)
        logger.info("Finished %s run", self.__class__.__name__)
