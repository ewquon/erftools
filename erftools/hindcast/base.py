"""Abstract base class for ERA5/GFS hindcast preprocessing workflows."""
from __future__ import annotations

import datetime as _dt
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union

from erftools.utils.projection import create_lcc_mapping

from .config import HindcastConfig

logger = logging.getLogger(__name__)


class HindcastBase(ABC):
    """Abstract base class for hindcast preprocessing workflows.

    Subclasses must implement :meth:`download` and :meth:`process`.
    The :meth:`run` method provides a template that calls both in order.

    The configuration can be supplied in one of two ways:

    1. **Legacy input file** – pass *config_file* as the first positional
       argument (a path to a ``key: value`` text file)::

           hindcast = ERA5Hindcast("input_for_Laura")

    2. **Inline kwargs** – omit *config_file* and supply *datetime* and
       *area* as keyword arguments::

           hindcast = ERA5Hindcast(
               datetime="2020-08-26 00:00",
               area=(50, -130, 10, -50),
           )

    Parameters
    ----------
    config_file:
        Path to the key:value input file (e.g. ``input_for_Laura``).
        Mutually exclusive with *datetime*/*area*.
    datetime:
        Date/time of the hindcast snapshot.  Accepts a ``str`` parseable by
        :class:`pandas.Timestamp`, a :class:`datetime.datetime`, or a
        :class:`pandas.Timestamp`.  Keyword-only; requires *area*.
    area:
        Domain extent as ``(lat_max, lon_min, lat_min, lon_max)``.
        Keyword-only; requires *datetime*.
    **kwargs:
        Additional keyword arguments forwarded to subclasses.
    """

    def __init__(
        self,
        config_file: Optional[Union[str, os.PathLike]] = None,
        *,
        datetime: Optional[Union[str, _dt.datetime]] = None,
        area: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> None:
        if config_file is not None:
            self.config_file: Optional[str] = os.fspath(config_file)
            self.config: HindcastConfig = HindcastConfig.from_file(self.config_file)
            logger.info("Loaded config from %s", self.config_file)
        elif datetime is not None and area is not None:
            self.config_file = None
            self.config = HindcastConfig.from_datetime(datetime, area)
            logger.info("Built config from datetime=%r, area=%r", datetime, area)
        else:
            raise ValueError(
                "Provide either config_file or both datetime and area keyword arguments"
            )
        self.config.validate()
        self.lambert_conformal: str = create_lcc_mapping(self.config.area)

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
