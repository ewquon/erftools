"""Configuration dataclass for ERA5/GFS hindcast preprocessing."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Union

logger = logging.getLogger(__name__)


@dataclass
class HindcastConfig:
    """Configuration parsed from a hindcast input file.

    The expected file format is a text file with ``key: value`` pairs::

        year: 2020
        month: 08
        day: 26
        time: 00:00
        area: 50,-130,10,-50

    The ``area`` field is ``lat_max, lon_min, lat_min, lon_max``.
    """

    year: int
    month: int
    day: int
    time: str  # "HH:MM"
    area: List[float]  # [lat_max, lon_min, lat_min, lon_max]

    @classmethod
    def from_file(cls, filename: Union[str, os.PathLike]) -> "HindcastConfig":
        """Parse a key:value input file into a :class:`HindcastConfig`.

        Parameters
        ----------
        filename:
            Path to the input file.  Accepts both ``str`` and
            :class:`os.PathLike` objects (e.g. :class:`pathlib.Path`).

        Returns
        -------
        HindcastConfig
        """
        filename = os.fspath(filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Input file not found: {filename}")
        data: dict = {}
        with open(filename, "r") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" not in stripped:
                    logger.warning(
                        "%s line %d: skipping unrecognised line: %r",
                        filename,
                        lineno,
                        stripped,
                    )
                    continue
                key, _, value = stripped.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "area":
                    data[key] = [float(x) for x in value.split(",")]
                elif key == "time":
                    data[key] = value
                else:
                    try:
                        data[key] = int(value)
                    except ValueError:
                        data[key] = value
        return cls(**data)

    def validate(self) -> None:
        """Validate the parsed configuration values.

        Raises
        ------
        ValueError
            If any required field is missing or any value is out of range.
        """
        for field_name in ("year", "month", "day", "time", "area"):
            if getattr(self, field_name, None) is None:
                raise ValueError(f"Missing required field: {field_name}")
        if len(self.area) != 4:
            raise ValueError(
                "'area' must have exactly 4 values: lat_max, lon_min, lat_min, lon_max"
            )
        lat_max, lon_min, lat_min, lon_max = self.area
        if lat_max <= lat_min:
            raise ValueError(
                "area: lat_max (1st value) must be greater than lat_min (3rd value)"
            )
        if lon_max <= lon_min:
            raise ValueError(
                "area: lon_max (4th value) must be greater than lon_min (2nd value)"
            )
