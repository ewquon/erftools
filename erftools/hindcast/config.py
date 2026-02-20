"""Configuration dataclass for ERA5/GFS hindcast preprocessing."""
from __future__ import annotations

import datetime as _dt
import logging
import os
from dataclasses import dataclass
from typing import List, Sequence, Union

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

    @classmethod
    def from_datetime(
        cls,
        dt: Union[str, _dt.datetime],
        area: Sequence[float],
    ) -> "HindcastConfig":
        """Construct a :class:`HindcastConfig` from a datetime-like object and area.

        Parameters
        ----------
        dt:
            The date/time of the hindcast snapshot.  Accepted types are:

            * ``str`` – ISO-format string (e.g. ``"2020-08-26"``,
              ``"2020-08-26 00:00"``).  When :mod:`pandas` is available,
              any string parseable by :class:`pandas.Timestamp` is accepted;
              otherwise :func:`datetime.datetime.fromisoformat` is used.
            * :class:`datetime.datetime` or :class:`pandas.Timestamp`
              (which is a subclass of :class:`datetime.datetime`).

        area:
            Sequence of four floats ``[lat_max, lon_min, lat_min, lon_max]``.

        Returns
        -------
        HindcastConfig
        """
        if isinstance(dt, str):
            try:
                import pandas as pd  # preferred: handles non-ISO formats too
                dt = pd.Timestamp(dt)
            except ImportError:
                # Fallback for environments without pandas: accept ISO format strings
                dt = _dt.datetime.fromisoformat(dt)
        # At this point dt is a datetime.datetime (pd.Timestamp inherits from it)
        if not isinstance(dt, _dt.datetime):
            raise TypeError(
                f"dt must be a str, datetime.datetime, or pandas.Timestamp; got {type(dt)!r}"
            )
        time_str = f"{dt.hour:02d}:{dt.minute:02d}"
        return cls(
            year=int(dt.year),
            month=int(dt.month),
            day=int(dt.day),
            time=time_str,
            area=list(area),
        )

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
