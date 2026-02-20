"""Tests for the hindcast preprocessing framework."""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from erftools.hindcast import ERA5Hindcast, GFSHindcast, HindcastBase, HindcastConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INPUT = """\
year: 2020
month: 08
day: 26
time: 00:00
area: 50,-130,10,-50
"""


@pytest.fixture()
def input_file(tmp_path):
    """Write a temporary input file and return its path."""
    p = tmp_path / "input_for_test"
    p.write_text(SAMPLE_INPUT)
    return str(p)


@pytest.fixture()
def preprocessing_mock(monkeypatch):
    """Inject a MagicMock for erftools.preprocessing into sys.modules.

    This avoids importing the real module (which requires cartopy/cdsapi).
    """
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "erftools.preprocessing", mock)
    return mock


# ---------------------------------------------------------------------------
# HindcastConfig tests
# ---------------------------------------------------------------------------


class TestHindcastConfig:
    def test_from_file_parses_fields(self, input_file):
        cfg = HindcastConfig.from_file(input_file)
        assert cfg.year == 2020
        assert cfg.month == 8
        assert cfg.day == 26
        assert cfg.time == "00:00"
        assert cfg.area == [50.0, -130.0, 10.0, -50.0]

    def test_from_file_missing_file(self):
        with pytest.raises(FileNotFoundError):
            HindcastConfig.from_file("/nonexistent/path/input_for_missing")

    def test_validate_passes_on_valid_config(self, input_file):
        cfg = HindcastConfig.from_file(input_file)
        cfg.validate()  # should not raise

    def test_validate_wrong_area_length(self, input_file):
        cfg = HindcastConfig.from_file(input_file)
        cfg.area = [50.0, -130.0, 10.0]
        with pytest.raises(ValueError, match="4 values"):
            cfg.validate()

    def test_validate_bad_lat_order(self, input_file):
        cfg = HindcastConfig.from_file(input_file)
        cfg.area = [10.0, -130.0, 50.0, -50.0]  # lat_max < lat_min
        with pytest.raises(ValueError, match="lat_max"):
            cfg.validate()

    def test_validate_bad_lon_order(self, input_file):
        cfg = HindcastConfig.from_file(input_file)
        cfg.area = [50.0, -50.0, 10.0, -130.0]  # lon_max < lon_min
        with pytest.raises(ValueError, match="lon_max"):
            cfg.validate()

    def test_from_file_accepts_pathlike(self, tmp_path):
        """from_file should accept pathlib.Path without str() conversion."""
        import pathlib

        p = tmp_path / "input_pathlike"
        p.write_text(SAMPLE_INPUT)
        cfg = HindcastConfig.from_file(p)  # Pass Path object directly
        assert cfg.year == 2020

    def test_from_file_ignores_comment_lines(self, tmp_path):
        content = "# comment\nyear: 2021\nmonth: 01\nday: 01\ntime: 12:00\narea: 40,-100,20,-80\n"
        p = tmp_path / "input_with_comment"
        p.write_text(content)
        cfg = HindcastConfig.from_file(p)
        assert cfg.year == 2021

    def test_from_file_strips_whitespace(self, tmp_path):
        content = (
            "  year  :  2019  \n  month : 07 \n  day : 04 \n"
            "  time : 06:00 \n  area : 45,-110,30,-90 \n"
        )
        p = tmp_path / "input_whitespace"
        p.write_text(content)
        cfg = HindcastConfig.from_file(p)
        assert cfg.year == 2019
        assert cfg.month == 7

    def test_from_file_warns_on_malformed_line(self, tmp_path, caplog):
        """Non-empty, non-comment lines without ':' should trigger a warning."""
        import logging

        content = "year: 2020\nmonth: 01\nday: 01\ntime: 00:00\narea: 40,-100,20,-80\nBAD_LINE\n"
        p = tmp_path / "input_bad"
        p.write_text(content)
        with caplog.at_level(logging.WARNING, logger="erftools.hindcast.config"):
            cfg = HindcastConfig.from_file(p)
        assert any("BAD_LINE" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# HindcastBase tests
# ---------------------------------------------------------------------------


class ConcreteHindcast(HindcastBase):
    """Minimal concrete subclass for testing the abstract base."""

    def download(self) -> Any:
        return "downloaded"

    def process(self, download_result: Any) -> None:
        self.last_result = download_result


class TestHindcastBase:
    def test_init_sets_config(self, input_file):
        h = ConcreteHindcast(input_file)
        assert h.config.year == 2020
        assert h.config_file == input_file

    def test_init_creates_lambert_conformal(self, input_file):
        h = ConcreteHindcast(input_file)
        assert "+proj=lcc" in h.lambert_conformal

    def test_run_calls_download_then_process(self, input_file):
        h = ConcreteHindcast(input_file)
        h.run()
        assert h.last_result == "downloaded"

    def test_setup_output_dirs(self, input_file, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        h = ConcreteHindcast(input_file)
        h._setup_output_dirs("a/b/c", "d/e")
        assert (tmp_path / "a" / "b" / "c").is_dir()
        assert (tmp_path / "d" / "e").is_dir()


# ---------------------------------------------------------------------------
# ERA5Hindcast tests
# ---------------------------------------------------------------------------


class TestERA5Hindcast:
    def test_init_reanalysis(self, input_file):
        h = ERA5Hindcast(input_file)
        assert not h.do_forecast
        assert h.forecast_time_hours is None

    def test_init_forecast_requires_params(self, input_file):
        with pytest.raises(ValueError, match="forecast_time_hours"):
            ERA5Hindcast(input_file, do_forecast=True)

    def test_init_forecast_ok_with_params(self, input_file):
        h = ERA5Hindcast(
            input_file,
            do_forecast=True,
            forecast_time_hours=72,
            interval_hours=3,
        )
        assert h.do_forecast
        assert h.forecast_time_hours == 72
        assert h.interval_hours == 3

    def test_download_reanalysis_calls_correct_function(
        self, input_file, preprocessing_mock
    ):
        area = [50.0, -130.0, 10.0, -50.0]
        preprocessing_mock.Download_ERA5_SurfaceData.return_value = (
            ["era5_surf_test.grib"],
            area,
        )
        h = ERA5Hindcast(input_file)
        result = h.download()
        preprocessing_mock.Download_ERA5_SurfaceData.assert_called_once_with(input_file)
        assert result[1] == area

    def test_download_forecast_calls_correct_function(
        self, input_file, preprocessing_mock
    ):
        area = [50.0, -130.0, 10.0, -50.0]
        preprocessing_mock.Download_ERA5_ForecastSurfaceData.return_value = (
            ["era5_surf_001.grib"],
            area,
        )
        h = ERA5Hindcast(
            input_file,
            do_forecast=True,
            forecast_time_hours=72,
            interval_hours=3,
        )
        result = h.download()
        preprocessing_mock.Download_ERA5_ForecastSurfaceData.assert_called_once_with(
            input_file, 72, 3
        )


# ---------------------------------------------------------------------------
# GFSHindcast tests
# ---------------------------------------------------------------------------


class TestGFSHindcast:
    def test_init_defaults(self, input_file, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        h = GFSHindcast(input_file)
        assert not h.do_forecast
        assert h.product == "forecast"

    def test_download_reanalysis_calls_correct_function(
        self, input_file, tmp_path, monkeypatch, preprocessing_mock
    ):
        monkeypatch.chdir(tmp_path)
        area = [50.0, -130.0, 10.0, -50.0]
        preprocessing_mock.Download_GFS_Data.return_value = (
            "gfs.0p25.file.grib2",
            area,
        )
        h = GFSHindcast(input_file)
        result = h.download()
        preprocessing_mock.Download_GFS_Data.assert_called_once_with(
            input_file, product="forecast"
        )
        assert result[0] == "gfs.0p25.file.grib2"

    def test_download_forecast_calls_correct_function(
        self, input_file, tmp_path, monkeypatch, preprocessing_mock
    ):
        monkeypatch.chdir(tmp_path)
        area = [50.0, -130.0, 10.0, -50.0]
        preprocessing_mock.Download_GFS_ForecastData.return_value = (
            ["gfs.0p25.f000.grib2", "gfs.0p25.f003.grib2"],
            area,
        )
        h = GFSHindcast(input_file, do_forecast=True)
        h.download()
        preprocessing_mock.Download_GFS_ForecastData.assert_called_once_with(input_file)

    def test_process_calls_read_for_each_file(
        self, input_file, tmp_path, monkeypatch, preprocessing_mock
    ):
        monkeypatch.chdir(tmp_path)
        h = GFSHindcast(input_file, do_forecast=True)
        area = [50.0, -130.0, 10.0, -50.0]
        files = ["gfs.f000.grib2", "gfs.f003.grib2"]
        h.process((files, area))
        assert preprocessing_mock.ReadGFS_3DData.call_count == 2
        for call, fname in zip(
            preprocessing_mock.ReadGFS_3DData.call_args_list, files
        ):
            assert call.args[0] == fname
            assert call.args[1] == area
