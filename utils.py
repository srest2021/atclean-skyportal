#!/usr/bin/env python

from abc import ABC, abstractmethod
import bisect
from configparser import ConfigParser
import configparser
from datetime import datetime
from functools import reduce
from getpass import getpass
from typing import Callable, Dict, Any, List, Optional, Self, Set, Tuple, Type
import re, json, requests, time, sys, io, os
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from astropy.nddata import bitmask
from collections import OrderedDict, namedtuple
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path

from constants import C4_SMALL_N
from pdastro import AnotB, not_AandB


class CustomLogger:
    """
    Simple custom logger that prints messages to stdout when verbose is enabled.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the logger.

        :param verbose: If True, messages will be printed.
        """
        self.verbose = verbose

    def info(self, text: str, prefix="", newline: bool = False):
        """
        Print an info message if verbose is enabled.

        :param text: Message content.
        :param prefix: Optional prefix (e.g. "WARNING: ").
        :param newline: Whether to prepend a newline before the message.
        """
        newline_part = "\n" if newline else ""
        if self.verbose:
            print(f"{newline_part}{prefix}{text}")

    def warning(self, text: str, newline: bool = False):
        """
        Print a warning message.

        :param text: Message content.
        :param newline: Whether to prepend a newline before the message.
        """
        self.info(text, prefix="WARNING: ", newline=newline)

    def error(self, text: str, newline: bool = False):
        """
        Print an error message.

        :param text: Message content.
        :param newline: Whether to prepend a newline before the message.
        """
        self.info(text, prefix="ERROR: ", newline=newline)

    def success(self, text: str = "Success", newline: bool = False):
        """
        Print a success message.

        :param text: Message content.
        :param newline: Whether to prepend a newline before the message.
        """
        self.info(text, newline=newline)


def hexstring_to_int(hexstring: str):
    """
    Convert a hexadecimal string to an integer.

    :param hexstring: Hex string (e.g. "0x10").
    :return: Integer value.
    """
    return int(hexstring, 16)


def nan_if_none(value: Any) -> Any:
    """
    Convert None to np.nan, otherwise return the value unchanged.

    :param value: Input value.
    :return: np.nan if value is None, else the original value.
    """
    return np.nan if value is None else value


def combine_flags(flags: List[int]) -> int:
    """
    Bitwise OR a list of integer flags into a single integer.

    :param flags: List of flag values.
    :return: Combined flag as integer.
    """
    return reduce(lambda x, y: x | y, flags, 0)


def new_row(t: Optional[pd.DataFrame], d: Optional[Dict] = None):
    """
    Append a new row to a DataFrame (or create one if empty).

    :param t: Original DataFrame or None.
    :param d: Dictionary representing a single row.
    :return: New DataFrame with the row added.
    """
    if d is None:
        d = {}

    new_row_df = pd.DataFrame([d])
    if t is None or t.empty:
        t = new_row_df
    else:
        new_row_df = new_row_df.reindex(columns=t.columns)
        t = pd.concat([t, new_row_df], axis=0, ignore_index=True)
    return t


class BaseAngle(ABC):
    """
    Abstract base class for angle parsing (RA or Dec).
    """

    def __init__(self, angle: str | Angle):
        """
        Initialize with an angle string or astropy Angle.

        :param angle: Input angle (string or Angle).
        """
        self.angle: Angle = (
            angle if isinstance(angle, Angle) else self._parse_angle(str(angle))
        )

    @abstractmethod
    def _parse_angle(self, string: str) -> Angle:
        """
        Abstract method to parse string input into Angle.

        :param string: Input angle string.
        :return: Parsed Angle object.
        """
        pass


class RA(BaseAngle):
    def _parse_angle(self, string: str) -> Angle:
        """
        Parse RA angle, using hours if ':' is present, degrees otherwise.
        """
        s = re.compile(":")
        if isinstance(string, str) and s.search(string):
            return Angle(string, u.hour)
        else:
            return Angle(string, u.degree)


class Dec(BaseAngle):
    def _parse_angle(self, string: str) -> Angle:
        """
        Parse Dec angle, always using degrees.
        """
        return Angle(string, u.degree)


class Coordinates:
    """
    Object representing sky coordinates (RA, Dec).
    """

    def __init__(self, ra: str | Angle | RA, dec: str | Angle | Dec):
        """
        Initialize coordinates from raw strings, astropy Angles, or RA/Dec objects.

        :param ra: Right Ascension.
        :param dec: Declination.
        """
        self.ra: RA = RA(ra) if not isinstance(ra, RA) else ra
        self.dec: Dec = Dec(dec) if not isinstance(dec, Dec) else dec

    def get_RA_str(self) -> str:
        """
        Return RA in degrees with high precision.

        :return: RA string in degrees.
        """
        return f"{self.ra.angle.degree:0.14f}"

    def get_Dec_str(self) -> str:
        """
        Return Dec in degrees with high precision.

        :return: Dec string in degrees.
        """
        return f"{self.dec.angle.degree:0.14f}"

    def get_distance(self, other: Self) -> Angle:
        """
        Compute angular separation from another coordinate.

        :param other: Other Coordinates object.
        :return: Angular separation as Angle.
        """
        c1 = SkyCoord(self.ra.angle, self.dec.angle, frame="fk5")
        c2 = SkyCoord(other.ra.angle, other.dec.angle, frame="fk5")
        return c1.separation(c2)

    def __str__(self):
        return f"RA {self.get_RA_str()}, Dec {self.get_Dec_str()}"


def parse_comma_separated_string(string: Optional[str]):
    """
    Parse a comma-separated string into a list of stripped strings.

    :param string: Input string like "RA, Dec".
    :return: List of strings or None.
    :raises RuntimeError: If parsing fails.
    """
    if string is None:
        return None

    try:
        return [item.strip() for item in string.split(",")]
    except Exception as e:
        raise RuntimeError(
            f"Could not parse comma-separated string: {string}" f"\nERROR: {str(e)}"
        )


def parse_arg_coords(arg_coords: Optional[str]) -> Optional[Coordinates]:
    """
    Parse a comma-separated string (i.e., a coordinates argument string, e.g., --center_coords) into a Coordinates object.

    :param arg_coords: Comma-separated RA and Dec string.
    :return: Coordinates object or None.
    :raises RuntimeError: If parsing fails or format is invalid.
    """
    parsed_coords = parse_comma_separated_string(arg_coords)
    if parsed_coords is None:
        return None
    if len(parsed_coords) > 2:
        raise RuntimeError(
            "Too many coordinates in argument! Please provide comma-separated RA and Dec onlyy."
        )
    if len(parsed_coords) < 2:
        raise RuntimeError(
            "Too few coordinates in argument! Please provide comma-separated RA and Dec."
        )

    return Coordinates(parsed_coords[0], parsed_coords[1])


class ColumnNames:
    """
    Utility class for managing required and optional column name mappings.
    """

    def __init__(
        self,
        required_colnames: Optional[Dict[str, str]] = None,
        optional_colnames: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ):
        """
        Initialize ColumnNames with dictionaries of required and optional column mappings.

        :param required_colnames: Mapping of internal keys to required column names.
        :param optional_colnames: Mapping of internal keys to optional column names.
        :param verbose: Whether to enable verbose logging.
        """
        self.logger = CustomLogger(verbose=verbose)

        self._required_colnames = (
            required_colnames if required_colnames is not None else {}
        )
        self._optional_colnames = (
            optional_colnames if optional_colnames is not None else {}
        )

    @property
    def required(self) -> Set:
        """
        Return the set of required column names.
        """
        return set(self._required_colnames.values())

    @property
    def optional(self) -> Set:
        """
        Return the set of optional column names.
        """
        return set(self._optional_colnames.values())

    @property
    def all(self) -> Set:
        """
        Return the set of all column names (required + optional).
        """
        return self.required.union(self.optional)

    def add(
        self, key: str, name: str, is_required: bool = False, overwrite: bool = False
    ):
        """
        Add a single column name mapping.

        :param key: Internal key name.
        :param name: Actual column name used in the DataFrame.
        :param is_required: Whether the column is required.
        :param overwrite: Whether to allow overwriting an existing mapping.
        :raises ValueError: If the key or name is invalid.
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Column key must be a non-empty string")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Column name for key '{key}' must be a non-empty string")

        # if key exists but the name is different, raise an error
        existing_name = self._required_colnames.get(key) or self._optional_colnames.get(
            key
        )
        if existing_name and existing_name != name and not overwrite:
            self.logger.warning(
                f"Column key '{key}' is already defined with a different name '{existing_name}'"
            )

        if is_required:
            self._required_colnames[key] = name
        else:
            self._optional_colnames[key] = name

    def add_many(self, coldict: Dict[str, str], is_required: bool = False):
        """
        Add multiple column name mappings.

        :param coldict: Dictionary of key-to-name mappings.
        :param is_required: Whether these columns are required.
        """
        for key, name in coldict.items():
            self.add(key, name, is_required=is_required)

    def update(self, key: str, name: str):
        """
        Update an existing column name mapping.

        :param key: Internal key name.
        :param name: New column name.
        :raises RuntimeError: If the key does not exist.
        """
        if key in self._required_colnames:
            self._required_colnames[key] = name
        elif key in self._optional_colnames:
            self._optional_colnames[key] = name
        else:
            raise RuntimeError(
                f"Cannot update non-existing column key '{key}' with name '{name}'"
            )

    def update_many(self, coldict: Dict[str, str]):
        """
        Update multiple column name mappings.

        :param coldict: Dictionary of key-to-name updates.
        :param is_required: Whether these columns are required.
        """
        for key, name in coldict.items():
            self.update(key, name)

    def remove(self, key: str):
        """
        Remove a column name mapping by key.
        """
        if key in self._required_colnames:
            del self._required_colnames[key]
        if key in self._optional_colnames:
            del self._optional_colnames[key]

    def has(self, name: str):
        """
        Check if a column name is defined (in required or optional).
        """
        return name in self._required_colnames or name in self._optional_colnames

    def __getattr__(self, name: str) -> Optional[str]:
        """
        Allow attribute-style access to column names by key.

        Example:
        1. `c = ColumnNames()`
        2. `c.add("mjd", "MJD")`

        Now `c.mjd` returns `"MJD"`.

        :param name: Internal key name.
        :return: Column name string.
        :raises AttributeError: If key is not found.
        """
        try:
            # Safely access dicts without triggering __getattr__
            required_colnames = object.__getattribute__(self, "_required_colnames")
            optional_colnames = object.__getattribute__(self, "_optional_colnames")
        except AttributeError:
            # If the object is half-initialized (e.g., during deepcopy), just fail gracefully
            raise AttributeError(
                f"Cannot access required or optional column name dicts"
            )

        if name in required_colnames:
            return required_colnames[name]
        if name in optional_colnames:
            return optional_colnames[name]

        raise AttributeError(f"{name} not found in column names")

    def __str__(self) -> str:
        lines = ["-- Required Columns --"]
        for k, v in self._required_colnames.items():
            lines.append(f"{k}: {v}")

        lines.append("-- Optional Columns --")
        for k, v in self._optional_colnames.items():
            lines.append(f"{k}: {v}")

        return "\n".join(lines)


class CleanedColumnNames(ColumnNames):
    """
    Predefined column names for cleaned light curve data.
    """

    def __init__(self):
        required_colnames = {
            "mjd": "MJD",
            "ra": "RA",
            "dec": "Dec",
            "mag": "m",
            "dmag": "dm",
            "flux": "uJy",
            "dflux": "duJy",
            "filter": "F",
            "limiting_mag": "mag5sig",
            "mask": "Mask",
        }
        optional_colnames = {"chisquare": "chi/N", "snr": "SNR"}
        super().__init__(
            required_colnames=required_colnames, optional_colnames=optional_colnames
        )


class BinnedColumnNames(ColumnNames):
    """
    Predefined column names for binned light curve data.
    """

    def __init__(self):
        required_colnames = {
            "mjdbin": "MJDbin",
            "mjd": "MJD",
            "flux": "flux",
            "dflux": "dflux",
            "mag": "m",
            "dmag": "dm",
            "filter": "F",
            "mask": "Mask",
        }
        optional_colnames = {
            "stdev": "stdev",
            "x2": "X2norm",
            "nclip": "Nclip",
            "ngood": "Ngood",
            "nexcluded": "Nexcluded",
        }
        super().__init__(
            required_colnames=required_colnames, optional_colnames=optional_colnames
        )


class StatParams:
    """
    Container for storing statistical parameters from SigmaClipper results.
    """

    def __init__(self, d: Optional[Dict[str, int | float | List[int] | None]] = None):
        """
        Initialize the StatParams object from a dictionary, or set all fields to None.

        :param d: Optional dictionary with initial values for fields.
        """
        self.FIELDS = [
            "mean",
            "mean_err",
            "stdev",
            "stdev_err",
            "X2norm",
            "Nclip",
            "Ngood",
            "Nchanged",
            "Nmask",
            "Nnan",
            "ix_good",
            "ix_clip",
        ]

        if d is not None:
            self.from_dict(d)
        else:
            self.from_dict({})

    def from_dict(self, d: Dict[str, int | float | None]):
        """
        Load values from a dictionary into the object's fields; only those in FIELDS are considered.

        :param d: Dictionary containing keys from self.FIELDS.
        """
        for key in self.FIELDS:
            setattr(self, key, d.get(key))

    def update(self, **kwargs):
        """
        Update fields with values from keyword arguments.

        :param kwargs: Named values to update; only those in self.FIELDS are considered.
        """
        for key in self.FIELDS:
            val = kwargs.get(key, None)
            if val is not None:
                setattr(self, key, val)

    def reset(self):
        """
        Reset all fields to None.
        """
        for key in self.FIELDS:
            setattr(self, key, None)

    def get_row(self, prefix="", skip=None) -> Dict[str, float]:
        """
        Convert parameters to a dictionary row for output (e.g., to a DataFrame).

        :param prefix: Optional string to prefix each key.
        :param skip: Optional list of field names to skip.
        :return: Dictionary mapping of (possibly prefixed) field names to values.
        """
        skip = set(skip or [])
        row = {}

        for name in self.FIELDS:
            if name not in skip:
                colname = prefix + name
                try:
                    row[colname] = getattr(self, name, np.nan)
                except Exception:
                    # Fallback in case something goes wrong (e.g., corrupted attribute)
                    row[colname] = np.nan

        return row

    def __str__(self):
        parts = []
        for key in self.FIELDS:
            val = getattr(self, key, None)
            if isinstance(val, float):
                parts.append(f"{key}={val:.17g}")  # Full float precision
            else:
                parts.append(f"{key}={val}")
        return f"StatParams({', '.join(parts)})"


class SigmaClipper:
    """
    Class for performing sigma clipping and robust averaging on numerical data.
    """

    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger(verbose=verbose)

        self.statparams = StatParams()
        self.converged = False
        self.i = 0

    @staticmethod
    def c4(n) -> float:
        """
        Returns the correction factor for the unbiased estimation of standard deviation.

        :param n: Sample size.
        :return: Correction factor.
        """
        """http://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation"""
        if n <= 6:
            return C4_SMALL_N[n]
        else:
            return (
                1.0
                - 1.0 / (4.0 * n)
                - 7.0 / (32.0 * n * n)
                - 19.0 / (128.0 * n * n * n)
            )

    @staticmethod
    def get_indices(default_length, indices=None) -> np.ndarray:
        """
        Get an array of indices.
        If no indices given, get all indices of array using default_length.

        :param default_length: Length of the array to index.
        :param indices: Optional array of indices to use.
        :return: Array of indices.
        """
        if indices is None:
            return np.arange(default_length)
        else:
            return np.asarray(indices)

    @staticmethod
    def ix_inrange(
        arr,
        lowlim=None,
        uplim=None,
        indices=None,
        exclude_lowlim=False,
        exclude_uplim=False,
    ) -> np.ndarray:
        """
        Returns indices where values are within specified limits.

        :param arr: Input array of values.
        :param lowlim: Lower bound for clipping.
        :param uplim: Upper bound for clipping.
        :param indices: Optional subset of indices to check.
        :param exclude_lowlim: If True, exclude values equal to lowlim.
        :param exclude_uplim: If True, exclude values equal to uplim.
        :return: Array of filtered indices.
        """
        indices = SigmaClipper.get_indices(len(arr), indices=indices)
        values = arr[indices]
        keep_mask = np.ones(len(indices), dtype=bool)

        if lowlim is not None:
            if exclude_lowlim:
                keep_mask &= values > lowlim
            else:
                keep_mask &= values >= lowlim

        if uplim is not None:
            if exclude_uplim:
                keep_mask &= values < uplim
            else:
                keep_mask &= values <= uplim

        return indices[keep_mask]

    @staticmethod
    def ix_unmasked(mask_arr, mask_val=None, indices=None) -> np.ndarray:
        """
        Returns indices of unmasked entries.

        :param mask_arr: Array of bitmask values.
        :param mask_val: Bitmask flag to test against.
        :param indices: Optional subset of indices.
        :return: Array of unmasked indices.
        """
        indices = SigmaClipper.get_indices(len(mask_arr), indices=indices)

        sub_mask = mask_arr[indices]
        if mask_val is None:
            keep = sub_mask == 0
        else:
            keep = bitmask.bitfield_to_boolean_mask(
                sub_mask.astype(int),
                ignore_flags=~mask_val,
                good_mask_value=True,
            )

        return indices[keep]

    @staticmethod
    def ix_not_null(arrays: List[np.ndarray], indices=None) -> np.ndarray:
        """
        Returns indices where all arrays have non-null values.

        :param arrays: List of arrays to check.
        :param indices: Optional subset of indices.
        :return: Array of indices with non-null values across all arrays.
        """
        if len(arrays) == 0:
            return np.array([], dtype=int)
        indices = SigmaClipper.get_indices(len(arrays[0]), indices=indices)

        keep_mask = np.ones(len(indices), dtype=bool)
        for arr in arrays:
            keep_mask &= pd.notnull(arr[indices])

        return indices[keep_mask]

    def reset(self):
        """
        Resets the internal state before a new sigma clipping run.
        """
        self.statparams = StatParams()
        self.converged = False
        self.i = 0

    def calcaverage_errorcut(
        self,
        data_arr: np.ndarray,
        noise_arr: np.ndarray,
        indices=None,
        mean: Optional[float] = None,
        Nsigma: Optional[float] = None,
        median_flag: bool = False,
    ):
        """
        Computes clipped mean and standard deviation based on individual errors.

        :param data_arr: Array of data.
        :param noise_arr: Array of uncertainties for each data point.
        :param indices: Optional subset of indices to operate on.
        :param mean: Initial guess for mean value.
        :param Nsigma: Sigma threshold for clipping.
        :param median_flag: Use median instead of weighted mean.
        :return: 1 if fewer than one good point, else 0.
        """
        indices = SigmaClipper.get_indices(len(data_arr), indices=indices)
        x = data_arr[indices]
        dx = noise_arr[indices]

        if Nsigma is not None and mean is not None:
            diff = np.abs(x - mean)
            good_ix = indices[diff <= Nsigma * dx]
            good_ix_bkp = deepcopy(self.statparams.ix_good)
        else:
            good_ix = indices
            good_ix_bkp = None

        Ngood = len(good_ix)

        if Ngood > 1:
            x_good = data_arr[good_ix]
            dx_good = noise_arr[good_ix]
            if median_flag:
                mean = np.median(x_good)
                stdev = np.sqrt(
                    np.sum((x_good - mean) ** 2.0) / (Ngood - 1.0)
                ) / SigmaClipper.c4(Ngood)
                mean_err = np.median(dx_good) / np.sqrt(Ngood - 1)
            else:
                w = 1.0 / (dx_good**2.0)
                mean = np.sum(x_good * w) / np.sum(w)
                mean_err = np.sqrt(1.0 / np.sum(w))
                stdev = np.std(x_good, ddof=1)

            stdev_err = stdev / np.sqrt(2.0 * Ngood)
            X2norm = np.sum(((x_good - mean) / dx_good) ** 2.0) / (Ngood - 1.0)
        elif Ngood == 1:
            mean = data_arr[good_ix[0]]
            mean_err = noise_arr[good_ix[0]]
            stdev = stdev_err = X2norm = None
        else:
            mean = mean_err = stdev = stdev_err = X2norm = None

        self.statparams.update(
            mean=mean,
            mean_err=mean_err,
            stdev=stdev,
            stdev_err=stdev_err,
            X2norm=X2norm,
            Ngood=Ngood,
            ix_good=good_ix,
            Nclip=len(indices) - Ngood,
            ix_clip=AnotB(indices, good_ix),
            Nchanged=(
                len(not_AandB(good_ix_bkp, good_ix)) if good_ix_bkp is not None else 0
            ),
        )

        return int(Ngood < 1)

    def calcaverage_sigmacut(
        self,
        data_arr: np.ndarray,
        noise_arr: Optional[np.ndarray] = None,
        indices=None,
        mean: Optional[float] = None,
        stdev: Optional[float] = None,
        Nsigma: Optional[float] = None,
        percentile_cut: bool = None,
        percentile_Nmin: Optional[float] = 3,
        median_flag: bool = False,
    ):
        """
        Computes clipped statistics using either sigma or percentile clipping.

        :param data_arr: Array of data.
        :param noise_arr: Optional array of uncertainties for each data point.
        :param indices: Optional subset of indices.
        :param mean: Initial guess for the mean.
        :param stdev: Initial guess for standard deviation.
        :param Nsigma: Sigma threshold for clipping.
        :param percentile_cut: Percentile cutoff for clipping residuals.
        :param percentile_Nmin: Minimum number of points for percentile clipping.
        :param median_flag: Use median for mean calculation.
        :return: 1 if fewer than one good point, else 0.
        """
        indices = SigmaClipper.get_indices(len(data_arr), indices=indices)
        if len(indices) == 0:
            self.reset()
            self.logger.warning("No data passed for sigma cut")
            return 2

        x = data_arr[indices]

        good_ix_bkp = None
        if percentile_cut is None or len(indices) <= percentile_Nmin:
            # if N-sigma cut and second iteration (i.e. we have a stdev from the first iteration), skip bad measurements
            if Nsigma is not None and stdev is not None and mean is not None:
                good_ix_bkp = deepcopy(self.statparams["ix_good"])
                good_ix = indices[np.abs(x - mean) <= Nsigma * stdev]
            else:
                good_ix = indices
        else:  # percentile clipping
            if mean is None:
                mean = np.median(x) if median_flag else np.mean(x)
            residuals = np.abs(x - mean)
            max_residual = np.percentile(residuals, percentile_cut)
            good_ix = indices[residuals < max_residual]

            if len(good_ix) < percentile_Nmin:
                sorted_residuals = np.sort(residuals)
                max_residual = sorted_residuals[percentile_Nmin - 1]
                good_ix = indices[residuals < max_residual]

        Ngood = len(good_ix)
        x_good = data_arr[good_ix]

        if Ngood > 1:
            if median_flag:
                mean = np.median(x_good)
                stdev = np.sqrt(
                    np.sum((x_good - mean) ** 2) / (Ngood - 1.0)
                ) / SigmaClipper.c4(Ngood)
            else:
                mean = np.mean(x_good)
                stdev = np.std(x_good, ddof=1)

            mean_err = stdev / np.sqrt(Ngood - 1.0)
            stdev_err = stdev / np.sqrt(2.0 * Ngood)
            if noise_arr is None:
                X2norm = np.sum(((x_good - mean) / stdev) ** 2) / (Ngood - 1.0)
            else:
                dx_good = noise_arr[good_ix]
                X2norm = np.sum(((x_good - mean) / dx_good) ** 2) / (Ngood - 1.0)
        elif Ngood == 1:
            mean = x_good[0]
            mean_err = noise_arr[good_ix[0]] if noise_arr is not None else None
            stdev = stdev_err = X2norm = None
        else:
            mean = mean_err = stdev = stdev_err = X2norm = None

        self.statparams.update(
            mean=mean,
            mean_err=mean_err,
            stdev=stdev,
            stdev_err=stdev_err,
            X2norm=X2norm,
            Ngood=Ngood,
            ix_good=good_ix,
            Nclip=len(indices) - Ngood,
            ix_clip=AnotB(indices, good_ix),
            Nchanged=(
                len(not_AandB(good_ix_bkp, good_ix)) if good_ix_bkp is not None else 0
            ),
        )

        return int(Ngood < 1)

    def calcaverage_sigmacutloop(
        self,
        data_arr: np.ndarray,
        indices=None,
        noise_arr: Optional[np.ndarray] = None,
        mask_arr: Optional[np.ndarray] = None,
        mask_val: float = None,
        Nsigma: float = 3.0,
        N_max_iterations: int = 10,
        remove_nan: bool = True,
        sigmacut_flag: bool = False,
        percentile_cut_firstiteration=None,
        median_firstiteration: bool = True,
    ):
        """
        Runs iterative sigma or error clipping until convergence or maximum iterations.

        :param data_arr: Array of data values.
        :param indices: Optional initial indices.
        :param noise_arr: Optional array of uncertainties.
        :param mask_arr: Optional mask array to exclude data.
        :param mask_val: Bitmask value used to determine exclusion.
        :param Nsigma: Sigma threshold for clipping.
        :param N_max_iterations: Maximum number of iterations to run.
        :param remove_nan: Whether to exclude NaN entries.
        :param sigmacut_flag: Whether to use sigma clipping (vs error clipping).
        :param percentile_cut_firstiteration: Optional percentile clipping for first iteration.
        :param median_firstiteration: Use median in first iteration.
        :return: True if not converged, False if converged.
        """
        self.reset()
        if noise_arr is None:
            sigmacut_flag = True
        indices = SigmaClipper.get_indices(len(data_arr), indices=indices)

        # exclude data if wanted
        if mask_arr is not None:
            Ntot = len(indices)
            indices = SigmaClipper.ix_unmasked(
                mask_arr, mask_val=mask_val, indices=indices
            )
            self.statparams.Nmask = Ntot - len(indices)
            self.logger.info(
                f"Keeping {len(indices)} out of {Ntot}, skipping {Ntot - len(indices)} because of masking (maskval={mask_val})"
            )
        else:
            self.statparams.Nmask = 0

        # remove null values if wanted
        if remove_nan:
            arrays = [data_arr]
            if noise_arr is not None:
                arrays.append(noise_arr)
            if mask_arr is not None:
                arrays.append(mask_arr)

            Ntot = len(indices)
            indices = SigmaClipper.ix_not_null(arrays, indices=indices)
            self.statparams.Nnan = Ntot - len(indices)
            self.logger.info(
                f"NaN filtering: kept {len(indices)} / {Ntot}, removed {Ntot - len(indices)}"
            )
        else:
            self.statparams.Nnan = 0

        for i in range(N_max_iterations):
            if self.converged:
                break
            self.i = i

            median_flag = median_firstiteration and i == 0 and Nsigma is not None
            percentile_cut = percentile_cut_firstiteration if i == 0 else None

            if sigmacut_flag:
                error_flag = self.calcaverage_sigmacut(
                    data_arr,
                    noise_arr=noise_arr,
                    indices=indices,
                    mean=self.statparams.mean,
                    stdev=self.statparams.stdev,
                    Nsigma=Nsigma,
                    median_flag=median_flag,
                    percentile_cut=percentile_cut,
                )
            else:
                error_flag = self.calcaverage_errorcut(
                    data_arr,
                    noise_arr,
                    indices=indices,
                    mean=self.statparams.mean,
                    Nsigma=Nsigma,
                    median_flag=median_flag,
                )

            if (
                error_flag
                or self.statparams.stdev is None
                or (self.statparams.stdev == 0.0 and sigmacut_flag)
                or self.statparams.mean is None
            ):
                self.converged = False
                break

            if Nsigma in (None, 0.0):
                self.converged = True
                break

            if i > 0 and self.statparams.Nchanged == 0 and not median_flag:
                self.converged = True
                break

        if not self.converged:
            self.logger.warning("No convergence")
        return not self.converged


class PrimaryFlag:
    """
    Represents a primary bitmask flag used to denote
    a "bad" measurement associated with a cut.
    """

    def __init__(
        self,
        name: str,
        value: int,
        description: Optional[str] = None,
        percent_cut: Optional[float] = None,
    ):
        """
        Initialize a PrimaryFlag.

        :param name: Name of the flag.
        :param value: Integer value of the bitmask flag.
        :param description: Optional description of the flag.
        :param percent_cut: Optional percent of measurements cut from the SN light curve.
        """
        self.name = name
        self.description = description
        self.value = value
        self.percent_cut = percent_cut
        self._is_primary = True

    @property
    def hex(self):
        """
        Return the hexadecimal string representation of the flag value.
        """
        return hex(self.value)

    def __str__(self):
        return f"'{self.name}' {'primary' if self._is_primary else 'secondary'} flag (hex value {self.hex}{f'; {self.percent_cut:0.2f}% flagged in SN' if self.percent_cut is not None else ''}):\n\t{self.description}"


class Flag(PrimaryFlag):
    """
    Represents a secondary bitmask flag associated with a cut.
    These can be used to provide additional metadata
    or denote "questionable" (but not "bad") conditions.
    """

    def __init__(self, name: str, value: int, description: Optional[str] = None):
        """
        Initialize a secondary Flag.

        :param name: Name of the flag.
        :param value: Integer value of the bitmask flag.
        :param description: Optional description of the flag.
        """
        super().__init__(name, value, description)
        self._is_primary = False


class Cut:
    """
    Represents a named cut with an optional primary flag and optional secondary flags.
    """

    def __init__(
        self,
        name: str,
        primary_flag: Optional[PrimaryFlag] = None,
        secondary_flags: Optional[List[Flag]] = None,
        description: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize a Cut with flags and metadata.

        :param name: Name of the cut.
        :param primary_flag: PrimaryFlag associated with the cut.
        :param secondary_flags: List of secondary Flag objects.
        :param description: Optional description of the cut.
        :param verbose: Whether to enable verbose logging.
        """
        self.name = name
        self.description: Optional[str] = description
        self._primary_flag: Optional[PrimaryFlag] = primary_flag
        self._secondary_flags: Dict[str, Flag] = (
            {flag.name: flag for flag in secondary_flags} if secondary_flags else {}
        )

        self.logger = CustomLogger(verbose=verbose)

    @property
    def primary_flag(self):
        return self._primary_flag

    def has_secondary_flag(self, name: str):
        """
        Check if a secondary flag with the given name exists.
        """
        return name in self._secondary_flags

    def add_flag(self, flag: Flag | PrimaryFlag):
        """
        Add a primary or secondary flag to the cut.
        If a primary flag or a secondary flag with the same name already exists, overwrite it.

        :param flag: Flag to add (either primary or secondary).
        :raises ValueError: If a primary flag name conflicts with existing secondary flag.
        """
        if isinstance(flag, Flag):
            if self.has_secondary_flag(flag.name):
                self.logger.warning(
                    f"Cut already has flag '{flag.name}'; overwriting..."
                )
            self._secondary_flags[flag.name] = flag
        else:
            if self.has_secondary_flag(flag.name):
                raise ValueError(
                    f"Cannot assign primary flag to name '{flag.name}' corresponding to existing secondary flag"
                )
            if self._primary_flag is not None:
                self.logger.warning(f"Overwriting primary flag...")
            self._primary_flag = flag

    def remove_flag(self, name: str):
        # TODO
        pass

    def get_flag(self, name: str):
        """
        Retrieve a flag by name (primary or secondary).

        :param name: Name of the flag.
        :return: The corresponding Flag or PrimaryFlag object.
        :raises ValueError: If the flag does not exist.
        """
        if self._primary_flag is not None and name == self._primary_flag.name:
            return self._primary_flag

        if not self.has_secondary_flag(name):
            raise ValueError(f"No such flag '{name}'")

        return self._secondary_flags[name]

    def get_combined_flags_value(self) -> int:
        """
        Return the combined integer value of all flags in the cut.
        """
        flags = [flag.value for flag in self._secondary_flags.values()]
        if self._primary_flag is not None:
            flags.append(self._primary_flag.value)
        return combine_flags(flags)

    def __str__(self):
        out = [f"{self.name}:{f' {self.description}' if self.description else ''}"]
        if self._primary_flag is not None:
            out.append("- " + self._primary_flag.__str__())
        if self._secondary_flags:
            for flag in self._secondary_flags.values():
                out.append("- " + flag.__str__())
        return "\n".join(out)


class CutHistory:
    """
    Tracks and manages the history of applied cuts.
    """

    def __init__(self, verbose: bool = False):
        self._cuts: OrderedDict[str, Cut] = OrderedDict()
        self.logger = CustomLogger(verbose=verbose)

    def has(self, name: str):
        """
        Check if a cut with the given name exists.
        """
        return name in self._cuts

    @property
    def cuts(self):
        """
        Return ordered list of applied cuts.
        """
        return list(self._cuts.values())

    @property
    def cut_names(self):
        """
        Return ordered list of applied cut names.
        """
        return list(self._cuts.keys())

    def add(self, cut: Cut):
        """
        Add a cut to the history.
        """
        if cut.name in self._cuts:
            self.logger.warning(f"Cut '{cut.name}' already exists; overwriting...")
        self._cuts[cut.name] = cut

    def remove(self, name: str):
        """
        Remove a cut by name.
        """
        if name in self._cuts:
            del self._cuts[name]
        else:
            self.logger.warning(f"Cannot remove nonexistent cut '{name}'")

    def get_primary_flags(self) -> int:
        """
        Get the combined integer value of all primary flags across cuts.
        """
        res = 0
        for cut in self._cuts.values():
            if cut.primary_flag is not None:
                res |= cut.primary_flag.value
        return res

    def get_secondary_flags(self) -> int:
        """
        Get the combined integer value of all secondary flags across cuts.
        """
        res = 0
        for cut in self._cuts.values():
            for flag in cut._secondary_flags.values():
                res |= flag.value
        return res

    def get_primary_flag(self, cut_name: str) -> Optional[PrimaryFlag]:
        """
        Get the primary flag for a cut by name if it exists.

        :param name: Name of the cut.
        :return: PrimaryFlag if it exists, otherwise None.
        """
        if cut_name in self._cuts and self._cuts[cut_name].primary_flag is not None:
            return self._cuts[cut_name].primary_flag
        return None

    def get_UncertaintyCut_flag(self) -> Optional[PrimaryFlag]:
        """
        Get the primary flag for the uncertainty cut if it exists.

        :return: PrimaryFlag for high uncertainty cut or None if not found.
        """
        return self.get_primary_flag("Uncertainty Cut")

    def get_ChiSquareCut_flag(self) -> Optional[PrimaryFlag]:
        """
        Get the primary flag for the chi-square cut if it exists.

        :return: PrimaryFlag for high PSF chi-square cut or None if not found.
        """
        return self.get_primary_flag("PSF Chi-Square Cut")

    def get_ControlLightCurveCut_flag(self) -> Optional[PrimaryFlag]:
        """
        Get the primary flag for the control light curve cut if it exists.

        :return: PrimaryFlag for control light curve cut or None if not found.
        """
        return self.get_primary_flag("Control Light Curve Cut")

    def get_BadDayCut_flag(self) -> Optional[PrimaryFlag]:
        """
        Get the primary flag for the bad day cut / binning if it exists.

        :return: PrimaryFlag for bad day cut or None if not found.
        """
        return self.get_primary_flag("Bad Day Cut (Binning)")

    def add_UncertaintyCut(
        self,
        flag: int = 0x2,
        max_value: float = 160,
        percent_cut: Optional[float] = None,
    ):
        """
        Add a cut for high measurement uncertainty.

        :param flag: Bitmask flag value.
        :param max_value: Maximum acceptable uncertainty.
        :param percent_cut: Optional percent of measurements cut from the SN light curve.
        """
        flag = PrimaryFlag(
            "high_uncertainty",
            flag,
            description=f"measurement has an uncertainty above {max_value}",
            percent_cut=percent_cut,
        )
        cut = Cut("Uncertainty Cut", primary_flag=flag, verbose=self.logger.verbose)
        self.add(cut)

    def add_UncertaintyEstimation(
        self, final_sigma_extra: float, temp_x2_max_value: float = 20
    ):
        """
        Add true uncertainties estimation with temporary PSF chi-square filtering.

        :param final_sigma_extra: Extra noise added in quadrature to the uncertainty column.
        :param temp_x2_max_value: Chi-square value used to pre-filter egregious outliers.
        """
        cut = Cut(
            "True Uncertainties Estimation",
            description=f"We attempt to account for an extra noise source in the data. We estimate the true typical uncertainty, derive the additional systematic uncertainty ({final_sigma_extra:0.2f}), and apply this extra noise to the uncertainty column. We also use a temporary, very high PSF chi-square cut value of {temp_x2_max_value} to eliminate the most egregious outliers from the data before estimating the true uncertainties.",
            verbose=self.logger.verbose,
        )
        self.add(cut)

    def add_ChiSquareCut(
        self,
        flag: int = 0x1,
        max_value: float = 10,
        percent_cut: Optional[float] = None,
    ):
        """
        Add a cut for high PSF chi-square values.

        :param flag: Bitmask flag value.
        :param max_value: Maximum acceptable chi-square value.
        :param percent_cut: Optional percent of measurements cut from the SN light curve.
        """
        flag = PrimaryFlag(
            "high_psf_chi_square",
            flag,
            description=f"measurement has a PSF chi-square above {max_value}",
            percent_cut=percent_cut,
        )
        cut = Cut("PSF Chi-Square Cut", primary_flag=flag, verbose=self.logger.verbose)
        self.add(cut)

    def add_ControlLightCurveCut(
        self,
        flag: int = 0x400000,
        questionable_flag: int = 0x80000,
        x2_max: float = 2.5,
        x2_flag: int = 0x100,
        snr_max: float = 3.0,
        snr_flag: int = 0x200,
        Nclip_max: int = 2,
        Nclip_flag: int = 0x400,
        Ngood_min: int = 4,
        Ngood_flag: int = 0x800,
        percent_cut: Optional[float] = None,
    ):
        """
        Add a cut based on control light curve statistics.

        :param flag: Primary flag for bad epochs.
        :param questionable_flag: Flag for questionable epochs.
        :param x2_max: Maximum chi-square threshold of the sigma-clipped epoch.
        :param x2_flag: Flag for high chi-square.
        :param snr_max: Maximum SNR threshold of the sigma-clipped epoch.
        :param snr_flag: Flag for high SNR.
        :param Nclip_max: Maximum number of clipped measurements.
        :param Nclip_flag: Flag for too many clipped measurements in the epoch.
        :param Ngood_min: Minimum required good measurements.
        :param Ngood_flag: Flag for too few good measurements in the epoch.
        :param percent_cut: Optional percent of measurements cut from the SN light curve.
        """
        primary_flag = PrimaryFlag(
            "bad_epoch",
            flag,
            description="control flux corresponding to this epoch is inconsistent with 0",
            percent_cut=percent_cut,
        )

        secondary_flags = [
            Flag(
                "high_control_x2",
                x2_flag,
                description=f"chi-square of control flux corresponding to this epoch is higher than {x2_max}",
            ),
            Flag(
                "high_control_snr",
                snr_flag,
                description=f"SNR of control flux corresponding to this epoch is higher than {snr_max}",
            ),
            Flag(
                "high_control_Nclip",
                Nclip_flag,
                description=f"number of clipped control measurements corresponding to this epoch is higher than {Nclip_max}",
            ),
            Flag(
                "low_control_Ngood",
                Ngood_flag,
                description=f"number of good control measurements corresponding to this epoch is lower than {Ngood_min}",
            ),
            Flag(
                "questionable_epoch",
                questionable_flag,
                description=f"control measurements in this epoch were not flagged, but has one or more were clipped",
            ),
        ]

        cut = Cut(
            "Control Light Curve Cut",
            description="For a given SN epoch, we can calculate the 3sigma-clipped average of the corresponding N control flux measurements falling within the same epoch. Given the expectation of control flux consistency with zero, the statistical properties accompanying the 3sigma-clipped average enable us to identify problematic epochs.",
            primary_flag=primary_flag,
            secondary_flags=secondary_flags,
            verbose=self.logger.verbose,
        )
        self.add(cut)

    def add_BadDayCut(
        self,
        flag: int = 0x800000,
        mjd_bin_size: float = 1.0,
        x2_max: float = 4.0,
        Nclip_max: int = 1,
        Ngood_min: int = 2,
        large_num_clipped_flag: int = 0x1000,
        small_num_unmasked_flag: int = 0x2000,
        percent_cut: Optional[float] = None,
    ):
        """
        Add a cut for identifying bad days using binning and statistics.

        :param flag: Primary flag for bad days.
        :param mjd_bin_size: Bin size in days.
        :param x2_max: Maximum chi-square allowed for a good bin.
        :param Nclip_max: Maximum allowed clipped points in a bin.
        :param Ngood_min: Minimum number of good points required in a bin.
        :param large_num_clipped_flag: Flag for excessive clipping.
        :param small_num_unmasked_flag: Flag for insufficient unmasked measurements.
        :param percent_cut: Optional percent of measurements cut from the SN light curve.
        """
        primary_flag = PrimaryFlag(
            "bad_day",
            flag,
            description=f"binned epoch has chi-square higher than {x2_max}, number of clipped measurements higher than {Nclip_max}, number of good measurements lower than {Ngood_min}",
            percent_cut=percent_cut,
        )

        secondary_flags = [
            Flag(
                "large_num_clipped_flag",
                large_num_clipped_flag,
                description=f"binned epoch had a nonzero number of measurements clipped during 3sigma-clipped average",
            ),
            Flag(
                "small_num_unmasked_flag",
                small_num_unmasked_flag,
                description=f"binned epoch had 2 or less unmasked measurements passed to 3sigma-clipped average",
            ),
        ]

        cut = Cut(
            "Bad Day Cut (Binning)",
            description=f"Bin the light curve into bins of {mjd_bin_size} day(s) and flag bad days",
            primary_flag=primary_flag,
            secondary_flags=secondary_flags,
            verbose=self.logger.verbose,
        )
        self.add(cut)

    def __str__(self):
        if not self._cuts:
            return "CutHistory: no cuts applied."
        return "-- CUT HISTORY --\n" + "\n".join(
            cut.__str__() for cut in self._cuts.values()
        )
