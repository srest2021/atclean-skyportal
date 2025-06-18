#!/usr/bin/env python

from abc import ABC, abstractmethod
import bisect
from configparser import ConfigParser
import configparser
from functools import reduce
from getpass import getpass
from typing import Callable, Dict, Any, List, Optional, Self, Set, Tuple, Type
import re, json, requests, time, sys, io, os
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from collections import OrderedDict, namedtuple
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path


class CustomLogger:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def info(self, text: str, prefix="", newline: bool = False):
        newline_part = "\n" if newline else ""
        if self.verbose:
            print(f"{newline_part}{prefix}{text}")

    def warning(self, text: str, newline: bool = False):
        self.info(text, prefix="WARNING: ", newline=newline)

    def error(self, text: str, newline: bool = False):
        self.info(text, prefix="ERROR: ", newline=newline)

    def success(self, text: str = "Success", newline: bool = False):
        self.info(text, newline=newline)


def hexstring_to_int(hexstring):
    return int(hexstring, 16)


def combine_flags(flags: List[int]) -> int:
    return reduce(lambda x, y: x | y, flags, 0)


def new_row(t: Optional[pd.DataFrame], d: Optional[Dict] = None):
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
    def __init__(self, angle: str | Angle):
        self.angle: Angle = (
            self._parse_angle(angle) if isinstance(angle, str) else angle
        )

    @abstractmethod
    def _parse_angle(self, string: str) -> Angle:
        """Abstract method to be implemented by subclasses to parse angles."""
        pass


class RA(BaseAngle):
    def _parse_angle(self, string: str) -> Angle:
        """Parse RA angle, using hours if ':' is present, degrees otherwise."""
        s = re.compile(":")
        if isinstance(string, str) and s.search(string):
            return Angle(string, u.hour)
        else:
            return Angle(string, u.degree)


class Dec(BaseAngle):
    def _parse_angle(self, string: str) -> Angle:
        """Parse Dec angle, always using degrees."""
        return Angle(string, u.degree)


class Coordinates:
    def __init__(self, ra: str | Angle | RA, dec: str | Angle | Dec):
        self.ra: RA = RA(ra) if not isinstance(ra, RA) else ra
        self.dec: Dec = Dec(dec) if not isinstance(dec, Dec) else dec

    def get_RA_str(self) -> str:
        return f"{self.ra.angle.degree:0.14f}"

    def get_Dec_str(self) -> str:
        return f"{self.dec.angle.degree:0.14f}"

    def get_distance(self, other: Self) -> Angle:
        c1 = SkyCoord(self.ra.angle, self.dec.angle, frame="fk5")
        c2 = SkyCoord(other.ra.angle, other.dec.angle, frame="fk5")
        return c1.separation(c2)

    def __str__(self):
        return f"RA {self.get_RA_str()}, Dec {self.get_Dec_str()}"


def parse_comma_separated_string(string: Optional[str]):
    if string is None:
        return None

    try:
        return [item.strip() for item in string.split(",")]
    except Exception as e:
        raise RuntimeError(
            f"Could not parse comma-separated string: {string}" f"\nERROR: {str(e)}"
        )


def parse_arg_coords(arg_coords: Optional[str]) -> Optional[Coordinates]:
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
    def __init__(
        self,
        required_colnames: Optional[Dict[str, str]] = None,
        optional_colnames: Optional[Dict[str, str]] = None,
        verbose: bool = False,
    ):
        self.logger = CustomLogger(verbose=verbose)

        self._required_colnames = (
            required_colnames if required_colnames is not None else {}
        )
        self._optional_colnames = (
            optional_colnames if optional_colnames is not None else {}
        )

    @property
    def required(self) -> Set:
        return set(self._required_colnames.values())

    @property
    def optional(self) -> Set:
        return set(self._optional_colnames.values())

    @property
    def all(self) -> Set:
        return self.required.union(self.optional)

    def add(
        self, key: str, name: str, is_required: bool = False, overwrite: bool = False
    ):
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Column key must be a non-empty string.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Column name for key '{key}' must be a non-empty string.")

        # if key exists but the name is different, raise an error
        existing_name = self._required_colnames.get(key) or self._optional_colnames.get(
            key
        )
        if existing_name and existing_name != name and not overwrite:
            raise RuntimeError(
                f"Column key '{key}' is already defined with a different name '{existing_name}'."
            )

        if is_required:
            self._required_colnames[key] = name
        else:
            self._optional_colnames[key] = name

    def add_many(self, coldict: Dict[str, str], is_required: bool = False):
        for key, name in coldict.items():
            self.add(key, name, is_required=is_required)

    def update(self, key: str, name: str, is_required: bool = False):
        if is_required:
            if key not in self._required_colnames:
                raise RuntimeError(
                    f"Cannot update non-existing required column name {key} with '{name}'"
                )
            self._required_colnames[key] = name
        else:
            if key not in self._optional_colnames:
                raise RuntimeError(
                    f"Cannot update non-existing optional column name {key} with '{name}'"
                )
            self._optional_colnames[key] = name

    def update_many(self, coldict: Dict[str, str], is_required: bool = False):
        for key, name in coldict.items():
            self.update(key, name, is_required=is_required)

    def remove(self, key: str):
        if key in self._required_colnames:
            del self._required_colnames[key]
        if key in self._optional_colnames:
            del self._optional_colnames[key]

    def has(self, name: str):
        return name in self._required_colnames or name in self._optional_colnames

    def __getattr__(self, name: str) -> str | None:
        """
        Dynamic access to column names, e.g., obj.mjd or obj.chisquare.
        """
        if name in self._required_colnames:
            return self._required_colnames[name]
        if name in self._optional_colnames:
            return self._optional_colnames[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __str__(self) -> str:
        """Readable string representation of all column names."""
        skip_colnames = ["mjdbin", "fdf", "mask"]

        lines = ["-- Required Columns --"]
        for k, v in self._required_colnames.items():
            if k in skip_colnames:
                continue
            lines.append(f"{k}: {v}")

        lines.append("-- Optional Columns --")
        for k, v in self._optional_colnames.items():
            lines.append(f"{k}: {v}")

        return "\n".join(lines)


class CleanedColumnNames(ColumnNames):
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


def nan_if_none(x):
    return x if x is not None else np.nan


class StatParams:
    def __init__(self, statparams: Dict[str, int | float | None]):
        statparams = deepcopy(statparams)
        self.mean: float = nan_if_none(statparams["mean"])
        self.mean_err: float = nan_if_none(statparams["mean_err"])
        self.stdev: float = nan_if_none(statparams["stdev"])
        self.X2norm: float = nan_if_none(statparams["X2norm"])
        self.Nclip: int | float = nan_if_none(statparams["Nclip"])
        self.Ngood: int | float = nan_if_none(statparams["Ngood"])
        # self.Nexcluded: int | float = nan_if_none(statparams["Nexcluded"])
        self.ix_good: List[int] = list(statparams["ix_good"])
        self.ix_clip: List[int] = list(statparams["ix_clip"])
