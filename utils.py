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
from collections import OrderedDict
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path


class CustomLogger:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def info(self, text: str, prefix=""):
        if self.verbose:
            print(f"{prefix}{text}")

    def warning(self, text: str):
        self.info(text, prefix="WARNING: ")

    def error(self, text: str):
        self.info(text, prefix="ERROR: ")

    def success(self, text: str = "Success"):
        self.info(text)


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
    def __init__(self, ra: str | Angle, dec: str | Angle):
        self.ra: RA = RA(ra)
        self.dec: Dec = Dec(dec)

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


class ColumnNames:
    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger(verbose=verbose)

        self._required_colnames = {
            "mjd": "mjd",
            "ra": "ra",
            "dec": "dec",
            "mag": "mag",
            "magerr": "magerr",
            "limiting_mag": "limiting_mag",
            "filter": "filter",
        }

        self._optional_colnames = {"magsys": "magsys", "origin": "origin"}

    @property
    def required(self) -> Set:
        return set(self._required_colnames.values())

    @property
    def optional(self) -> Set:
        return set(self._optional_colnames.values())

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
