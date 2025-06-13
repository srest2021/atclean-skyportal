#!/usr/bin/env python

from copy import deepcopy
from typing import List, Optional

import numpy as np
import pandas as pd
from pdastro import AorB, pdastrostatsclass
from utils import ColumnNames, Coordinates, CustomLogger


class LightCurve(pdastrostatsclass):
    def __init__(self, control_index: int, coords: Coordinates, verbose: bool = False):
        pdastrostatsclass.__init__(self)

        self.logger = CustomLogger(verbose=verbose)
        self.colnames = ColumnNames()

        self.control_index = control_index
        self.coords = coords

    def set_and_preprocess_df(self, t: pd.DataFrame, flux2mag_sigmalimit: float = 3.0):
        self.t = deepcopy(t)
        self._preprocess(flux2mag_sigmalimit=flux2mag_sigmalimit)
        self._convert_to_skyportal_format()

    def _convert_to_skyportal_format(self):
        # convert column names
        if not self.colnames.required.issubset(set(self.t.columns)):
            raise ValueError("Missing expected column")
        self.t.rename(
            columns={
                "RA": self.colnames.ra,
                "Dec": self.colnames.dec,
                "m": self.colnames.mag,
                "dm": self.colnames.magerr,
                "mag5sig": self.colnames.limiting_mag,
                "F": self.colnames.filter,
            },
            inplace=True,
        )

        # drop unnecessary columns
        drop_columns = list(set(self.t.columns.values) - self.colnames.required)
        self.t.drop(columns=drop_columns, inplace=True)

        # replace 'o' -> 'atlaso' and 'c' -> 'atlasc'
        c_ix = self.ix_equal(self.colnames.filter, "c")
        o_ix = self.ix_equal(self.colnames.filter, "o")
        self.t.loc[c_ix, self.colnames.filter] = "atlasc"
        self.t.loc[o_ix, self.colnames.filter] = "atlaso"

        # add optional columns
        self.t[self.colnames.magsys] = "ab"
        self.t[self.colnames.origin] = "fp"

    def _preprocess(self, flux2mag_sigmalimit=3.0):
        # sort by mjd
        self.t = self.t.sort_values(by=["MJD"], ignore_index=True)

        # remove rows with duJy=0 or uJy=nan
        dflux_zero_ix = self.ix_inrange(colnames="duJy", lowlim=0, uplim=0)
        flux_nan_ix = self.ix_is_null(colnames="uJy")
        self.logger.info(
            f'Deleting {len(dflux_zero_ix) + len(flux_nan_ix)} rows with "duJy"==0 or "uJy"==NaN'
        )
        if len(AorB(dflux_zero_ix, flux_nan_ix)) > 0:
            self.t = self.t.drop(AorB(dflux_zero_ix, flux_nan_ix))

        # convert flux to magnitude
        self.logger.info(
            "Converting flux to magnitude (and overwriting original ATLAS 'm' and 'dm' columns)"
        )
        self.flux2mag(
            "uJy", "duJy", "m", "dm", zpt=23.9, upperlim_Nsigma=flux2mag_sigmalimit
        )
        # drop extra SNR column
        if "__tmp_SN" in self.t.columns:
            self.t.drop(columns=["__tmp_SN"], inplace=True)

    def get_filt_lens(self):
        if self.t is None:
            raise RuntimeError("Table (self.t) cannot be None")

        total_len = len(self.t)
        filt_lens = {
            "o": len(np.where(self.t["F"] == "o")[0]),
            "c": len(np.where(self.t["F"] == "c")[0]),
        }
        return total_len, filt_lens

    def get_good_indices(self, flag: Optional[int] = None) -> List[int]:
        # if flag is 0, return all indices
        if flag == 0:
            return self.getindices()

        if flag is None:  # if no flag is given
            # check if mask column exists
            if not self.colnames.mask in self.t.columns:
                return self.getindices()

            # return all unmasked indices
            flag = self.get_flags()
        return self.ix_unmasked(self.colnames.mask, maskval=flag)

    def get_bad_indices(self, flag: Optional[int] = None) -> List[int]:
        # if flag is 0, return no indices
        if flag == 0:
            return []

        if flag is None:  # if no flag is given
            # check if mask column exists
            if not self.colnames.mask in self.t.columns:
                return self.getindices()

            # return all masked indices
            flag = self.get_flags()
        return self.ix_masked(self.colnames.mask, maskval=flag)
