#!/usr/bin/env python

from copy import deepcopy
from typing import List, Optional

import numpy as np
import pandas as pd
from pdastro import AnotB, AorB, pdastrostatsclass
from utils import (
    BinnedColumnNames,
    CleanedColumnNames,
    ColumnNames,
    Coordinates,
    CustomLogger,
    Cut,
)


class BaseLightCurve(pdastrostatsclass):
    def __init__(
        self, control_index: int, colnames: ColumnNames, verbose: bool = False
    ):
        pdastrostatsclass.__init__(self)
        self.logger = CustomLogger(verbose=verbose)
        self.control_index = control_index
        self.colnames = colnames

    def set(self, t: pd.DataFrame):
        self.t = deepcopy(t)

    def get_filt_ix(self, filt: str):
        return self.ix_equal(self.colnames.filter, filt)

    def get_filt_lens(self):
        if self.t is None:
            raise RuntimeError("Table (self.t) cannot be None")

        total_len = len(self.t)
        filt_lens = {
            "o": len(self.get_filt_ix("o")),
            "c": len(self.get_filt_ix("c")),
        }
        return total_len, filt_lens

    def get_flags(self) -> int:
        if self.t is None or len(self.t) < 1:
            return 0
        np.bitwise_or.reduce(self.t[self.colnames.mask])

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

    def remove_flag(self, flag):
        self.t[self.colnames.mask] = np.bitwise_and(
            self.t[self.colnames.mask].astype(int), ~flag
        )

    def update_mask_column(self, flag, indices, remove_old=True):
        if remove_old:
            # remove any old flags of the same value
            self.remove_flag(flag)

        if len(indices) > 1:
            flag_arr = np.full(self.t.loc[indices, [self.colnames.mask]].shape, flag)
            self.t.loc[indices, self.colnames.mask] = np.bitwise_or(
                self.t.loc[indices, [self.colnames.mask]].astype(int), flag_arr
            )
        elif len(indices) == 1:
            self.t.loc[indices, self.colnames.mask] = (
                int(self.t.at[indices[0], self.colnames.mask]) | flag
            )

    def apply_cut(self, cut: Cut, indices: Optional[List[int]] = None):
        if not cut.can_apply_directly():
            raise RuntimeError(f"Cannot directly apply the following cut: {cut}")
        if not cut.column in self.t.columns:
            raise RuntimeError(
                f"No column name '{cut.column}' exists in light curve; cannot apply cut"
            )

        all_ix = self.getindices(indices)
        kept_ix = self.ix_inrange(
            colnames=[cut.column],
            lowlim=cut.min_value,
            uplim=cut.max_value,
            indices=all_ix,
        )
        cut_ix = AnotB(all_ix, kept_ix)

        self.update_mask_column(cut.flag, cut_ix)

        percent_cut = 100 * len(cut_ix) / len(all_ix)
        return percent_cut


class LightCurve(BaseLightCurve):
    def __init__(self, control_index: int, coords: Coordinates, verbose: bool = False):
        BaseLightCurve.__init__(
            self, control_index, CleanedColumnNames(), verbose=verbose
        )
        self.coords = coords

    def set(self, t: pd.DataFrame, flux2mag_sigmalimit: float = 3.0):
        super().set(t)
        self._preprocess(flux2mag_sigmalimit=flux2mag_sigmalimit)

    def _preprocess(self, flux2mag_sigmalimit=3.0):
        # create mask column
        self.t[self.colnames.mask] = 0

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

    def postprocess(self):
        # convert column names
        if not self.colnames.required.issubset(set(self.t.columns)):
            raise ValueError("Missing expected column")
        update_cols_dict = {
            self.colnames.mjd: "mjd",
            self.colnames.ra: "ra",
            self.colnames.dec: "dec",
            self.colnames.mag: "mag",
            self.colnames.dmag: "magerr",
            self.colnames.flux: "flux",
            self.colnames.dflux: "dflux",
            self.colnames.limiting_mag: "limiting_mag",
            self.colnames.filter: "filter",
        }
        self.t.rename(
            columns=update_cols_dict,
            inplace=True,
        )
        self.colnames.update_many(update_cols_dict)

        # drop unnecessary columns
        drop_columns = list(set(self.t.columns.values) - self.colnames.required)
        self.t.drop(columns=drop_columns, inplace=True)

        # replace 'o' -> 'atlaso' and 'c' -> 'atlasc'
        c_ix = self.ix_equal(self.colnames.filter, "c")
        o_ix = self.ix_equal(self.colnames.filter, "o")
        self.t.loc[c_ix, self.colnames.filter] = "atlasc"
        self.t.loc[o_ix, self.colnames.filter] = "atlaso"

        # add optional columns
        self.colnames.add_many({"magsys": "magsys", "origin": "origin"})
        self.t[self.colnames.magsys] = "ab"
        self.t[self.colnames.origin] = "fp"


class BinnedLightCurve(BaseLightCurve):
    def __init__(self, control_index: int, verbose: bool = False):
        super().__init__(control_index, BinnedColumnNames(), verbose=verbose)
