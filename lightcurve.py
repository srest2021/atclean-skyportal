#!/usr/bin/env python

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pdastro import AandB, AnotB, AorB, pdastrostatsclass
from utils import (
    BinnedColumnNames,
    CleanedColumnNames,
    ColumnNames,
    Coordinates,
    CustomLogger,
    Cut,
    UncertaintyEstimation,
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

    def get_stdev_flux(self, indices=None):
        self.calcaverage_sigmacutloop(
            self.colnames.flux, indices=indices, Nsigma=3.0, median_firstiteration=True
        )
        return self.statparams["stdev"]

    def get_median_dflux(self, indices=None):
        if indices is None:
            indices = self.getindices()
        return np.nanmedian(self.t.loc[indices, self.colnames.dflux])


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
        if self.colnames.mask not in self.t:
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

    def add_noise_to_dflux(self, sigma_extra):
        # new_dflux_colname = f"{self.colnames.dflux}_new"
        self.t[self.colnames.dflux] = np.sqrt(
            self.t[self.colnames.dflux] * self.t[self.colnames.dflux] + sigma_extra**2
        )
        # self.colnames.update("dflux_new", new_dflux_colname)
        # self.calculate_fdf_column()

    def postprocess(self):
        # convert column names
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
        desired_cols = set(update_cols_dict.values())
        self.t.rename(
            columns=update_cols_dict,
            inplace=True,
        )
        self.colnames.update_many(update_cols_dict)
        if not desired_cols.issubset(set(self.t.columns)):
            raise ValueError("Missing expected column")

        # drop unnecessary columns
        drop_columns = list(set(self.t.columns.values) - desired_cols)
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


class BaseTransient:
    def __init__(self, verbose: bool = False):
        self.lcs: Dict[int, BaseLightCurve] = {}
        self.logger = CustomLogger(verbose=verbose)

    def get(self, control_index: int) -> BaseLightCurve:
        if control_index not in self.lcs:
            raise ValueError(
                f"Light curve with control index {control_index} does not exist"
            )
        return self.lcs[control_index]

    def get_sn(self) -> BaseLightCurve:
        return self.get(0)

    @property
    def colnames(self):
        return self.get_sn().colnames

    @property
    def num_controls(self):
        return len(self.lcs)

    @property
    def lc_indices(self) -> List[int]:
        return list(self.lcs.keys())

    @property
    def control_lc_indices(self) -> List[int]:
        return [i for i in self.lc_indices if i != 0]

    def add(self, lc: BaseLightCurve):
        if lc.control_index in self.lcs:
            self.logger.warning(
                f"Control index {lc.control_index} already exists; overwriting..."
            )
        self.lcs[lc.control_index] = deepcopy(lc)

    def iterator(self):
        for i in self.lcs:
            yield self.lcs[i]

    def apply_cut(self, cut: Cut):
        if not cut.can_apply_directly():
            raise ValueError(f"Cannot apply this cut directly: {cut.name()}")

        for i in self.lcs:
            self.get(i).apply_cut(cut)


class Transient(BaseTransient):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.lcs: Dict[int, LightCurve] = {}

    def match_control_mjds(self):
        """Make sure all control light curve MJDs exactly match SN MJDs"""
        self.logger.info("Making sure SN and control light curve MJDs match up exactly")
        if self.num_controls == 0:
            return

        self.get_sn().t.sort_values(
            by=[self.colnames.mjd], ignore_index=True, inplace=True
        )
        sn_sorted_mjd = self.get_sn().t[self.colnames.mjd].to_numpy()

        for i in self.lcs:
            # sort by MJD
            self.get(i).t.sort_values(
                by=[self.colnames.mjd], ignore_index=True, inplace=True
            )
            control_sorted_mjd = self.get(i).t[self.colnames.mjd].to_numpy()

            if (len(sn_sorted_mjd) != len(control_sorted_mjd)) or not np.array_equal(
                sn_sorted_mjd, control_sorted_mjd
            ):
                self.logger.warning(
                    f"MJDs out of agreement for control light curve {i}; fixing...",
                )

                only_sn_mjd = AnotB(sn_sorted_mjd, control_sorted_mjd)
                only_control_mjd = AnotB(control_sorted_mjd, sn_sorted_mjd)

                # for the MJDs only in SN, add row with that MJD to control light curve,
                # with all values of other columns NaN
                if len(only_sn_mjd) > 0:
                    for mjd in only_sn_mjd:
                        self.get(i).newrow(
                            {
                                self.colnames.mjd: mjd,
                                self.colnames.mask: 0,
                            }
                        )

                # remove indices of rows in control light curve for which there is no MJD in the SN lc
                if len(only_control_mjd) > 0:
                    ix_to_skip = []
                    for mjd in only_control_mjd:
                        matching_ix = self.get(i).ix_equal(self.colnames.mjd, mjd)
                        if len(matching_ix) != 1:
                            raise RuntimeError(
                                f"Couldn't find MJD={mjd} in MJD column, but should be there!"
                            )
                        ix_to_skip.extend(matching_ix)
                    ix = AnotB(self.get(i).getindices(), ix_to_skip)
                else:
                    ix = self.get(i).getindices()

                # sort again
                sorted_ix = self.get(i).ix_sort_by_cols(self.colnames.mjd, indices=ix)
                self.get(i).t = self.get(i).t.loc[sorted_ix]

            self.get(i).t.reset_index(drop=True, inplace=True)

        self.logger.success()

    def get_uncert_est_stats(self, cut: UncertaintyEstimation) -> pd.DataFrame:
        """Get the median uncertainty, standard deviation of the flux,
        and sigma_extra for each control light curve."""

        def get_sigma_extra(median_dflux, stdev):
            diff = stdev**2 - median_dflux**2
            return max(0, np.sqrt(diff)) if diff > 0 else 0

        stats = pd.DataFrame(
            columns=["control_index", "median_dflux", "stdev", "sigma_extra"]
        )
        stats["control_index"] = self.control_lc_indices
        stats.set_index("control_index", inplace=True)

        for i in self.control_lc_indices:
            dflux_clean_ix = self.get(i).ix_unmasked(
                self.colnames.mask, maskval=cut.uncert_cut_flag
            )
            x2_clean_ix = self.get(i).ix_inrange(
                colnames=[self.colnames.chisquare],
                uplim=cut.temp_x2_max_value,
                exclude_uplim=True,
            )
            clean_ix = AandB(dflux_clean_ix, x2_clean_ix)

            median_dflux = self.get(i).get_median_dflux(indices=clean_ix)
            stdev_flux = self.get(i).get_stdev_flux(indices=clean_ix)

            if stdev_flux is None:
                self.logger.warning(
                    f"Could not get flux std dev using clean indices; retrying without preliminary chi-square cut of {cut.temp_x2_max_value}"
                )
                stdev_flux = self.get(i).get_stdev_flux(indices=dflux_clean_ix)
                if stdev_flux is None:
                    self.logger.warning(
                        "Could not get flux std dev using clean indices; retrying with all indices"
                    )
                    stdev_flux = self.get(i).get_stdev_flux()

            sigma_extra = get_sigma_extra(median_dflux, stdev_flux)

            stats.loc[i, "median_dflux"] = median_dflux
            stats.loc[i, "stdev"] = stdev_flux
            stats.loc[i, "sigma_extra"] = sigma_extra

        return stats

    def add_noise_to_dflux(self, sigma_extra):
        for i in self.lc_indices:
            self.get(i).add_noise_to_dflux(sigma_extra)


class BinnedTransient(BaseTransient):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.lcs: Dict[int, BinnedLightCurve] = {}
