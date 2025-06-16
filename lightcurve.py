#!/usr/bin/env python

from copy import deepcopy
from typing import Dict, List, Optional, Self

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
        self,
        control_index: int,
        coords: Coordinates,
        colnames: ColumnNames,
        filt: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.logger = CustomLogger(verbose=verbose)
        self.control_index = control_index
        self.coords = coords
        self.filt = filt
        self.colnames = colnames

    def set(self, t: pd.DataFrame, indices: Optional[List[int]] = None, **kwargs):
        if t is None:
            self.t = pd.DataFrame()
            return
        if t.empty:
            self.t = pd.DataFrame(columns=t.columns)
            return

        if indices is not None:
            if len(indices) == 0:
                self.t = pd.DataFrame(columns=t.columns)
            else:
                self.t = deepcopy(t.iloc[indices])
        else:
            self.t = deepcopy(t)

        self._preprocess(**kwargs)

    @property
    def default_mjd_colname(self):
        return self.colnames.mjd

    def _preprocess(self, **kwargs):
        if self.t is None or self.t.empty:
            return

        # calculate SNR
        self.calculate_snr_col()

    def get_filt_ix(self, filt: str):
        if self.t is None or self.t.empty:
            return []

        if self.filt is not None:
            return self.getindices()

        if self.colnames.filter not in self.t.columns:
            raise RuntimeError(f"Filter column '{self.colnames.filter}' not found")

        return self.ix_equal(self.colnames.filter, filt)

    def get_filt_lens(self) -> tuple[int, int]:
        if self.t is None:
            raise RuntimeError("Table (self.t) cannot be None")

        return len(self.get_filt_ix("o")), len(self.get_filt_ix("c"))

    def get_flags(self) -> int:
        if self.t is None or self.t.empty or self.colnames.mask not in self.t.columns:
            return 0

        return np.bitwise_or.reduce(self.t[self.colnames.mask])

    def get_good_indices(self, flag: Optional[int] = None) -> List[int]:
        if self.t is None or self.t.empty:
            return []

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
        if self.t is None or self.t.empty:
            return []

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
        if self.t is None or self.t.empty or self.colnames.mask not in self.t.columns:
            return

        self.t[self.colnames.mask] = np.bitwise_and(
            self.t[self.colnames.mask].astype(int), ~flag
        )

    def update_mask_column(self, flag, indices, remove_old=True):
        if self.t is None or self.t.empty:
            return

        if self.colnames.mask not in self.t.columns:
            self.t[self.colnames.mask] = 0

        if remove_old:
            # remove any old flags of the same value
            self.remove_flag(flag)

        if len(indices) > 1:
            flag_arr = np.full(len(indices), flag)
            self.t.loc[indices, self.colnames.mask] = np.bitwise_or(
                self.t.loc[indices, self.colnames.mask].astype(int), flag_arr
            )
        elif len(indices) == 1:
            self.t.at[indices[0], self.colnames.mask] = (
                int(self.t.at[indices[0], self.colnames.mask]) | flag
            )

    def apply_cut(self, cut: Cut, indices: Optional[List[int]] = None):
        if self.t is None or self.t.empty:
            return

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

    def get_stdev_flux(self, indices=None):
        if self.t is None or self.t.empty:
            return np.nan

        self.calcaverage_sigmacutloop(
            self.colnames.flux, indices=indices, Nsigma=3.0, median_firstiteration=True
        )
        return self.statparams["stdev"]

    def get_median_dflux(self, indices=None):
        if self.t is None or self.t.empty or len(indices) < 1:
            return np.nan
        if indices is None:
            indices = self.getindices()
        return np.nanmedian(self.t.loc[indices, self.colnames.dflux])

    def calculate_snr_col(self):
        if self.t is None or self.t.empty:
            return

        # replace infs with NaNs
        self.logger.info("Replacing infs with NaNs")
        self.t.replace([np.inf, -np.inf], np.nan, inplace=True)

        # calculate flux/dflux
        self.logger.info(f"Calculating flux/dflux in for '{self.colnames.snr}' column")
        self.t[self.colnames.snr] = (
            self.t[self.colnames.flux] / self.t[self.colnames.dflux]
        )

    def merge(self, other: Self) -> Self:
        """
        Merge this LightCurve with another one and return a new LightCurve instance
        with the combined and sorted data. Used for merging two light curves with
        different filters together.
        """
        if not isinstance(other, BaseLightCurve):
            raise TypeError("Can only merge with another BaseLightCurve instance.")
        if self.control_index != other.control_index:
            raise ValueError(
                f"Control indices ({self.control_index} and {other.control_index}) do not match"
            )
        # if self.colnames.required != other.colnames.required:
        #     raise ValueError("Column name configurations do not match")

        merged_filt = self.filt
        if self.filt is None or other.filt is None or self.filt != other.filt:
            merged_filt = None

        # combine tables
        merged_t = pd.concat(
            [self.t or pd.DataFrame(), other.t or pd.DataFrame()], ignore_index=True
        )
        if not merged_t.empty:
            merged_t.sort_values(
                by=[self.default_mjd_colname], ignore_index=True, inplace=True
            )

        # create new lc
        new_lc = self.__class__(
            self.control_index,
            self.coords,
            self.colnames,
            filt=merged_filt,
            verbose=self.logger.verbose,
        )
        new_lc.t = deepcopy(merged_t)  # skip preprocessing
        new_lc.calculate_snr_col()
        return new_lc

    def split_by_filt(self) -> tuple[Self, Self]:
        if self.colnames.filter not in self.t.columns:
            raise RuntimeError(
                f"Filter column '{self.colnames.filter}' not found in data"
            )

        if self.filt is not None:
            self.logger.warning("Splitting a single-filter light curve by filter")

        split_lcs = {}
        for filt in ["o", "c"]:
            new_lc = self.__class__(
                control_index=self.control_index,
                coords=self.coords,
                colnames=self.colnames,
                filt=filt,
                verbose=self.logger.verbose,
            )
            new_lc.set(self.t, indices=self.get_filt_ix(filt))
            new_lc.calculate_snr_col()
            split_lcs[filt] = new_lc

        return split_lcs["o"], split_lcs["c"]


class LightCurve(BaseLightCurve):
    def __init__(
        self,
        control_index: int,
        coords: Coordinates,
        filt: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(
            control_index,
            coords,
            CleanedColumnNames(),
            filt=filt,
            verbose=verbose,
        )

    def set(
        self,
        t: pd.DataFrame,
        indices: Optional[List[int]] = None,
        flux2mag_sigmalimit: float = 3.0,
    ):
        super().set(t, indices=indices, flux2mag_sigmalimit=flux2mag_sigmalimit)

    def _preprocess(self, flux2mag_sigmalimit=3.0):
        """
        Prepare the raw ATLAS light curve for cleaning
        (add mask column, sort by MJD, remove rows with duJy=0 or uJy=NaN, overwrite ATLAS magnitudes with our own)
        """
        if self.t is None or self.t.empty:
            return

        if self.colnames.mask not in self.t.columns:
            # create mask column
            self.t[self.colnames.mask] = 0

        # sort by mjd
        self.t = self.t.sort_values(by=[self.default_mjd_colname], ignore_index=True)

        # remove rows with duJy=0 or uJy=nan
        dflux_zero_ix = self.ix_inrange(colnames=self.colnames.dflux, lowlim=0, uplim=0)
        flux_nan_ix = self.ix_is_null(colnames=self.colnames.flux)
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
            self.colnames.flux,
            self.colnames.dflux,
            self.colnames.mag,
            self.colnames.dmag,
            zpt=23.9,
            upperlim_Nsigma=flux2mag_sigmalimit,
        )
        # drop extra SNR column
        if "__tmp_SN" in self.t.columns:
            self.t.drop(columns=["__tmp_SN"], inplace=True)

        # calculate SNR
        self.calculate_snr_col()

    def add_noise_to_dflux(self, sigma_extra):
        """
        Add sigma_extra to the dflux column in quadrature.
        """
        # add in quadrature to dflux column
        self.t[self.colnames.dflux] = np.sqrt(
            self.t[self.colnames.dflux] ** 2 + sigma_extra**2
        )

        # store the sigma_extra we added in a new column
        self.colnames.update("dflux_offset", "dflux_offset_in_quadrature")
        self.t[self.colnames.dflux_offset] = np.full(len(self.t), sigma_extra)

        # recalculate SNR
        self.calculate_snr_col()

    def remove_noise_from_dflux(self):
        """
        Remove sigma_extra that was previously added in quadrature to the dflux column.
        Assumes sigma_extra is stored in self.colnames.dflux_offset as a constant value.
        """
        # get sigma_extra from the offset column
        sigma_extra = self.t[self.colnames.dflux_offset].iloc[0]

        # remove in quadrature, ensuring no negative values
        corrected_squared = self.t[self.colnames.dflux] ** 2 - sigma_extra**2
        corrected_squared[corrected_squared < 0] = 0  # avoid NaNs from sqrt of negative
        self.t[self.colnames.dflux] = np.sqrt(corrected_squared)

        # clean up columns
        self.t.drop(columns=[self.colnames.dflux_offset], inplace=True)
        self.colnames.remove("dflux_offset")

        # recalculate SNR
        self.calculate_snr_col()

    def postprocess(self):
        """
        Prepare the cleaned light curve for SkyPortal by handling desired columnss
        """
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
        self.t.loc[self.get_filt_ix("c"), self.colnames.filter] = "atlasc"
        self.t.loc[self.get_filt_ix("o"), self.colnames.filter] = "atlaso"

        # add optional columns
        self.colnames.add_many({"magsys": "magsys", "origin": "origin"})
        self.t[self.colnames.magsys] = "ab"
        self.t[self.colnames.origin] = "fp"


class BinnedLightCurve(BaseLightCurve):
    def __init__(
        self,
        control_index: int,
        coords: Coordinates,
        filt: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(
            control_index, coords, BinnedColumnNames(), filt=filt, verbose=verbose
        )

    @property
    def default_mjd_colname(self):
        return self.colnames.mjdbin


class BaseTransient:
    def __init__(self, filt: Optional[str] = None, verbose: bool = False):
        self.filt = filt
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

    def add(self, lc: BaseLightCurve, deep: bool = True):
        if lc.control_index in self.lcs:
            self.logger.warning(
                f"Control index {lc.control_index} already exists; overwriting..."
            )
        self.lcs[lc.control_index] = deepcopy(lc) if deep else lc

    def iterator(self):
        for i in self.lcs:
            yield self.lcs[i]

    def apply_cut(self, cut: Cut):
        if not cut.can_apply_directly():
            raise ValueError(f"Cannot apply this cut directly: {cut.name()}")

        for i in self.lcs:
            self.get(i).apply_cut(cut)

    def merge(self, other: Self) -> Self:
        if not isinstance(other, BaseTransient):
            raise TypeError("Can only merge with another BaseTransient instance.")

        merged_filt = self.filt
        if self.filt is None or other.filt is None or self.filt != other.filt:
            merged_filt = None

        new_transient = self.__class__(filt=merged_filt, verbose=self.logger.verbose)
        all_indices = set(self.lcs.keys()).union(set(other.lcs.keys()))

        for idx in all_indices:
            if idx in self.lcs and idx in other.lcs:
                new_transient.add(self.lcs[idx].merge(other.lcs[idx]))
            elif idx in self.lcs:
                new_transient.add(self.lcs[idx])
            else:
                new_transient.add(other.lcs[idx])

        return new_transient

    def split_by_filt(self) -> tuple[Self, Self]:
        # create two new BaseTransient instances for filters 'o' and 'c'
        transient_o = self.__class__(filt="o", verbose=self.logger.verbose)
        transient_c = self.__class__(filt="c", verbose=self.logger.verbose)

        # iterate over all BaseLightCurves in this transient
        for lc in self.iterator():
            # split the BaseLightCurve into two, one for each filter
            lc_o, lc_c = lc.split_by_filt()

            # add the split light curves to the corresponding BaseTransient objects
            transient_o.add(lc_o, deep=False)
            transient_c.add(lc_c, deep=False)

        return transient_o, transient_c


class Transient(BaseTransient):
    def __init__(self, filt: Optional[str] = None, verbose: bool = False):
        super().__init__(filt=filt, verbose=verbose)
        self.lcs: Dict[int, LightCurve] = {}

    def get(self, control_index: int) -> LightCurve:
        return super().get(control_index)

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

    def postprocess(self):
        if self.filt is not None:
            raise RuntimeError(
                f"Filter '{self.filt}' should be None for postprocessing"
            )

        for i in self.lc_indices:
            self.get(i).postprocess()


class BinnedTransient(BaseTransient):
    def __init__(self, filt: Optional[str] = None, verbose: bool = False):
        super().__init__(filt=filt, verbose=verbose)
        self.lcs: Dict[int, BinnedLightCurve] = {}

    def get(self, control_index: int) -> BinnedLightCurve:
        return super().get(control_index)
