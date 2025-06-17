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
    ControlLightCurveCut,
    Coordinates,
    CustomLogger,
    Cut,
    UncertaintyEstimation,
    combine_flags,
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

    def set(
        self,
        t: pd.DataFrame,
        indices: Optional[List[int]] = None,
        deep: bool = True,
    ):
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
                self.t = t.iloc[indices].copy(deep=deep)
        else:
            self.t = t.copy(deep=deep)

    @property
    def default_mjd_colname(self):
        return self.colnames.mjd

    def preprocess(self, **kwargs):
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
        """
        Return the list of indices corresponding to "good" (unmasked) rows.

        A row is considered "good" if its value in the mask column does not contain
        the specified `flag`. If `flag` is 0 or None, all unmasked rows will be returned.
        """
        if self.t is None or self.t.empty:
            return []

        if not self.colnames.mask in self.t.columns:
            return self.getindices()

        if flag == 0:
            flag = None

        # note: if flag is None, this will just return all unmasked indices
        return self.ix_unmasked(self.colnames.mask, maskval=flag)

    def get_bad_indices(self, flag: Optional[int] = None) -> List[int]:
        """
        Return the list of indices corresponding to "bad" (masked) rows.

        A row is considered "bad" if its value in the mask column contains
        the specified `flag`. If `flag` is 0 or None, all masked rows will be returned.
        """
        if self.t is None or self.t.empty:
            return []

        if not self.colnames.mask in self.t.columns:
            return self.getindices()

        if flag == 0:
            flag = None

        # note: if flag is None, this will just return all masked indices
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

    def apply_cut(
        self,
        column: str,
        flag: int,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        indices: Optional[List[int]] = None,
    ):
        if self.t is None or self.t.empty:
            return

        if not column in self.t.columns:
            raise RuntimeError(
                f"Column '{column}' not found in light curve; cannot apply cut"
            )

        all_ix = self.getindices(indices)
        # kept_ix = self.ix_inrange(
        #     colnames=[column],
        #     lowlim=min_value,
        #     uplim=max_value,
        #     indices=all_ix,
        # )
        # cut_ix = AnotB(all_ix, kept_ix)
        cut_ix = self.ix_outrange(
            colnames=[column], lowlim=min_value, uplim=max_value, indices=all_ix
        )

        self.update_mask_column(flag, cut_ix)

    def get_stdev_flux(self, indices=None):
        if self.t is None or self.t.empty:
            return np.nan

        self.calcaverage_sigmacutloop(
            self.colnames.flux, indices=indices, Nsigma=3.0, median_firstiteration=True
        )
        return self.statparams["stdev"]

    def get_median_dflux(self, indices=None):
        if self.t is None or self.t.empty or (indices is not None and len(indices) < 1):
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
        return new_lc

    def split_by_filt(self) -> tuple[Self, Self]:
        """
        Create two new class instances for filters 'o' and 'c'.
        Underlying dataframes are deepcopied and independent of self.
        """
        if self.colnames.filter not in self.t.columns:
            raise RuntimeError(
                f"Filter column '{self.colnames.filter}' not found in data"
            )

        if self.filt is not None:
            self.logger.warning("Splitting a single-filter light curve by filter")

        split_lcs = {}
        for filt in ["o", "c"]:
            new_lc = self.__class__(
                self.control_index,
                self.coords,
                filt=filt,
                verbose=self.logger.verbose,
            )
            new_lc.set(self.t, indices=self.get_filt_ix(filt))
            split_lcs[filt] = new_lc

        return split_lcs["o"], split_lcs["c"]

    def __str__(self):
        if self.t is None:
            return None
        return self.t.to_string()


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

    def preprocess(self, flux2mag_sigmalimit=3.0):
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

    def flag_by_control_stats(
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
    ):
        # flag SN measurements according to given bounds
        flag_x2_ix = self.ix_inrange(
            colnames=["c2_X2norm"], lowlim=x2_max, exclude_lowlim=True
        )
        flag_stn_ix = self.ix_inrange(
            colnames=["c2_abs_stn"], lowlim=snr_max, exclude_lowlim=True
        )
        flag_nclip_ix = self.ix_inrange(
            colnames=["c2_Nclip"], lowlim=Nclip_max, exclude_lowlim=True
        )
        flag_ngood_ix = self.ix_inrange(
            colnames=["c2_Ngood"], uplim=Ngood_min, exclude_uplim=True
        )
        self.update_mask_column(x2_flag, flag_x2_ix)
        self.update_mask_column(snr_flag, flag_stn_ix)
        self.update_mask_column(Nclip_flag, flag_nclip_ix)
        self.update_mask_column(Ngood_flag, flag_ngood_ix)

        # update mask column with control light curve cut on any measurements flagged according to given bounds
        zero_Nclip_ix = self.ix_equal("c2_Nclip", 0)
        unmasked_ix = self.ix_unmasked(
            self.colnames.mask,
            maskval=x2_flag | snr_flag | Nclip_flag | Ngood_flag,
        )
        self.update_mask_column(questionable_flag, AnotB(unmasked_ix, zero_Nclip_ix))
        self.update_mask_column(flag, AnotB(self.getindices(), unmasked_ix))

    def copy_flags(self, flag_arr):
        self.t[self.colnames.mask] = self.t[self.colnames.mask].astype(np.int32)
        if len(self.t) < 1:
            return
        elif len(self.t) == 1:
            self.t.at[0, self.colnames.mask] = (
                int(self.t.at[0, self.colnames.mask]) | flag_arr
            )
        else:
            self.t[self.colnames.mask] = np.bitwise_or(
                self.t[self.colnames.mask], flag_arr
            )

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

    def preprocess(self, flux2mag_sigmalimit: float = 3.0):
        for i in self.lc_indices:
            self.get(i).preprocess(flux2mag_sigmalimit=flux2mag_sigmalimit)

    @property
    def colnames(self):
        return self.get_sn().colnames

    @property
    def default_mjd_colname(self):
        return self.get_sn().default_mjd_colname

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

    def apply_cut(
        self,
        column: str,
        flag: int,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        indices: Optional[List[int]] = None,
    ):
        for i in self.lcs:
            self.get(i).apply_cut(
                column, flag, min_value=min_value, max_value=max_value, indices=indices
            )

    def merge(self, other: Self) -> Self:
        """
        Merge this BaseTransient instance with another, combining their contained light curves.

        :returns: A new BaseTransient instance containing merged light curves from both inputs.

        - For light curves with the same index in both, their `merge` method is called and combined.
        - Light curves unique to either instance are added as-is.
        - The `filt` attribute of the new instance is set to `self.filt` only if both have the same non-None filter;
          otherwise, it is set to None.
        """
        if not isinstance(other, BaseTransient):
            raise TypeError("Can only merge with another BaseTransient instance.")

        merged_filt = self.filt
        if self.filt is None or other.filt is None or self.filt != other.filt:
            merged_filt = None

        new_transient = self.__class__(filt=merged_filt, verbose=self.logger.verbose)
        all_indices = set(self.lcs.keys()).union(set(other.lcs.keys()))

        for i in all_indices:
            if i in self.lcs and i in other.lcs:
                new_transient.add(self.get(i).merge(other.get(i)))
            elif i in self.lcs:
                new_transient.add(self.get(i))
            else:
                new_transient.add(other.get(i))

        return new_transient

    def split_by_filt(self) -> tuple[Self, Self]:
        """
        Create two new BaseTransient instances for filters 'o' and 'c'.
        Underlying dataframes are independent of self.
        """
        transient_o = self.__class__(filt="o", verbose=self.logger.verbose)
        transient_c = self.__class__(filt="c", verbose=self.logger.verbose)

        # iterate over all BaseLightCurves in this transient
        for lc in self.iterator():
            # split the BaseLightCurve into two independent ones, one for each filter
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

    def get_sn(self) -> LightCurve:
        return super().get_sn()

    def preprocess(self, flux2mag_sigmalimit=3.0):
        super().preprocess(flux2mag_sigmalimit=flux2mag_sigmalimit)
        self.match_control_mjds()

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

    def get_uncert_est_stats(
        self, temp_x2_max_value: float = 20, uncert_cut_flag: int = 0x2
    ) -> pd.DataFrame:
        """
        Get a table containing the median uncertainty, standard deviation of the flux,
        and sigma_extra for each control light curve.
        """

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
                self.colnames.mask, maskval=uncert_cut_flag
            )
            x2_clean_ix = self.get(i).ix_inrange(
                colnames=[self.colnames.chisquare],
                uplim=temp_x2_max_value,
                exclude_uplim=True,
            )
            clean_ix = AandB(dflux_clean_ix, x2_clean_ix)

            median_dflux = self.get(i).get_median_dflux(indices=clean_ix)
            stdev_flux = self.get(i).get_stdev_flux(indices=clean_ix)

            if stdev_flux is None:
                self.logger.warning(
                    f"Could not get flux std dev using clean indices; retrying without preliminary chi-square cut of {temp_x2_max_value}"
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

    def add_noise_to_dflux(self, sigma_extra: float):
        """
        Add sigma_extra to the SN and control light curves in quadrature.
        """
        for i in self.lc_indices:
            self.get(i).add_noise_to_dflux(sigma_extra)

    def calculate_control_stats(self, previous_flags: int):
        self.logger.info("Calculating control light curve statistics")

        len_mjd = len(self.get_sn().t[self.default_mjd_colname])

        # construct arrays for control lc data
        uJy = np.full((self.num_controls, len_mjd), np.nan)
        duJy = np.full((self.num_controls, len_mjd), np.nan)
        Mask = np.full((self.num_controls, len_mjd), 0, dtype=np.int32)

        i = 1
        for control_index in self.control_lc_indices:
            if len(self.get(control_index).t) != len_mjd or not np.array_equal(
                self.get_sn().t[self.default_mjd_colname],
                self.get(control_index).t[self.default_mjd_colname],
            ):
                raise RuntimeError(
                    f"SN light curve not equal to control light curve #{control_index}! Rerun or debug preprocessing."
                )
            else:
                uJy[i - 1, :] = self.get(control_index).t[self.colnames.flux]
                duJy[i - 1, :] = self.get(control_index).t[self.colnames.dflux]
                Mask[i - 1, :] = self.get(control_index).t[self.colnames.mask]

            i += 1

        c2_param2columnmapping = self.get_sn().intializecols4statparams(
            prefix="c2_", format4outvals="{:.2f}", skipparams=["converged", "i"]
        )

        for index in range(uJy.shape[-1]):
            pda4MJD = pdastrostatsclass()
            pda4MJD.t[self.colnames.flux] = uJy[0:, index]
            pda4MJD.t[self.colnames.dflux] = duJy[0:, index]
            pda4MJD.t[self.colnames.mask] = np.bitwise_and(
                Mask[0:, index], previous_flags
            )

            pda4MJD.calcaverage_sigmacutloop(
                self.colnames.flux,
                noisecol=self.colnames.dflux,
                maskcol=self.colnames.mask,
                maskval=previous_flags,
                verbose=1,
                Nsigma=3.0,
                median_firstiteration=True,
            )
            self.get_sn().statresults2table(
                pda4MJD.statparams, c2_param2columnmapping, destindex=index
            )

        self.get_sn().t["c2_abs_stn"] = (
            self.get_sn().t["c2_mean"] / self.get_sn().t["c2_mean_err"]
        )
        self.logger.success()

    def flag_by_control_stats(
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
    ):
        """
        Use control light curve statistics to flag measurements
        where control flux is inconsistent with 0.
        """
        # flag SN measurements
        self.get_sn().flag_by_control_stats(
            flag=flag,
            questionable_flag=questionable_flag,
            x2_max=x2_max,
            x2_flag=x2_flag,
            snr_max=snr_max,
            snr_flag=snr_flag,
            Nclip_max=Nclip_max,
            Nclip_flag=Nclip_flag,
            Ngood_min=Ngood_min,
            Ngood_flag=Ngood_flag,
        )

        # copy over SN mask column to control light curve mask columns
        flags_arr = np.full(
            self.get_sn().t[self.colnames.mask].shape,
            combine_flags(
                [flag, questionable_flag, x2_flag, snr_flag, Nclip_flag, Ngood_flag]
            ),
        )
        flags_to_copy = np.bitwise_and(self.get_sn().t[self.colnames.mask], flags_arr)
        for i in self.control_lc_indices:
            self.get(i).copy_flags(flags_to_copy)

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
