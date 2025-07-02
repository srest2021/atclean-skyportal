#!/usr/bin/env python

from copy import deepcopy
import sys
from typing import Dict, List, Optional, Self
from astropy.nddata import bitmask

import numpy as np
import pandas as pd
from pdastro import AandB, AnotB, AorB, not_AandB, pdastrostatsclass
from utils import (
    BinnedColumnNames,
    CleanedColumnNames,
    ColumnNames,
    Coordinates,
    CustomLogger,
    SigmaClipper,
    StatParams,
    combine_flags,
)


class BaseLightCurve(pdastrostatsclass):
    """
    Base class for storing and manipulating a light curve, including flux, uncertainty, mask flags, and metadata.
    Designed to work with pandas DataFrames and handle masking, filtering, clipping, and merging operations.
    """

    def __init__(
        self,
        control_index: int,
        coords: Coordinates,
        colnames: ColumnNames,
        filt: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize a BaseLightCurve instance.

        :param control_index: Index of the SN or control light curve (0 for SN light curve, 1+ for control light curves).
        :param coords: Coordinates of the object.
        :param colnames: ColumnNames instance describing column mappings (i.e., internal key to actual column string).
        :param filt: Optional filter name (e.g., 'o' or 'c').
        :param verbose: Whether to enable verbose logging.
        """
        super().__init__()
        self.logger = CustomLogger(verbose=verbose)
        self.control_index = control_index
        self.coords = coords
        self.filt = filt
        self.colnames = colnames

    def set(
        self, t: pd.DataFrame, indices: Optional[List[int]] = None, deep: bool = True
    ):
        """
        Set the internal DataFrame (`self.t`) with optional filtering and deep copy.

        :param t: Input DataFrame.
        :param indices: Optional subset of indices to keep.
        :param deep: Whether to deep copy the DataFrame.
        """
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
        """
        Return the default column name for MJD (time).
        Will be "mjdbin" for BinnedLightCurve but "mjd" here and everywhere else.
        """
        return self.colnames.mjd

    def preprocess(self, **kwargs):
        """
        Preprocess the light curve data.
        Calculates the SNR column as flux/dflux and replaces infs with NaNs.
        """
        if self.t is None or self.t.empty:
            return

        # calculate SNR
        self.calculate_snr_col()

    def _get_postprocess_cols_dict(self):
        """
        Return a dictionary mapping current column names to SkyPortal-compatible names.
        """
        return {
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

    def postprocess(self):
        """
        Prepares the cleaned light curve for SkyPortal ingestion.
        Renames columns, drops unnecessary columns, updates filter names,
        and adds optional columns for SkyPortal compatibility.
        """
        # convert column names
        update_cols_dict = self._get_postprocess_cols_dict()
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

    def get_filt_ix(self, filt: str):
        """
        Get indices of rows with a given filter value.

        :param filt: Filter to match (e.g., 'o', 'c').
        :return: List of row indices.
        """
        if self.t is None or self.t.empty:
            return []

        if self.filt is not None:
            return self.getindices()

        if self.colnames.filter not in self.t.columns:
            raise RuntimeError(f"Filter column '{self.colnames.filter}' not found")

        return self.ix_equal(self.colnames.filter, filt)

    def get_filt_lens(self) -> tuple[int, int]:
        """
        Return the number of entries in each filter ('o', 'c').

        :return: Tuple of (n_o, n_c).
        """
        if self.t is None:
            raise RuntimeError("Table (self.t) cannot be None")

        return len(self.get_filt_ix("o")), len(self.get_filt_ix("c"))

    def get_flags(self) -> int:
        """
        Get the bitwise OR of all mask flags in the light curve.
        """
        if self.t is None or self.t.empty or self.colnames.mask not in self.t.columns:
            return 0

        return np.bitwise_or.reduce(self.t[self.colnames.mask])

    def get_percent_flagged(self, flag: Optional[int] = None) -> float:
        """
        Return the fraction of rows flagged with a mask value.
        If no flag value specified, count all flags.

        :param flag: Optionally specify a single flag.
        :return: Percent of rows flagged.
        """
        bad_ix = self.get_bad_indices(flag=flag)
        return 100 * len(bad_ix) / len(self.t)

    def get_good_indices(self, flag: Optional[int] = None) -> List[int]:
        """
        Return the list of indices corresponding to "good" (unmasked) rows.
        A row is considered "good" if its value in the mask column does not contain
        the specified flag.
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
        the specified flag.
        """
        if self.t is None or self.t.empty:
            return []

        if not self.colnames.mask in self.t.columns:
            return self.getindices()

        if flag == 0:
            flag = None

        # note: if flag is None, this will just return all masked indices
        return self.ix_masked(self.colnames.mask, maskval=flag)

    def remove_flag(self, flag: int, indices: Optional[List[int]] = None):
        """
        Remove a bitmask flag from selected rows.

        :param flag: Integer bitmask to remove.
        :param indices: Optional list of row indices.
        """
        if self.t is None or self.t.empty or self.colnames.mask not in self.t.columns:
            return

        indices = self.getindices(indices)
        self.t.loc[indices, self.colnames.mask] = np.bitwise_and(
            self.t.loc[indices, self.colnames.mask].astype(int), ~flag
        )

    def update_mask_column(
        self, flag, indices: Optional[List[int]] = None, remove_old: bool = True
    ):
        """
        Add a bitmask flag to selected rows in the mask column.

        :param flag: Flag value to add.
        :param indices: Indices to modify.
        :param remove_old: If True, remove any existing instances of the flag first.
        """
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
        """
        Apply a flag to rows where the column value is outside the allowed range.

        :param column: Name of the column to cut on.
        :param flag: Bitmask flag to apply.
        :param min_value: Minimum allowed value.
        :param max_value: Maximum allowed value.
        :param indices: Optional subset of rows to check.
        """
        if self.t is None or self.t.empty:
            return

        if not column in self.t.columns:
            raise RuntimeError(
                f"Column '{column}' not found in light curve; cannot apply cut"
            )

        all_ix = self.getindices(indices)
        cut_ix = self.ix_outrange(
            colnames=[column],
            lowlim=min_value,
            uplim=max_value,
            exclude_lowlim=True,
            exclude_uplim=True,
            indices=all_ix,
        )

        self.update_mask_column(flag, cut_ix)

    def get_stdev_flux(self, indices=None):
        """
        Compute the sigma-clipped standard deviation of the flux.

        :param indices: Optional list of indices to include.
        :return: Clipped standard deviation.
        """
        if self.t is None or self.t.empty:
            return np.nan

        self.calcaverage_sigmacutloop(
            self.colnames.flux, indices=indices, Nsigma=3.0, median_firstiteration=True
        )
        return self.statparams["stdev"]

    def get_median_dflux(self, indices=None):
        """
        Return the median of the flux uncertainties.

        :param indices: Optional indices to include.
        :return: Median of dflux values.
        """
        if self.t is None or self.t.empty or (indices is not None and len(indices) < 1):
            return np.nan
        if indices is None:
            indices = self.getindices()
        return np.nanmedian(self.t.loc[indices, self.colnames.dflux])

    def calculate_snr_col(self):
        """
        Calculates the SNR column as flux/dflux and replaces infs with NaNs.
        """
        if self.t is None or self.t.empty:
            return

        # replace infs with NaNs
        # self.logger.info("Replacing infs with NaNs")
        self.t.replace([np.inf, -np.inf], np.nan, inplace=True)

        # calculate flux/dflux
        # self.logger.info(f"Calculating flux/dflux for '{self.colnames.snr}' column")
        self.t[self.colnames.snr] = (
            self.t[self.colnames.flux] / self.t[self.colnames.dflux]
        )

    def _get_average_flux(self, indices=None) -> StatParams:
        """
        Run sigma clipping on the flux column.

        :param indices: Optional indices to include.
        :return: StatParams with results.
        """
        sigma_clipper = SigmaClipper(verbose=False)
        sigma_clipper.calcaverage_sigmacutloop(
            self.t[self.colnames.flux].values,
            noise_arr=self.t[self.colnames.dflux].values,
            indices=indices,
            Nsigma=3.0,
            median_firstiteration=True,
        )
        return sigma_clipper.statparams

    def _get_average_mjd(self, indices=None) -> float:
        """
        Return the average MJD using sigma clipping.

        :param indices: Optional indices to include.
        :return: Mean MJD value.
        """
        sigma_clipper = SigmaClipper(verbose=False)
        sigma_clipper.calcaverage_sigmacutloop(
            self.t[self.default_mjd_colname].values,
            indices=indices,
            Nsigma=0,
            median_firstiteration=False,
        )
        return sigma_clipper.statparams.mean

    def new(self, filt: str | None):
        return self.__class__(
            self.control_index,
            self.coords,
            self.colnames,
            filt=filt,
            verbose=self.logger.verbose,
        )

    def merge(self, other: Self) -> Self:
        """
        Merge this LightCurve with another one and return a new LightCurve instance
        with the combined and sorted data. Used for merging two light curves with
        different filters together.

        :param other: Another light curve to merge with.
        :return: New BaseLightCurve instance with merged and sorted data.
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
        merged_t: pd.DataFrame = pd.concat(
            [
                pd.DataFrame() if self.t.empty else self.t,
                pd.DataFrame() if other.t.empty else other.t,
            ],
            ignore_index=True,
        )
        if not merged_t.empty:
            merged_t.sort_values(
                by=[self.default_mjd_colname], ignore_index=True, inplace=True
            )

        # create new lc
        new_lc = self.new(merged_filt)
        new_lc.t = deepcopy(merged_t)  # skip preprocessing
        return new_lc

    def split_by_filt(self) -> tuple[Self, Self]:
        """
        Split the light curve into two new instances for filters 'o' and 'c'.
        Underlying DataFrames are deepcopied and independent of self.

        :return: Two new BaseLightCurve instances for filters 'o' and 'c'.
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
    """
    Class for representing and cleaning an ATLAS light curve.

    Inherits from BaseLightCurve and provides additional methods for
    preprocessing, uncertainty handling, flagging, and preparing data
    for SkyPortal ingestion.
    """

    def __init__(
        self,
        control_index: int,
        coords: Coordinates,
        filt: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize a LightCurve with cleaned column names.

        :param control_index: Index of the SN or control light curve (0 = SN, >0 = control).
        :param coords: Sky coordinates of the light curve.
        :param filt: Optional filter name ('o' or 'c').
        :param verbose: Whether to enable verbose logging.
        """
        super().__init__(
            control_index, coords, CleanedColumnNames(), filt=filt, verbose=verbose
        )

    def new(self, filt: str | None):
        return LightCurve(
            self.control_index,
            self.coords,
            filt=filt,
            verbose=self.logger.verbose,
        )

    def preprocess(self, flux2mag_sigmalimit=3.0):
        """
        Prepare the raw ATLAS light curve for cleaning.

        Adds a mask column if missing, sorts by MJD, removes rows with duJy=0 or uJy=NaN,
        and overwrites ATLAS magnitudes with calculated values.

        :param flux2mag_sigmalimit: Sigma limit for converting flux to magnitude (default: 3.0).
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
        # self.logger.info(
        #     f'Deleting {len(dflux_zero_ix) + len(flux_nan_ix)} rows with "duJy"==0 or "uJy"==NaN'
        # )
        if len(AorB(dflux_zero_ix, flux_nan_ix)) > 0:
            self.t = self.t.drop(AorB(dflux_zero_ix, flux_nan_ix))

        # convert flux to magnitude
        # self.logger.info(
        #     "Converting flux to magnitude (and overwriting original ATLAS 'm' and 'dm' columns)"
        # )
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

    def add_noise_to_dflux(self, sigma_extra: float):
        """
        Add additional noise in quadrature to the dflux column.
        Store the sigma_extra we added in a new column `dflux_offset_in_quadrature`.
        """
        # add in quadrature to dflux column
        self.t[self.colnames.dflux] = np.sqrt(
            self.t[self.colnames.dflux] ** 2 + sigma_extra**2
        )

        # store the sigma_extra we added in a new column
        self.colnames.add("dflux_offset", "dflux_offset_in_quadrature")
        self.t[self.colnames.dflux_offset] = np.full(len(self.t), sigma_extra)

        # recalculate SNR
        self.calculate_snr_col()

    def remove_noise_from_dflux(self):
        """
        Remove sigma_extra that was previously added in quadrature to the dflux column.
        Assumes sigma_extra is stored in self.colnames.dflux_offset as a constant value.
        """
        if (
            not self.colnames.has("dflux_offset")
            or self.colnames.dflux_offset not in self.t.columns
        ):
            self.logger.warning(
                f"dflux offset column not found in control light curve #{self.control_index}; skipping noise removal..."
            )
            return

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

    def update_statparams(
        self,
        statparams: StatParams,
        index: Optional[int] = None,
        prefix="controls_",
    ):
        """
        Update the light curve table with values from a StatParams object.

        :param statparams: StatParams instance to pull data from.
        :param index: Row index to update. If None, defaults to last row.
        :param prefix: Column prefix to use for inserted columns.
        """
        if index < 0 or index >= len(self.t):
            raise IndexError(
                f"Index {index} out of range for DataFrame of length {len(self.t)}"
            )

        row = statparams.get_row(prefix=prefix, skip=["ix_good", "ix_clip"])

        # add missing columns
        for col in row:
            if col not in self.t.columns:
                self.t[col] = np.nan

        if index is None:
            index = len(self.t) - 1
        for col, val in row.items():
            self.t.at[index, col] = val

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
        prefix="controls_",
    ):
        """
        Flag SN measurements based on control light curve statistics.

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
        :param prefix: Prefix used for control statistic column names.
        """
        # flag SN measurements according to given bounds
        flag_x2_ix = self.ix_inrange(
            colnames=[f"{prefix}X2norm"], lowlim=x2_max, exclude_lowlim=True
        )
        flag_stn_ix = self.ix_inrange(
            colnames=[f"{prefix}abs_stn"], lowlim=snr_max, exclude_lowlim=True
        )
        flag_nclip_ix = self.ix_inrange(
            colnames=[f"{prefix}Nclip"], lowlim=Nclip_max, exclude_lowlim=True
        )
        flag_ngood_ix = self.ix_inrange(
            colnames=[f"{prefix}Ngood"], uplim=Ngood_min, exclude_uplim=True
        )
        self.update_mask_column(x2_flag, flag_x2_ix)
        self.update_mask_column(snr_flag, flag_stn_ix)
        self.update_mask_column(Nclip_flag, flag_nclip_ix)
        self.update_mask_column(Ngood_flag, flag_ngood_ix)

        # update mask column with control light curve cut on any measurements flagged according to given bounds
        zero_Nclip_ix = self.ix_equal(f"{prefix}Nclip", 0)
        unmasked_ix = self.ix_unmasked(
            self.colnames.mask,
            maskval=x2_flag | snr_flag | Nclip_flag | Ngood_flag,
        )
        self.update_mask_column(questionable_flag, AnotB(unmasked_ix, zero_Nclip_ix))
        self.update_mask_column(flag, AnotB(self.getindices(), unmasked_ix))

    def copy_flags(self, flag_arr):
        """
        Copy flag values into the mask column for all rows.

        :param flag_arr: Array-like flag(s) to apply to the mask column.
        """
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

    def _compute_mjd_bins(self, mjd_arr, mjd_bin_size: float = 1.0):
        """
        Compute MJD bin edges and assign bin indices.

        :param mjd_arr: Array of MJD values.
        :param mjd_bin_size: Bin size in days.
        :return: Tuple of (bin edges, bin index array).
        """
        mjd_min = np.floor(mjd_arr.min())
        mjd_max = np.ceil(mjd_arr.max())
        bins = np.arange(mjd_min, mjd_max + mjd_bin_size, mjd_bin_size)
        bin_indices = np.digitize(mjd_arr, bins, right=False) - 1
        return bins, bin_indices

    def get_BinnedLightCurve(
        self,
        previous_flags: int,
        flag: int = 0x800000,
        mjd_bin_size: float = 1.0,
        x2_max: float = 4.0,
        Nclip_max: int = 1,
        Ngood_min: int = 2,
        large_num_clipped_flag: int = 0x1000,
        small_num_unmasked_flag: int = 0x2000,
        flux2mag_sigmalimit: float = 3.0,
    ):
        """
        Bin light curve data in time and flag problematic bins.

        :param previous_flags: Flags to mask out before binning.
        :param flag: Flag to apply to bins with invalid or bad data.
        :param mjd_bin_size: Size of MJD bins in days.
        :param x2_max: Maximum chi-square allowed for a good bin.
        :param Nclip_max: Maximum allowed clipped points in a bin.
        :param Ngood_min: Minimum number of good points required in a bin.
        :param large_num_clipped_flag: Flag for excessive clipping.
        :param small_num_unmasked_flag: Flag for insufficient unmasked measurements.
        :param flux2mag_sigmalimit: Sigma threshold for converting flux to mag in final binned LC.
        :return: A new BinnedLightCurve object.
        """
        binned_lc = BinnedLightCurve(
            self.control_index,
            self.coords,
            mjd_bin_size,
            filt=self.filt,
            verbose=self.logger.verbose,
        )
        binned_lc.t = pd.DataFrame(columns=list(self.colnames.all))

        if self.t is None or self.t.empty:
            return binned_lc

        mjd_arr = self.t[self.colnames.mjd].values
        mask_arr = self.t[self.colnames.mask].values.astype(int)
        bins, bin_indices = self._compute_mjd_bins(mjd_arr, mjd_bin_size)

        if self.control_index == 0:
            self.logger.info(f"Now averaging SN light curve")
        else:
            self.logger.info(f"Now averaging control light curve {self.control_index}")

        for b in range(len(bins) - 1):
            bin_ix = np.where(bin_indices == b)[0]

            bin_center = bins[b] + 0.5 * mjd_bin_size

            # if no measurements present, flag and skip over day
            if len(bin_ix) < 1:
                cur_index = binned_lc.newrow(
                    {
                        binned_lc.colnames.mjdbin: bin_center,
                        binned_lc.colnames.nclip: 0,
                        binned_lc.colnames.ngood: 0,
                        binned_lc.colnames.nexcluded: 0,
                        binned_lc.colnames.mask: flag,
                    }
                )
                continue

            good_bin_ix = SigmaClipper.ix_unmasked(
                mask_arr, mask_val=previous_flags, indices=bin_ix
            )
            cur_index = binned_lc.newrow(
                {
                    binned_lc.colnames.mjdbin: bin_center,
                    binned_lc.colnames.nclip: 0,
                    binned_lc.colnames.ngood: 0,
                    binned_lc.colnames.nexcluded: len(bin_ix) - len(good_bin_ix),
                    binned_lc.colnames.mask: 0,
                }
            )

            # if no good measurements, average values anyway and flag
            if len(good_bin_ix) == 0:
                flux_statparams = self._get_average_flux(
                    indices=bin_ix,
                )
                avg_mjd = self._get_average_mjd(
                    indices=bin_ix,
                )
                binned_lc.add2row(
                    cur_index,
                    {
                        binned_lc.colnames.mjd: avg_mjd,
                        binned_lc.colnames.flux: flux_statparams.mean,
                        binned_lc.colnames.dflux: flux_statparams.mean_err,
                        binned_lc.colnames.stdev: flux_statparams.stdev,
                        binned_lc.colnames.x2: flux_statparams.X2norm,
                        binned_lc.colnames.nclip: flux_statparams.Nclip,
                        binned_lc.colnames.ngood: flux_statparams.Ngood,
                        binned_lc.colnames.mask: flag,
                    },
                )
                self.update_mask_column(flag, indices=bin_ix, remove_old=False)
                continue

            flux_statparams = self._get_average_flux(indices=good_bin_ix)
            if np.isnan(flux_statparams.mean) or len(flux_statparams.ix_good) < 1:
                self.update_mask_column(flag, indices=bin_ix, remove_old=False)
                binned_lc.update_mask_column(
                    flag, indices=[cur_index], remove_old=False
                )
                continue

            avg_mjd = self._get_average_mjd(indices=flux_statparams.ix_good)

            binned_lc.add2row(
                cur_index,
                {
                    binned_lc.colnames.mjd: avg_mjd,
                    binned_lc.colnames.flux: flux_statparams.mean,
                    binned_lc.colnames.dflux: flux_statparams.mean_err,
                    binned_lc.colnames.stdev: flux_statparams.stdev,
                    binned_lc.colnames.x2: flux_statparams.X2norm,
                    binned_lc.colnames.nclip: flux_statparams.Nclip,
                    binned_lc.colnames.ngood: flux_statparams.Ngood,
                    binned_lc.colnames.mask: 0,
                },
            )

            if len(flux_statparams.ix_clip) > 0:
                self.update_mask_column(
                    large_num_clipped_flag,
                    indices=flux_statparams.ix_clip,
                    remove_old=False,
                )

            if len(good_bin_ix) < 3:  # TODO: un-hardcode this!
                self.update_mask_column(
                    small_num_unmasked_flag, indices=bin_ix, remove_old=False
                )
                binned_lc.update_mask_column(
                    small_num_unmasked_flag, indices=[cur_index], remove_old=False
                )
            else:
                is_bad = (
                    flux_statparams.Ngood < Ngood_min
                    or flux_statparams.Nclip > Nclip_max
                    or (
                        flux_statparams.X2norm is not None
                        and flux_statparams.X2norm > x2_max
                    )
                )
                if is_bad:
                    self.update_mask_column(flag, indices=bin_ix, remove_old=False)
                    binned_lc.update_mask_column(
                        flag, indices=[cur_index], remove_old=False
                    )

        binned_lc.flux2mag(
            binned_lc.colnames.flux,
            binned_lc.colnames.dflux,
            binned_lc.colnames.mag,
            binned_lc.colnames.dmag,
            zpt=23.9,
            upperlim_Nsigma=flux2mag_sigmalimit,
        )
        if "__tmp_SN" in binned_lc.t.columns:
            binned_lc.t.drop(columns=["__tmp_SN"], inplace=True)

        return binned_lc


class BinnedLightCurve(BaseLightCurve):
    def __init__(
        self,
        control_index: int,
        coords: Coordinates,
        mjd_bin_size: float,
        filt: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize a BinnedLightCurve with specified MJD bin size.

        :param control_index: Index of the SN or control light curve (0 for SN light curve, 1+ for control light curves).
        :param coords: Coordinates of the object.
        :param mjd_bin_size: Bin size in MJD days.
        :param filt: Optional filter name ('o' or 'c').
        :param verbose: Whether to enable verbose logging.
        """
        super().__init__(
            control_index, coords, BinnedColumnNames(), filt=filt, verbose=verbose
        )
        self.mjd_bin_size = mjd_bin_size

    def new(self, filt: str | None):
        return BinnedLightCurve(
            self.control_index,
            self.coords,
            self.mjd_bin_size,
            filt=filt,
            verbose=self.logger.verbose,
        )

    @property
    def default_mjd_colname(self):
        """
        Return the default MJD column name for binne light curves
        (i.e., the middle of the bin, not the mean mjd for the bin).
        """
        return self.colnames.mjdbin

    def _get_postprocess_cols_dict(self):
        """
        Return a dictionary mapping current column names to SkyPortal-compatible names,
        including the MJD bin column.
        """
        cols_dict = super()._get_postprocess_cols_dict()
        cols_dict[self.colnames.mjdbin] = "mjd_bin"
        return cols_dict

    def get_percent_flagged(self, flag: Optional[int] = None) -> float:
        """
        Return the fraction of rows flagged with a mask value.
        If no flag value specified, count all flags.
        Exclude NaN bins from the percentage.

        :param flag: Optionally specify a single flag.
        :return: Percent of rows flagged.
        """
        non_null_ix = self.ix_not_null(self.colnames.mjd)
        bad_ix = self.ix_masked(self.colnames.mask, maskval=flag, indices=non_null_ix)
        return 100 * len(bad_ix) / len(non_null_ix)


class BaseTransient:
    """
    Store and do operations on a collection of light curves
    (i.e., the SN light curve and its control light curves).
    """

    def __init__(self, filt: Optional[str] = None, verbose: bool = False):
        """
        Initialize a BaseTransient with an optional filter and verbosity.

        :param filt: Optional filter name ('o', 'c', or None) for this transient.
        :param verbose: Enable verbose logging.
        """
        self.filt = filt
        self.lcs: Dict[int, BaseLightCurve] = {}
        self.logger = CustomLogger(verbose=verbose)

    def get(self, control_index: int) -> BaseLightCurve:
        """
        Get the BaseLightCurve for the given control index.

        :param control_index: Index of the SN or control light curve.
        :raises ValueError: If no light curve exists with that control index.
        :returns: The BaseLightCurve instance.
        """
        if control_index not in self.lcs:
            raise ValueError(
                f"Light curve with control index {control_index} does not exist"
            )
        return self.lcs[control_index]

    def get_sn(self) -> BaseLightCurve:
        """
        Get the SN light curve, which has control index 0.
        """
        return self.get(0)

    def preprocess(self, flux2mag_sigmalimit: float = 3.0):
        """
        Preprocess all contained light curves.

        :param flux2mag_sigmalimit: Sigma limit used when converting flux to magnitude.
        """
        self.logger.info("\nPreprocessing all light curves")
        for i in self.lc_indices:
            self.get(i).preprocess(flux2mag_sigmalimit=flux2mag_sigmalimit)

    def postprocess(self):
        """
        Postprocess all contained light curves.

        :raises RuntimeError: If the transient filter is not None (postprocessing requires no filter).
        """
        if self.filt is not None:
            raise RuntimeError(
                f"Filter '{self.filt}' should be None for postprocessing"
            )

        for i in self.lc_indices:
            self.get(i).postprocess()

    @property
    def colnames(self):
        """
        Get the column names from the SN light curve.
        """
        return self.get_sn().colnames

    @property
    def default_mjd_colname(self):
        """
        Get the default MJD column name from the SN light curve.
        Will be MJD for LightCurve and MJD bin for BinnedLightCurve.
        """
        return self.get_sn().default_mjd_colname

    @property
    def num_controls(self):
        """
        Count the number of control light curves (excluding SN).
        """
        return sum(1 for k in self.lcs if k != 0)

    @property
    def lc_indices(self) -> List[int]:
        """
        List indices of all light curves currently stored.
        """
        return list(self.lcs)

    @property
    def control_lc_indices(self) -> List[int]:
        """
        List all control light curve indices.
        """
        return [i for i in self.lcs if i != 0]

    def add(self, lc: BaseLightCurve, deep: bool = True):
        """
        Add or overwrite a light curve with the given control index.

        :param lc: BaseLightCurve instance to add.
        :param deep: If True, add a deep copy; otherwise add the object directly.
        """
        if lc.control_index in self.lcs:
            self.logger.warning(
                f"Control index {lc.control_index} already exists; overwriting..."
            )
        if (self.filt is not None or lc.filt is not None) and self.filt != lc.filt:
            self.logger.warning(
                f"Control index {lc.control_index} filter '{lc.filt}' does not match transient filter '{self.filt}'"
            )
        self.lcs[lc.control_index] = deepcopy(lc) if deep else lc

    def update(self, lc: BaseLightCurve, deep: bool = False):
        """
        Update a light curve by control index.

        :param lc: BaseLightCurve instance to update.
        :param deep: If True, store a deep copy; otherwise store the original object.
        """
        if not lc.control_index in self.lcs:
            self.logger.warning(
                f"Control index {lc.control_index} doesn't exist yet; adding..."
            )
        self.lcs[lc.control_index] = deepcopy(lc) if deep else lc

    def iterator(self):
        """
        Iterator over all contained BaseLightCurves.
        """
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
        """
        Apply a flag across all light curves to rows where the column value is outside the allowed range.

        :param column: Name of the column to cut on.
        :param flag: Bitmask flag to apply.
        :param min_value: Minimum allowed value.
        :param max_value: Maximum allowed value.
        :param indices: Optional subset of rows to check.
        """
        for i in self.lcs:
            self.get(i).apply_cut(
                column, flag, min_value=min_value, max_value=max_value, indices=indices
            )

    def remove_flag(self, flag: int):
        """
        Remove a specified flag from all light curves.
        """
        for i in self.lc_indices:
            self.get(i).remove_flag(flag)

    def get_all_controls(self):
        """
        Concatenate all control light curves into a single light curve instance.
        """
        cls = type(self.get(0))
        all_controls = cls(None, None, verbose=self.logger.verbose)

        all_controls.t = pd.concat(
            [self.get(i).t for i in self.control_lc_indices],
            ignore_index=True,
            copy=False,
        )
        return all_controls

    def new(self, filt: str | None):
        return self.__class__(
            filt=filt,
            verbose=self.logger.verbose,
        )

    def merge(self, other: Self) -> Self:
        """
        Merge this BaseTransient instance with another, combining their contained light curves.

        :return: A new BaseTransient instance containing merged light curves from both inputs.

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

        new_transient = self.new(merged_filt)
        all_indices = set(self.lcs.keys()).union(set(other.lcs.keys()))

        for i in all_indices:
            if i in self.lcs and i in other.lcs:
                new_transient.add(self.get(i).merge(other.get(i)))
            elif i in self.lcs:
                self.get(i).filt = None
                new_transient.add(self.get(i))
            else:
                other.get(i).filt = None
                new_transient.add(other.get(i))

        return new_transient

    def split_by_filt(self) -> tuple[Self, Self]:
        """
        Create two new BaseTransient instances for filters 'o' and 'c'.
        Underlying DataFrames are independent of self.
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
    """
    Store and do operations on a collection of light curves
    (i.e., the SN light curve and its control light curves).
    """

    def __init__(self, filt: Optional[str] = None, verbose: bool = False):
        """
        Initialize a Transient with optional filter and verbosity.

        :param filt: Optional filter name ('o', 'c', or None).
        :param verbose: Enable verbose logging.
        """
        super().__init__(filt=filt, verbose=verbose)
        self.lcs: Dict[int, LightCurve] = {}

    def get(self, control_index: int) -> LightCurve:
        return super().get(control_index)

    def get_sn(self) -> LightCurve:
        return super().get_sn()

    def preprocess(self, flux2mag_sigmalimit=3.0):
        """
        Preprocess all light curves and align control MJDs to SN.

        :param flux2mag_sigmalimit: Sigma limit for flux-to-mag conversion.
        """
        super().preprocess(flux2mag_sigmalimit=flux2mag_sigmalimit)
        self.match_control_mjds()

    def match_control_mjds(self):
        """
        Ensure control light curves have the same MJDs as the SN light curve.
        Adds or removes rows in control light curves as necessary.
        """
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
        self, temp_x2_max_value: float = 20, uncertainty_cut_flag: int = 0x2
    ) -> pd.DataFrame:
        """
        Get a table containing the median uncertainty, standard deviation of the flux,
        and sigma_extra for each control light curve.

        :param temp_x2_max_value: Temporary PSF chi-square upper bound for filtering out egregious outliers.
        :param uncertainty_cut_flag: Flag used in the previous uncertainty cut for filtering out egregious outliers.
        :return: DataFrame with median_dflux, stdev, and sigma_extra for each control index.
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
                self.colnames.mask, maskval=uncertainty_cut_flag
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
        Add extra noise in quadrature to all light curves.
        Store the sigma_extra we added in a new column `dflux_offset_in_quadrature`.
        """
        for i in self.lc_indices:
            self.get(i).add_noise_to_dflux(sigma_extra)

    def remove_noise_from_dflux(self):
        for i in self.lc_indices:
            self.get(i).remove_noise_from_dflux()

    def calculate_control_stats(self, previous_flags: int, prefix="controls_"):
        """
        Compute control light curve statistics for each MJD epoch
        (i.e., for each epoch, take sigma clipping of the n control measurements corresponding to that epoch,
        then use results to flag that epoch across all light curves).

        :param previous_flags: Combined bitmask flags of previous cuts which we use to exclude bad control measurements from the sigma clipping.
        :param prefix: Prefix to use for output column names.
        """
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

        sigma_clipper = SigmaClipper(verbose=False)
        for index in range(uJy.shape[-1]):
            sigma_clipper.calcaverage_sigmacutloop(
                uJy[0:, index],
                noise_arr=duJy[0:, index],
                mask_arr=np.bitwise_and(Mask[0:, index], previous_flags),
                mask_val=previous_flags,
                Nsigma=3.0,
                median_firstiteration=True,
            )
            self.get_sn().update_statparams(
                sigma_clipper.statparams, index=index, prefix=prefix
            )

        self.get_sn().t[f"{prefix}abs_stn"] = (
            self.get_sn().t[f"{prefix}mean"] / self.get_sn().t[f"{prefix}mean_err"]
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
        Use control light curve statistics to flag SN epochs where control flux is inconsistent with 0.
        Propagate resulting flags to controls.

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

    def get_BinnedTransient(
        self,
        previous_flags: int,
        flag: int = 0x800000,
        mjd_bin_size: float = 1.0,
        x2_max: float = 4.0,
        Nclip_max: int = 1,
        Ngood_min: int = 2,
        large_num_clipped_flag: int = 0x1000,
        small_num_unmasked_flag: int = 0x2000,
        flux2mag_sigmalimit=3.0,
    ):
        """
        Generate a binned version of this Transient.

        :param previous_flags: Flags to mask out before binning.
        :param flag: Flag to apply to bins with invalid or bad data.
        :param mjd_bin_size: Size of MJD bins in days.
        :param x2_max: Maximum chi-square allowed for a good bin.
        :param Nclip_max: Maximum allowed clipped points in a bin.
        :param Ngood_min: Minimum number of good points required in a bin.
        :param large_num_clipped_flag: Flag for excessive clipping.
        :param small_num_unmasked_flag: Flag for insufficient unmasked measurements.
        :param flux2mag_sigmalimit: Sigma threshold for converting flux to mag in final binned LC.
        :return: BinnedTransient containing binned light curves.
        """
        binned_transient = BinnedTransient(
            mjd_bin_size, filt=self.filt, verbose=self.logger.verbose
        )

        for lc in self.iterator():
            binned_lc = lc.get_BinnedLightCurve(
                previous_flags,
                flag=flag,
                mjd_bin_size=mjd_bin_size,
                x2_max=x2_max,
                Nclip_max=Nclip_max,
                Ngood_min=Ngood_min,
                large_num_clipped_flag=large_num_clipped_flag,
                small_num_unmasked_flag=small_num_unmasked_flag,
                flux2mag_sigmalimit=flux2mag_sigmalimit,
            )

            binned_transient.add(binned_lc)

        return binned_transient


class BinnedTransient(BaseTransient):
    """
    Store and do operations on a collection of binned light curves
    (i.e., the binned SN light curve and its binned control light curves).
    """

    def __init__(
        self, mjd_bin_size: float, filt: Optional[str] = None, verbose: bool = False
    ):
        """
        Initialize a BinnedTransient, which holds binned light curves for a transient.

        :param mjd_bin_size: Size of MJD bins in days.
        :param filt: Optional filter string (e.g., 'o' or 'c').
        :param verbose: Enable verbose logging.
        """
        super().__init__(filt=filt, verbose=verbose)
        self.lcs: Dict[int, BinnedLightCurve] = {}
        self.mjd_bin_size = mjd_bin_size

    def get(self, control_index: int) -> BinnedLightCurve:
        return super().get(control_index)

    def new(self, filt: Optional[str] = None):
        return BinnedTransient(
            self.mjd_bin_size,
            filt=self.filt if filt is None else filt,
            verbose=self.logger.verbose,
        )

    def merge(self, other: Self) -> Self:
        if self.mjd_bin_size != other.mjd_bin_size:
            raise ValueError(
                f"Can only merge with a BinnedTransient instance of the same MJD bin size (self: {self.mjd_bin_size:0.2f} days; other: {other.mjd_bin_size:0.2f} days)"
            )
        return super().merge(other)
