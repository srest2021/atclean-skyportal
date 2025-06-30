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
    StatParams,
    combine_flags,
)


class BaseLightCurve(pdastrostatsclass):
    """
    Base class for representing and manipulating a single light curve.

    Provides methods for masking, flagging, filtering, merging, and basic statistics
    on light curve data, as well as utilities for handling different filters and cuts.
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

        Parameters
        ----------
        control_index : int
            Index identifying the control light curve (0 if it's the SN).
        coords : Coordinates
            Object containing coordinate information (RA, Dec) for the light curve.
        colnames : ColumnNames
            Object containing column name mappings.
        filt : str, optional
            Filter name (e.g., 'o' or 'c'). If None, includes all filters.
        verbose : bool, optional
            If True, enables verbose logging.
        """
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
        """
        Set the internal data table for the light curve.

        Parameters
        ----------
        t : pd.DataFrame
            DataFrame containing light curve data.
        indices : list of int, optional
            Indices to select from the DataFrame. If None, use all rows.
        deep : bool, optional
            If True, perform a deep copy of the data.
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

        Returns
        -------
        str
            The column name for MJD.
        """
        return self.colnames.mjd

    def preprocess(self, **kwargs):
        """
        Preprocess the light curve data.

        Calculates the SNR column.
        """
        if self.t is None or self.t.empty:
            return

        # calculate SNR
        self.calculate_snr_col()

    def _get_postprocess_cols_dict(self):
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
        Prepare the cleaned light curve for SkyPortal ingestion.

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
        Get indices of rows matching a specific filter.

        Parameters
        ----------
        filt : str
            Filter name to select (e.g., 'o' or 'c').

        Returns
        -------
        list of int
            Indices of rows with the specified filter.
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
        Get the number of rows for each filter ('o' and 'c').

        Returns
        -------
        tuple of int
            Number of rows for filters 'o' and 'c', respectively.
        """
        if self.t is None:
            raise RuntimeError("Table (self.t) cannot be None")

        return len(self.get_filt_ix("o")), len(self.get_filt_ix("c"))

    def get_flags(self) -> int:
        """
        Get the bitwise OR of all mask flags in the light curve.

        Returns
        -------
        int
            Combined mask flags for all rows.
        """
        if self.t is None or self.t.empty or self.colnames.mask not in self.t.columns:
            return 0

        return np.bitwise_or.reduce(self.t[self.colnames.mask])

    def get_percent_flagged(self, flag: Optional[int] = None) -> float:
        bad_ix = self.get_bad_indices(flag=flag)
        return len(bad_ix) / len(self.t)

    def get_good_indices(self, flag: Optional[int] = None) -> List[int]:
        """
        Return the list of indices corresponding to "good" (unmasked) rows.
        A row is considered "good" if its value in the mask column does not contain
        the specified flag.

        Parameters
        ----------
        flag : int, optional
            Bitmask flag to check. If None or 0, returns all unmasked rows.

        Returns
        -------
        list of int
            Indices of good rows.
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

        Parameters
        ----------
        flag : int, optional
            Bitmask flag to check. If None or 0, returns all masked rows.

        Returns
        -------
        list of int
            Indices of bad rows.
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
        Remove a specific flag from the mask column for the given rows.

        Parameters
        ----------
        flag : int
            Bitmask flag to remove.
        indices : list[int], optional
            List of row indices to update. If None, the flag is removed from all rows.
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
        Update the mask column for specified indices by adding a flag.

        Parameters
        ----------
        flag : int
            Bitmask flag to add.
        indices : list of int
            Indices to update.
        remove_old : bool, optional
            If True, remove the flag from all rows before adding it to the specified indices.
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
        Apply a value-based cut to a column, masking out-of-range data.

        Parameters
        ----------
        column : str
            Name of the column to cut on.
        flag : int
            Bitmask flag to apply to masked data.
        min_value : float, optional
            Minimum allowed value (inclusive).
        max_value : float, optional
            Maximum allowed value (inclusive).
        indices : list of int, optional
            Indices to consider for the cut. If None, all indices are used.
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
        Calculate the standard deviation of the flux column, optionally for a subset of indices.

        Parameters
        ----------
        indices : list of int, optional
            Indices to use for the calculation. If None, use all rows.

        Returns
        -------
        float
            Standard deviation of the flux.
        """
        if self.t is None or self.t.empty:
            return np.nan

        self.calcaverage_sigmacutloop(
            self.colnames.flux, indices=indices, Nsigma=3.0, median_firstiteration=True
        )
        return self.statparams["stdev"]

    def get_median_dflux(self, indices=None):
        """
        Calculate the median uncertainty (dflux), optionally for a subset of indices.

        Parameters
        ----------
        indices : list of int, optional
            Indices to use for the calculation. If None, use all rows.

        Returns
        -------
        float
            Median value of the dflux column.
        """
        if self.t is None or self.t.empty or (indices is not None and len(indices) < 1):
            return np.nan
        if indices is None:
            indices = self.getindices()
        return np.nanmedian(self.t.loc[indices, self.colnames.dflux])

    def calculate_snr_col(self):
        """
        Calculate the SNR column as flux/dflux and replace infs with NaNs.
        """
        if self.t is None or self.t.empty:
            return

        # replace infs with NaNs
        self.logger.info("Replacing infs with NaNs")
        self.t.replace([np.inf, -np.inf], np.nan, inplace=True)

        # calculate flux/dflux
        self.logger.info(f"Calculating flux/dflux for '{self.colnames.snr}' column")
        self.t[self.colnames.snr] = (
            self.t[self.colnames.flux] / self.t[self.colnames.dflux]
        )

    def _get_average_flux(self, indices=None) -> StatParams:
        self.calcaverage_sigmacutloop_np(
            self.t[self.colnames.flux].values,
            noise_arr=self.t[self.colnames.dflux].values,
            indices=indices,
            Nsigma=3.0,
            median_firstiteration=True,
        )
        return StatParams(self.statparams)

    def _get_average_mjd(self, indices=None) -> float:
        self.calcaverage_sigmacutloop_np(
            self.t[self.default_mjd_colname].values,
            indices=indices,
            Nsigma=0,
            median_firstiteration=False,
        )
        return StatParams(self.statparams).mean

    def calcaverage_errorcut_np(
        self,
        data,
        noise,
        indices=None,
        mean=None,
        Nsigma=None,
        median_flag=False,
    ):
        ix = np.asarray(self.getindices(indices))
        x = data[ix]
        dx = noise[ix]

        if Nsigma is not None and mean is not None:
            diff = np.abs(x - mean)
            good_ix = ix[diff <= Nsigma * dx]
            good_ix_bkp = deepcopy(self.statparams["ix_good"])
        else:
            good_ix = ix
            good_ix_bkp = None

        Ngood = len(good_ix)

        if Ngood > 1:
            x_good = data[good_ix]
            dx_good = noise[good_ix]
            if median_flag:
                mean = np.median(x_good)
                stdev = np.sqrt(
                    np.sum((x_good - mean) ** 2.0) / (Ngood - 1.0)
                ) / self.c4(Ngood)
                mean_err = np.median(dx_good) / np.sqrt(Ngood - 1)
            else:
                w = 1.0 / (dx_good**2.0)
                mean = np.sum(x_good * w) / np.sum(w)
                mean_err = np.sqrt(1.0 / np.sum(w))
                stdev = np.std(x_good, ddof=1)

            stdev_err = stdev / np.sqrt(2.0 * Ngood)
            X2norm = np.sum(((x_good - mean) / dx_good) ** 2.0) / (Ngood - 1.0)
        elif Ngood == 1:
            mean = data[good_ix[0]]
            mean_err = noise[good_ix[0]]
            stdev = stdev_err = X2norm = None
        else:
            mean = mean_err = stdev = stdev_err = X2norm = None

        self.statparams.update(
            {
                "ix_good": good_ix,
                "Ngood": Ngood,
                "ix_clip": AnotB(ix, good_ix),
                "Nclip": len(ix) - Ngood,
                "mean": mean,
                "stdev": stdev,
                "mean_err": mean_err,
                "stdev_err": stdev_err,
                "X2norm": X2norm,
                "Nchanged": (
                    len(not_AandB(good_ix_bkp, good_ix))
                    if good_ix_bkp is not None
                    else 0
                ),
            }
        )

        return int(Ngood < 1)

    def calcaverage_sigmacut_np(
        self,
        data,
        noise=None,
        indices=None,
        mean=None,
        stdev=None,
        Nsigma=None,
        percentile_cut=None,
        percentile_Nmin=3,
        median_flag=False,
    ):
        indices = self.getindices(indices)
        if len(indices) == 0:
            self.reset()
            self.logger.warning("No data passed for sigma cut")
            return 2

        x = data[indices]
        ix = np.asarray(indices)

        good_ix_bkp = None
        if percentile_cut is None or len(ix) <= percentile_Nmin:
            # if N-sigma cut and second iteration (i.e. we have a stdev from the first iteration), skip bad measurements
            if Nsigma is not None and stdev is not None and mean is not None:
                good_ix_bkp = deepcopy(self.statparams["ix_good"])
                good_ix = ix[np.abs(x - mean) <= Nsigma * stdev]
            else:
                good_ix = ix
        else:  # percentile clipping
            if mean is None:
                mean = np.median(x) if median_flag else np.mean(x)
            residuals = np.abs(x - mean)
            max_residual = np.percentile(residuals, percentile_cut)
            good_ix = ix[residuals < max_residual]

            if len(good_ix) < percentile_Nmin:
                sorted_residuals = np.sort(residuals)
                max_residual = sorted_residuals[percentile_Nmin - 1]
                good_ix = ix[residuals < max_residual]

        Ngood = len(good_ix)
        x_good = data[good_ix]

        if Ngood > 1:
            if median_flag:
                mean = np.median(x_good)
                stdev = np.sqrt(np.sum((x_good - mean) ** 2) / (Ngood - 1.0)) / self.c4(
                    Ngood
                )
            else:
                mean = np.mean(x_good)
                stdev = np.std(x_good, ddof=1)

            mean_err = stdev / np.sqrt(Ngood - 1.0)
            stdev_err = stdev / np.sqrt(2.0 * Ngood)
            if noise is None:
                X2norm = np.sum(((x_good - mean) / stdev) ** 2) / (Ngood - 1.0)
            else:
                dx_good = noise[good_ix]
                X2norm = np.sum(((x_good - mean) / dx_good) ** 2) / (Ngood - 1.0)
        elif Ngood == 1:
            mean = x_good[0]
            mean_err = noise[good_ix[0]] if noise is not None else None
            stdev = stdev_err = X2norm = None
        else:
            mean = mean_err = stdev = stdev_err = X2norm = None

        self.statparams.update(
            {
                "ix_good": good_ix,
                "Ngood": Ngood,
                "ix_clip": AnotB(ix, good_ix),
                "Nclip": len(ix) - Ngood,
                "mean": mean,
                "stdev": stdev,
                "mean_err": mean_err,
                "stdev_err": stdev_err,
                "X2norm": X2norm,
                "Nchanged": (
                    len(not_AandB(good_ix_bkp, good_ix))
                    if good_ix_bkp is not None
                    else 0
                ),
            }
        )

        return int(Ngood < 1)

    def ix_inrange_np(
        self,
        colnames=None,
        lowlim=None,
        uplim=None,
        indices=None,
        exclude_lowlim=False,
        exclude_uplim=False,
    ) -> List[int]:
        indices = np.asarray(self.getindices(indices))
        colnames = self.getcolnames(colnames)

        keep_mask = np.ones(len(indices), dtype=bool)

        for colname in colnames:
            values = self.t[colname].values[indices]

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

    def ix_unmasked_np(self, mask_arr, maskval=None, indices=None):
        if indices is None:
            indices = np.arange(len(mask_arr))

        sub_mask = mask_arr[indices]

        if maskval is None:
            keep = sub_mask == 0
        else:
            keep = bitmask.bitfield_to_boolean_mask(
                sub_mask.astype(int),
                ignore_flags=~maskval,
                good_mask_value=True,
            )

        return indices[keep]

    def ix_not_null_np(self, arrays: List, indices=None):
        if indices is None:
            if len(arrays) == 0:
                return np.array([], dtype=int)
            indices = np.arange(len(arrays[0]))
        else:
            indices = np.asarray(indices)

        keep_mask = np.ones(len(indices), dtype=bool)
        for arr in arrays:
            keep_mask &= pd.notnull(arr[indices])

        return indices[keep_mask]

    def calcaverage_sigmacutloop_np(
        self,
        data_arr,
        indices=None,
        noise_arr=None,
        sigmacut_flag=False,
        mask_arr=None,
        maskval=None,
        removeNaNs=True,
        Nsigma=3.0,
        Nitmax=10,
        verbose=0,
        percentile_cut_firstiteration=None,
        median_firstiteration=True,
    ):
        if noise_arr is None:
            sigmacut_flag = True
        indices = np.asarray(self.getindices(indices))
        self.reset()

        # exclude data if wanted
        if mask_arr is not None:
            Ntot = len(indices)
            indices = self.ix_unmasked_np(mask_arr, maskval=maskval, indices=indices)
            self.statparams["Nmask"] = Ntot - len(indices)
            if verbose > 1:
                print(
                    f"Keeping {len(indices)} out of {Ntot}, skipping {Ntot - len(indices)} because of masking (maskval={maskval})"
                )
        else:
            self.statparams["Nmask"] = 0

        # remove null values if wanted
        if removeNaNs:
            arrays = [data_arr]
            if noise_arr is not None:
                arrays.append(noise_arr)
            if mask_arr is not None:
                arrays.append(mask_arr)

            Ntot = len(indices)
            indices = self.ix_not_null_np(arrays, indices=indices)
            self.statparams["Nnan"] = Ntot - len(indices)
            if verbose > 1:
                print(
                    f"NaN filtering: kept {len(indices)} / {Ntot}, removed {Ntot - len(indices)}"
                )
        else:
            self.statparams["Nnan"] = 0

        for i in range(Nitmax):
            if self.statparams["converged"]:
                break

            self.statparams["i"] = i
            medianflag = median_firstiteration and (i == 0) and (Nsigma is not None)
            percentile_cut = percentile_cut_firstiteration if i == 0 else None

            if sigmacut_flag:
                errorflag = self.calcaverage_sigmacut_np(
                    data=data_arr,
                    noise=noise_arr,
                    indices=indices,
                    mean=self.statparams.get("mean"),
                    stdev=self.statparams.get("stdev"),
                    Nsigma=Nsigma,
                    median_flag=medianflag,
                    percentile_cut=percentile_cut,
                )
            else:
                errorflag = self.calcaverage_errorcut_np(
                    data=data_arr,
                    noise=noise_arr,
                    indices=indices,
                    mean=self.statparams.get("mean"),
                    Nsigma=Nsigma,
                    median_flag=medianflag,
                )

            if verbose > 2:
                print(self.statstring())

            if (
                errorflag
                or self.statparams["stdev"] is None
                or (self.statparams["stdev"] == 0.0 and sigmacut_flag)
                or self.statparams["mean"] is None
            ):
                self.statparams["converged"] = False
                break

            if Nsigma in (None, 0.0):
                self.statparams["converged"] = True
                break

            if i > 0 and self.statparams["Nchanged"] == 0 and not medianflag:
                self.statparams["converged"] = True
                break

        if not (self.statparams["converged"]):
            if self.verbose > 1:
                print("WARNING: No convergence")

        return not self.statparams["converged"]

    def merge(self, other: Self) -> Self:
        """
        Merge this LightCurve with another one and return a new LightCurve instance
        with the combined and sorted data. Used for merging two light curves with
        different filters together.

        Parameters
        ----------
        other : BaseLightCurve
            Another light curve to merge with.

        Returns
        -------
        BaseLightCurve
            New instance with merged and sorted data.
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
        Split the light curve into two new instances for filters 'o' and 'c'.
        Underlying dataframes are deepcopied and independent of self.

        Returns
        -------
        tuple
            Two new BaseLightCurve instances for filters 'o' and 'c'.
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
        Initialize a LightCurve instance.

        Parameters
        ----------
        control_index : int
            Index identifying the control light curve.
        coords : Coordinates
            Object containing coordinate information for the light curve.
        filt : str, optional
            Filter name (e.g., 'o' or 'c'). If None, includes all filters.
        verbose : bool, optional
            If True, enables verbose logging.
        """
        super().__init__(
            control_index,
            coords,
            CleanedColumnNames(),
            filt=filt,
            verbose=verbose,
        )

    def preprocess(self, flux2mag_sigmalimit=3.0):
        """
        Prepare the raw ATLAS light curve for cleaning.

        Adds a mask column if missing, sorts by MJD, removes rows with duJy=0 or uJy=NaN,
        and overwrites ATLAS magnitudes with calculated values.

        Parameters
        ----------
        flux2mag_sigmalimit : float, optional
            Sigma limit for converting flux to magnitude (default: 3.0).
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
        Add additional noise in quadrature to the dflux column.

        Parameters
        ----------
        sigma_extra : float
            Value to add in quadrature to the dflux column.
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
        """
        Flag light curve measurements based on control statistics.

        Parameters
        ----------
        flag : int, optional
            Bitmask flag for bad data (default: 0x400000).
        questionable_flag : int, optional
            Bitmask flag for questionable data (default: 0x80000).

        The following max values and their corresponding flags are used in reference to the statistics returned from the 3-sigma clipping on a single epoch across controls.

        x2_max : float, optional
            Maximum allowed chi-square value (default: 2.5).
        x2_flag : int, optional
            Bitmask flag for chi-square cut (default: 0x100).
        snr_max : float, optional
            Maximum allowed SNR (default: 3.0).
        snr_flag : int, optional
            Bitmask flag for SNR cut (default: 0x200).
        Nclip_max : int, optional
            Maximum allowed number of clipped points (default: 2).
        Nclip_flag : int, optional
            Bitmask flag for Nclip cut (default: 0x400).
        Ngood_min : int, optional
            Minimum number of good points required (default: 4).
        Ngood_flag : int, optional
            Bitmask flag for Ngood cut (default: 0x800).
        """
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
        """
        Copy flag values into the mask column for all rows.

        Parameters
        ----------
        flag_arr : array-like
            Flag(s) to apply to the mask column.
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

            good_bin_ix = binned_lc.ix_unmasked_np(
                mask_arr, maskval=previous_flags, indices=bin_ix
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
        super().__init__(
            control_index, coords, BinnedColumnNames(), filt=filt, verbose=verbose
        )
        self.mjd_bin_size = mjd_bin_size

    @property
    def default_mjd_colname(self):
        return self.colnames.mjdbin

    def _get_postprocess_cols_dict(self):
        cols_dict = super()._get_postprocess_cols_dict()
        cols_dict[self.colnames.mjdbin] = "mjd_bin"
        return cols_dict


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

    def postprocess(self):
        if self.filt is not None:
            raise RuntimeError(
                f"Filter '{self.filt}' should be None for postprocessing"
            )

        for i in self.lc_indices:
            self.get(i).postprocess()

    @property
    def colnames(self):
        return self.get_sn().colnames

    @property
    def default_mjd_colname(self):
        return self.get_sn().default_mjd_colname

    @property
    def num_controls(self):
        return sum(1 for k in self.lcs if k != 0)

    @property
    def lc_indices(self) -> List[int]:
        return list(self.lcs)

    @property
    def control_lc_indices(self) -> List[int]:
        return [i for i in self.lcs if i != 0]

    def add(self, lc: BaseLightCurve, deep: bool = True):
        if lc.control_index in self.lcs:
            self.logger.warning(
                f"Control index {lc.control_index} already exists; overwriting..."
            )
        self.update(lc, deep=deep)

    def update(self, lc: BaseLightCurve, deep: bool = False):
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

    def remove_flag(self, flag: int):
        for i in self.lc_indices:
            self.get(i).remove_flag(flag)

    def get_all_controls(self):
        all_controls = LightCurve(None, None, verbose=self.logger.verbose)
        all_controls.t = pd.concat(
            [self.get(i).t for i in self.control_lc_indices],
            ignore_index=True,
            copy=False,
        )
        return all_controls

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
        self, temp_x2_max_value: float = 20, uncertainty_cut_flag: int = 0x2
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
    def __init__(
        self, mjd_bin_size: float, filt: Optional[str] = None, verbose: bool = False
    ):
        super().__init__(filt=filt, verbose=verbose)
        self.lcs: Dict[int, BinnedLightCurve] = {}
        self.mjd_bin_size = mjd_bin_size

    def get(self, control_index: int) -> BinnedLightCurve:
        return super().get(control_index)

    def add(self, lc: BinnedLightCurve, deep: bool = False):
        return super().add(lc, deep=deep)
