#!/usr/bin/env python

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Any, List, Optional, Self, Set, Tuple, Type

import numpy as np
import pandas as pd

from constants import CHISQUARECUTS_TABLE_COLNAMES
from lightcurve import BaseTransient, BinnedTransient, LightCurve, Transient
from pdastro import AandB, AnotB
from utils import CustomLogger, CutHistory, combine_flags, new_row


class ChiSquareCutsTable:
    """
    Utility class for evaluating the effects of chi-square cuts on a light curve.

    This class computes contamination and loss rates for a range of different chi-square
    cuts, classifying data as "good" and "bad" based on SNR.

    - Loss = Number of good cut measurements / Number of good measurements
    - Contamination = Number of bad kept measurements / Number of bad measurements
    """

    def __init__(self, lc: LightCurve, snr_bound: Optional[float] = 3, indices=None):
        self.logger = CustomLogger(self.__class__.__name__)
        self.t = None

        self.lc = lc
        if indices is None:
            indices = self.lc.getindices()
        self.indices = indices

        self.good_ix, self.bad_ix = self.get_goodbad_indices(snr_bound=snr_bound)

    def get_goodbad_indices(self, snr_bound: Optional[float] = 3):
        if not self.lc.colnames.snr in self.lc.t.columns:
            self.lc.calculate_snr_col()

        good_ix = self.lc.ix_inrange(
            colnames=[self.lc.colnames.snr],
            lowlim=-snr_bound,
            uplim=snr_bound,
            indices=self.indices,
        )
        bad_ix = AnotB(self.indices, good_ix)
        return good_ix, bad_ix

    def get_keptcut_indices(self, x2_max: float):
        kept_ix = self.lc.ix_inrange(
            colnames=self.lc.colnames.chisquare, uplim=x2_max, indices=self.indices
        )
        cut_ix = AnotB(self.indices, kept_ix)
        return kept_ix, cut_ix

    def calculate_row(
        self,
        x2_max: float,
        kept_ix: Optional[List[int]] = None,
        cut_ix: Optional[List[int]] = None,
    ):
        if kept_ix is None or cut_ix is None:
            kept_ix, cut_ix = self.get_keptcut_indices(x2_max)
        data = {
            "PSF Chi-Square Cut": x2_max,
            "N": len(self.indices),
            "Ngood": len(self.good_ix),
            "Nbad": len(self.bad_ix),
            "Nkept": len(kept_ix),
            "Ncut": len(cut_ix),
            "Ngood,kept": len(AandB(self.good_ix, kept_ix)),
            "Ngood,cut": len(AandB(self.good_ix, cut_ix)),
            "Nbad,kept": len(AandB(self.bad_ix, kept_ix)),
            "Nbad,cut": len(AandB(self.bad_ix, cut_ix)),
            "Pgood,kept": 100 * len(AandB(self.good_ix, kept_ix)) / len(self.indices),
            "Pgood,cut": 100 * len(AandB(self.good_ix, cut_ix)) / len(self.indices),
            "Pbad,kept": 100 * len(AandB(self.bad_ix, kept_ix)) / len(self.indices),
            "Pbad,cut": 100 * len(AandB(self.bad_ix, cut_ix)) / len(self.indices),
            "Ngood,kept/Ngood": 100
            * len(AandB(self.good_ix, kept_ix))
            / len(self.good_ix),
            "Ploss": 100 * len(AandB(self.good_ix, cut_ix)) / len(self.good_ix),
            "Pcontamination": 100 * len(AandB(self.bad_ix, kept_ix)) / len(kept_ix),
        }
        return data

    def get_contamination_and_loss(
        self,
        x2_max: float,
        kept_ix: Optional[List[int]] = None,
        cut_ix: Optional[List[int]] = None,
    ):
        if kept_ix is None or cut_ix is None:
            kept_ix, cut_ix = self.get_keptcut_indices(x2_max)

        contamination = 100 * len(AandB(self.bad_ix, kept_ix)) / len(kept_ix)
        loss = 100 * len(AandB(self.good_ix, cut_ix)) / len(self.good_ix)
        return contamination, loss

    def calculate_table(
        self,
        start: Optional[float] = 3,
        stop: Optional[float] = 50,
        step: Optional[float] = 1,
    ):
        self.logger.info(
            f"Calculating loss and contamination for chi-square cuts from {start} to {stop}"
        )
        self.t = pd.DataFrame(columns=CHISQUARECUTS_TABLE_COLNAMES)

        # for different x2 cuts decreasing from 50
        for cut in range(start, stop + 1, step):
            kept_ix, cut_ix = self.get_keptcut_indices(cut)
            percent_kept = 100 * len(kept_ix) / len(self.indices)
            if percent_kept < 10:
                # less than 10% of measurements kept, so no chi-square cuts beyond this point are valid
                continue
            row = self.calculate_row(cut, kept_ix=kept_ix, cut_ix=cut_ix)
            self.t = new_row(self.t, row)

        self.logger.success()


class LightCurveCleaner:
    """Utility class for cleaning ATLAS light curvess"""

    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger(verbose=verbose)
        self.cut_history = CutHistory(verbose=verbose)

    def apply_cut(
        self,
        transient: Transient,
        column: str,
        flag: int,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        indices: Optional[List[int]] = None,
    ) -> Transient:
        """
        Apply a value-based cut to a column in the transient, masking out-of-range data.

        Parameters
        ----------
        transient : Transient
            The transient object to apply the cut to.
        column : str
            The name of the column to cut on.
        flag : int
            The bitmask flag to apply to masked data.
        min_value : float, optional
            Minimum allowed value (inclusive).
        max_value : float, optional
            Maximum allowed value (inclusive).
        indices : list of int, optional
            Indices to consider for the cut. If None, all indices are used.

        Returns
        -------
        Transient
            The transient with the cut applied.
        """
        transient.apply_cut(
            column, flag, min_value=min_value, max_value=max_value, indices=indices
        )
        return transient

    def apply_UncertaintyCut(
        self,
        transient: Transient,
        flag: int = 0x2,
        max_value: float = 160,
    ) -> Transient:
        """
        Mask data points with uncertainty above a threshold.

        Parameters
        ----------
        transient : Transient
            The transient to apply the uncertainty cut to.
        flag : int, optional
            The bitmask flag to apply to masked data (default: 0x2).
        max_value : float, optional
            Maximum allowed uncertainty value (default: 160).
        """
        self.apply_cut(transient, transient.colnames.dflux, flag, max_value=max_value)
        self.cut_history.add_UncertaintyCut(flag=flag, max_value=max_value)
        return transient

    def apply_UncertaintyEstimation(
        self,
        transient: Transient,
        temp_x2_max_value: float = 20,
        uncertainty_cut_flag: int = 0x2,
    ) -> Transient:
        """
        Estimate and, if needed, increase uncertainties by adding systematic noise in quadrature.

        A new column, "dflux_offset_in_quadrature", containing the calculated systematic uncertainty will be added to each light curve.

        Parameters
        ----------
        transient : Transient
            The transient to update uncertainties for.
        temp_x2_max_value : float, optional
            Maximum chi-square value for temporary cut (default: 20), by which we remove egregious outliers before calculating how much noise to add.
        uncert_cut_flag : int, optional
            Bitmask flag for uncertainty cut (default: 0x2), by which we remove egregious outliers before calculating how much noise to add.

        Returns
        -------
        Transient
            The transient with an updated uncertainty column and a new "dflux_offset_in_quadrature" column.
        """
        stats_table = transient.get_uncert_est_stats(
            temp_x2_max_value=temp_x2_max_value,
            uncertainty_cut_flag=uncertainty_cut_flag,
        )
        final_sigma_extra = np.median(stats_table["sigma_extra"])
        sigma_typical_old = np.median(stats_table["median_dflux"])
        sigma_typical_new = np.sqrt(final_sigma_extra**2 + sigma_typical_old**2)
        percent_greater = 100 * (
            (sigma_typical_new - sigma_typical_old) / sigma_typical_old
        )

        self.logger.info(
            f"We can increase the typical uncertainties from {sigma_typical_old:0.2f} to {sigma_typical_new:0.2f} by adding an additional systematic uncertainty of {final_sigma_extra:0.2f} in quadrature"
        )
        self.logger.info(
            f"New typical uncertainty is {percent_greater:0.2f}% greater than old typical uncertainty"
        )

        if percent_greater >= 10:
            self.logger.info("Applying true uncertainties estimation")
            transient.add_noise_to_dflux(final_sigma_extra)
            self.logger.success()
            self.logger.info(
                'The extra noise was added to the uncertainties of the SN light curve and copied to the "duJy_new" column'
            )

        self.cut_history.add_UncertaintyEstimation(temp_x2_max_value=temp_x2_max_value)

        return transient

    def apply_ChiSquareCut(
        self,
        transient: Transient,
        flag: int = 0x1,
        max_value: float = 10,
        snr_bound: float = 3,
        table_start: int = 3,
        table_stop: int = 50,
        table_step: int = 1,
    ) -> Transient:
        """
        Apply a chi-square cut to the transient and compute contamination/loss statistics for a range of possible chi-squares.

        Parameters
        ----------
        transient : Transient
            The transient to apply the chi-square cut to.
        flag : int, optional
            Bitmask flag for masked data (default: 0x1).
        max_value : float, optional
            Maximum allowed chi-square value (default: 10).
        snr_bound : float, optional
            SNR bound for good/bad classification (default: 3).
        table_start : int, optional
            Chi-square start value for table containing contamination and loss statistics (default: 3).
        table_stop : int, optional
            Chi-square stop value for table containing contamination and loss statistics (default: 50).
        table_step : int, optional
            Step size for the range of chi-square cuts listed in the table (default: 1).

        Returns
        -------
        tuple
            The transient with the cut applied and the table containing contamination and loss statistics for the range of possible chi-squares.
        """
        # calculate contamination and loss for possible chi-square cuts in range (min_cut, max_cut)
        stats_table = ChiSquareCutsTable(
            (
                transient.get_all_controls()
                if transient.num_controls > 0
                else transient.get_sn()
            ),
            snr_bound=snr_bound,
        )
        stats_table.calculate_table(
            start=table_start,
            stop=max(max_value, table_stop),
            step=table_step,
        )

        # get exact contamination and loss for selected cut
        contamination, loss = stats_table.get_contamination_and_loss(max_value)
        self.logger.info(
            f"Applying chi-square cut of {max_value:0.2f} with {contamination:0.2f}% contamination and {loss:0.2f}% loss"
        )

        # apply it
        transient.apply_cut(transient.colnames.chisquare, flag, max_value=max_value)

        self.cut_history.add_ChiSquareCut(flag=flag, max_value=max_value)

        return transient, stats_table

    def apply_ControlLightCurveCut(
        self,
        transient: Transient,
        previous_flags: int,
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
    ) -> Transient:
        """
        Apply a set of control light curve quality cuts and flag questionable data.

        Parameters
        ----------
        transient : Transient
            The transient to apply control cuts to.
        previous_flags : int
            Any flags from previously applied cuts; used to remove bad measurements before calculating control statistics.
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

        Returns
        -------
        Transient
            The transient with control cuts applied.
        """
        transient.calculate_control_stats(previous_flags)
        transient.flag_by_control_stats(
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

        self.cut_history.add_ControlLightCurveCut(
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

        return transient

    def apply_BadDayCut(
        self,
        transient: Transient,
        previous_flags: int,
        flag: int = 0x800000,
        mjd_bin_size: float = 1.0,
        x2_max: float = 4.0,
        Nclip_max: int = 1,
        Ngood_min: int = 2,
        large_num_clipped_flag: int = 0x1000,
        small_num_unmasked_flag: int = 0x2000,
        flux2mag_sigmalimit: float = 3.0,
    ) -> tuple[Transient, BinnedTransient]:
        """
        Apply a "bad day" cut by binning data in time and flagging problematic bins.

        Parameters
        ----------
        transient : Transient
            The transient to apply the bad day cut to.
        flag : int, optional
            Bitmask flag for bad days (default: 0x800000).
        mjd_bin_size : float, optional
            Size of the time bin in MJD in days (default: 1.0).
        flux2mag_sigmalimit : float, optional
            The sigma limit used when converting flux to magnitude. Magnitudes are set as limits when their uncertainties are `NaN`.

        The following max values and flags are used in reference to the statistics returned from the 3-sigma clipping on a single time bin.

        x2_max : float, optional
            Maximum allowed chi-square value per bin (default: 4.0).
        Nclip_max : int, optional
            Maximum allowed number of clipped points per bin (default: 1).
        Ngood_min : int, optional
            Minimum number of good points per bin (default: 2).
        large_num_clipped_flag : int, optional
            Bitmask flag for bins with too many clipped points (default: 0x1000).
        small_num_unmasked_flag : int, optional
            Bitmask flag for bins with too few points (default: 0x2000).

        Returns
        -------
        tuple
            The transient and the binned transient after applying the cut.
        """
        binned_transient = transient.get_BinnedTransient(
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

        self.cut_history.add_BadDayCut(
            flag=flag,
            mjd_bin_size=mjd_bin_size,
            x2_max=x2_max,
            Nclip_max=Nclip_max,
            Ngood_min=Ngood_min,
            large_num_clipped_flag=large_num_clipped_flag,
            small_num_unmasked_flag=small_num_unmasked_flag,
        )

        return transient, binned_transient

    def clean_default(
        self,
        transient: Transient,
        uncertainty_cut: float = 160.0,
        uncertainty_cut_flag: int = 0x1,
        chi_square_cut: float = 10.0,
        chi_square_cut_flag: int = 0x2,
        controls_cut_flag=0x400000,
    ) -> tuple[Transient, BinnedTransient]:
        """
        Run the default cleaning pipeline on a transient.

        Parameters
        ----------
        transient : Transient
            The transient to clean.

        Returns
        -------
        tuple
            The cleaned transient and the corresponding binned transient.
        """
        transient = self.apply_UncertaintyCut(
            transient, flag=uncertainty_cut_flag, max_value=uncertainty_cut
        )

        transient = self.apply_UncertaintyEstimation(
            transient, uncertainty_cut_flag=uncertainty_cut_flag
        )

        # stats_table contains the contamination and loss statistics
        # for the range of possible chi-square cuts:
        transient, stats_table = self.apply_ChiSquareCut(
            transient, flag=chi_square_cut_flag, max_value=chi_square_cut
        )

        previous_flags = uncertainty_cut_flag | chi_square_cut_flag
        transient = self.apply_ControlLightCurveCut(
            transient, previous_flags, flag=controls_cut_flag
        )

        previous_flags = previous_flags | controls_cut_flag
        transient, binned_transient = self.apply_BadDayCut(transient, previous_flags)

        return transient, binned_transient, self.cut_history
