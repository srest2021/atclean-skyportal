#!/usr/bin/env python

from typing import List, Optional
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

    def __init__(
        self,
        lc: LightCurve,
        snr_bound: Optional[float] = 3,
        indices=None,
        verbose: bool = False,
    ):
        """
        Initialize the ChiSquareCutsTable object.

        :param lc: The LightCurve instance to analyze.
        :param snr_bound: SNR threshold for separating good vs. bad data (default: 3).
        :param indices: Optional subset of indices to consider (default: all).
        :param verbose: Enable verbose logging.
        """
        self.logger = CustomLogger(verbose=verbose)
        self.t = None

        self.lc = lc
        self.indices = self.lc.getindices(indices)
        self.good_ix, self.bad_ix = self.get_goodbad_indices(snr_bound=snr_bound)

    def get_goodbad_indices(self, snr_bound: Optional[float] = 3):
        """
        Get indices of good and bad data based on SNR threshold.

        :param snr_bound: SNR threshold separating good vs. bad data.
        :return: Tuple (good_ix, bad_ix)
        """
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
        """
        Determine which measurements are kept or cut by a chi-square threshold.

        :param x2_max: Chi-square threshold.
        :return: Tuple (kept_ix, cut_ix)
        """
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
        """
        Compute statistics (e.g., contamination and loss) for a single chi-square cut.

        :param x2_max: Chi-square threshold.
        :param kept_ix: Optional list of indices kept by the cut. Otherwise, will be calculated.
        :param cut_ix: Optional list of indices cut by the cut. Otherwise, will be calculated.
        :return: Dictionary with computed statistics.
        """
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
        """
        Compute contamination and loss percentages for a single chi-square cut.

        :param x2_max: Chi-square threshold.
        :param kept_ix: Optional list of indices kept by the cut. Otherwise, will be calculated.
        :param cut_ix: Optional list of indices cut by the cut. Otherwise, will be calculated.
        :return: Tuple (contamination_percent, loss_percent)
        """
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
        """
        Generate a table of cut statistics (e.g., contamination and loss) over a range of chi-square thresholds.

        :param start: Starting value of possible range of chi-square cuts (default: 3).
        :param stop: Final value of possible range of chi-square cuts (default: 50).
        :param step: Step size between cuts (default: 1).
        """
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

    def __str__(self):
        return self.t.to_string()


class LightCurveCleaner:
    """
    Utility class for cleaning ATLAS light curves
    """

    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger(verbose=verbose)
        self.cut_history = CutHistory(verbose=verbose)

    def reset(self):
        """
        Clear any prior cut history.
        """
        self.cut_history = CutHistory(verbose=self.logger.verbose)

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
        Apply a flag across all light curves to rows where the column value is outside the allowed range.

        :param column: Name of the column to cut on.
        :param flag: Bitmask flag to apply.
        :param min_value: Minimum allowed value.
        :param max_value: Maximum allowed value.
        :param indices: Optional subset of rows to check.
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
        Mask measurements with uncertainty greater than `max_value`.

        :param transient: Transient to modify.
        :param flag: Bitmask flag to apply.
        :param max_value: Maximum allowed uncertainty in `duJy`.
        :return: Transient with uncertainty cut applied.
        """
        self.logger.info(f"\nApplying uncertainty cut of {max_value}")

        transient = self.apply_cut(
            transient, transient.colnames.dflux, flag, max_value=max_value
        )

        percent_cut = transient.get_sn().get_percent_flagged(flag=flag)
        self.cut_history.add_UncertaintyCut(
            flag=flag, max_value=max_value, percent_cut=percent_cut
        )

        return transient

    def apply_UncertaintyEstimation(
        self,
        transient: Transient,
        temp_x2_max_value: float = 20,
        uncertainty_cut_flag: int = 0x2,
    ) -> Transient:
        """
        Estimate true uncertainties based on control light curves.

        :param transient: Transient to update.
        :param temp_x2_max_value: Temporary PSF chi-square upper bound for filtering out egregious outliers.
        :param uncertainty_cut_flag: Flag used in the previous uncertainty cut for filtering out egregious outliers.
        :return: Updated transient with true uncertainties in dflux column and the `sigma_extra` we added stored in a new column `dflux_offset_in_quadrature`.
        """
        self.logger.info(f"\nApplying true uncertainties estimation")

        if (
            transient.colnames.has("dflux_offset")
            and transient.colnames.dflux_offset in transient.get_sn().t.columns
        ):
            transient.remove_noise_from_dflux()

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

        self.cut_history.add_UncertaintyEstimation(
            final_sigma_extra=final_sigma_extra, temp_x2_max_value=temp_x2_max_value
        )

        return transient

    def apply_ChiSquareCut(
        self,
        transient: Transient,
        flag: int = 0x1,
        max_value: float = 10.0,
        snr_bound: float = 3.0,
        table_start: int = 3,
        table_stop: int = 50,
        table_step: int = 1,
    ) -> tuple[Transient, ChiSquareCutsTable]:
        """
        Mask measurements with PSF chi-square greater than `max_value`.
        Return updated transient and table of contamination/loss for a range of possible chi-square cuts.

        :param transient: Transient to apply cut to.
        :param flag: Bitmask flag to apply to bad chi-square measurements.
        :param max_value: Max allowed chi-square value.
        :param snr_bound: SNR threshold to define good vs. bad points.
        :param table_start: Starting chi-square value for stats table.
        :param table_stop: Ending chi-square value for stats table.
        :param table_step: Step size for chi-square sweep.
        :return: Tuple (transient, ChiSquareCutsTable instance)
        """
        self.logger.info(f"\nApplying chi-square cut of {max_value}")
        # calculate contamination and loss for possible chi-square cuts in range (min_cut, max_cut)
        stats_table = ChiSquareCutsTable(
            (
                transient.get_all_controls()
                if transient.num_controls > 0
                else transient.get_sn()
            ),
            snr_bound=snr_bound,
            verbose=self.logger.verbose,
        )
        stats_table.calculate_table(
            start=table_start,
            stop=table_stop,
            step=table_step,
        )

        # get the exact contamination and loss for the selected cut
        # they are not getting returned right now because `stats_table` will have contamination and loss for the chi-square cuts in the range (`table_start`, `table_stop`)
        contamination, loss = stats_table.get_contamination_and_loss(max_value)
        self.logger.info(
            f"Selected chi-square cut of {max_value:0.2f} has {contamination:0.2f}% contamination and {loss:0.2f}% loss"
        )

        # apply it
        transient.apply_cut(transient.colnames.chisquare, flag, max_value=max_value)

        percent_cut = transient.get_sn().get_percent_flagged(flag=flag)
        self.cut_history.add_ChiSquareCut(
            flag=flag, max_value=max_value, percent_cut=percent_cut
        )

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
        Flag SN and control epochs based on statistics from control light curves.

        :param transient: Transient to modify.
        :param previous_flags: Combined bitmask flags of previous cuts which we use to exclude bad control measurements from the sigma clipping.
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
        :return: Updated transient with control-informed mask applied.
        """
        self.logger.info(f"\nApplying control light curve cut")

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

        percent_cut = transient.get_sn().get_percent_flagged(flag=flag)
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
            percent_cut=percent_cut,
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
        Identify and flag "bad days" using binned statistics across control light curves.

        :param transient: Transient to clean.
        :param previous_flags: Combined bitmask flags of previous cuts which we use to exclude bad measurements from the binning.
        :param flag: Flag to apply to bins with invalid or bad data.
        :param mjd_bin_size: Size of MJD bins in days.
        :param x2_max: Maximum chi-square allowed for a good bin.
        :param Nclip_max: Maximum allowed clipped points in a bin.
        :param Ngood_min: Minimum number of good points required in a bin.
        :param large_num_clipped_flag: Flag for excessive clipping.
        :param small_num_unmasked_flag: Flag for insufficient unmasked measurements.
        :param flux2mag_sigmalimit: Sigma threshold for converting flux to mag in final binned LC.
        :return: Tuple (cleaned transient, binned transient).
        """
        self.logger.info(
            f"\nApplying bad day cut and binning with MJD bin size of {mjd_bin_size} days"
        )

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

        percent_cut = binned_transient.get_sn().get_percent_flagged(flag=flag)
        self.cut_history.add_BadDayCut(
            flag=flag,
            mjd_bin_size=mjd_bin_size,
            x2_max=x2_max,
            Nclip_max=Nclip_max,
            Ngood_min=Ngood_min,
            large_num_clipped_flag=large_num_clipped_flag,
            small_num_unmasked_flag=small_num_unmasked_flag,
            percent_cut=percent_cut,
        )

        return transient, binned_transient

    def apply_all_default(
        self,
        transient: Transient,
        uncertainty_cut: float = 160.0,
        uncertainty_cut_flag: int = 0x2,
        chi_square_cut: float = 10.0,
        chi_square_cut_flag: int = 0x1,
        controls_cut_flag: int = 0x400000,
        bad_day_cut_flag: int = 0x800000,
        mjd_bin_size: float = 1.0,
    ) -> tuple[Transient, BinnedTransient, CutHistory]:
        """
        Run the default cleaning pipeline on a transient.

        :param transient: The transient to clean.
        :param uncertainty_cut: Maximum allowed uncertainty.
        :param uncertainty_cut_flag: Bitmask flag to apply for uncertainty cut.
        :param chi_square_cut: Maximum allowed PSF chi-square value.
        :param chi_square_cut_flag: Bitmask flag to apply for chi-square cut.
        :param controls_cut_flag: Bitmask flag to apply for bad epochs in control light curve cut.
        :param bad_day_cut_flag: Bitmask flag to apply for bad days in bad day cut / binning.
        :param mjd_bin_size: Size of MJD bins for bad day cut.
        :returns: Tuple(transient, binned_transient, cut_history)
        """
        self.reset()

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
        transient, binned_transient = self.apply_BadDayCut(
            transient, previous_flags, flag=bad_day_cut_flag, mjd_bin_size=mjd_bin_size
        )

        return transient, binned_transient, self.cut_history
