#!/usr/bin/env python

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Any, List, Optional, Self, Set, Tuple, Type

import numpy as np
import pandas as pd

from constants import CHISQUARECUTS_TABLE_COLNAMES
from lightcurve import BaseTransient, BinnedTransient, LightCurve, Transient
from pdastro import AandB, AnotB
from utils import CustomLogger, combine_flags, new_row


class ChiSquareCutsTable:
    def __init__(self, lc: LightCurve, snr_bound: Optional[float] = 3, indices=None):
        self.logger = CustomLogger(self.__class__.__name__)
        self.t = None

        self.lc = deepcopy(lc)
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
    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger(verbose=verbose)

    def apply_cut(
        self,
        transient: BaseTransient,
        column: str,
        flag: int,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        indices: Optional[List[int]] = None,
    ) -> BaseTransient:
        transient.apply_cut(
            column, flag, min_value=min_value, max_value=max_value, indices=indices
        )
        return transient

    def apply_UncertaintyCut(
        self,
        transient: Transient,
        flag: int = 0x2,
        max_value: float = 160,
    ):
        self.apply_cut(transient, transient.colnames.dflux, flag, max_value=max_value)

    def apply_UncertaintyEstimation(
        self,
        transient: Transient,
        temp_x2_max_value: float = 20,
        uncert_cut_flag: int = 0x2,
    ) -> Transient:
        stats_table = transient.get_uncert_est_stats(
            temp_x2_max_value=temp_x2_max_value, uncert_cut_flag=uncert_cut_flag
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
        # calculate contamination and loss for possible chi-square cuts in range (min_cut, max_cut)
        stats_table = ChiSquareCutsTable(transient.get_sn(), snr_bound=snr_bound)
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
        return transient

    def apply_BadDayCut(
        self,
        transient: Transient,
        flag: int = 0x800000,
        mjd_bin_size: float = 1.0,
        x2_max: float = 4.0,
        Nclip_max: int = 1,
        Ngood_min: int = 2,
        ixclip_flag: int = 0x1000,
        smallnum_flag: int = 0x2000,
    ) -> tuple[Transient, BinnedTransient]:
        # TODO
        binned_transient = BinnedTransient(
            filt=transient.filt, verbose=transient.logger.verbose
        )

        return transient, binned_transient

    def clean_default(self, transient: Transient) -> tuple[Transient, BinnedTransient]:
        transient = self.apply_UncertaintyCut(transient)

        transient = self.apply_UncertaintyEstimation(transient)

        # stats_table contains the contamination and loss statistics
        # for the range of possible chi-square cuts:
        transient, stats_table = self.apply_ChiSquareCut(transient)

        transient = self.apply_ControlLightCurveCut(transient, previous_flags)

        transient, binned_transient = self.apply_BadDayCut(transient)

        return transient, binned_transient
