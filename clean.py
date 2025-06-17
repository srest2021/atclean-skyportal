#!/usr/bin/env python

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Any, List, Optional, Self, Set, Tuple, Type

import numpy as np
import pandas as pd

from constants import CHISQUARECUTS_TABLE_COLNAMES
from lightcurve import LightCurve, Transient
from pdastro import AandB, AnotB
from utils import (
    ChiSquareCut,
    CustomCut,
    CustomLogger,
    Cut,
    CutList,
    UncertaintyCut,
    ChiSquareCut,
    UncertaintyEstimation,
    ControlLightCurveCut,
    BadDayCut,
    new_row,
)


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
        self.transient = None

    def _apply_UncertaintyEstimation(self, uncert_est: UncertaintyEstimation):
        stats_table = self.transient.get_uncert_est_stats(uncert_est)
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
            self.transient.add_noise_to_dflux(final_sigma_extra)
            self.logger.success()
            self.logger.info(
                'The extra noise was added to the uncertainties of the SN light curve and copied to the "duJy_new" column'
            )

    def _apply_ChiSquareCut(self, x2_cut: ChiSquareCut):
        # calculate contamination and loss for possible chi-square cuts in range (min_cut, max_cut)
        stats_table = ChiSquareCutsTable(
            self.transient.get_sn(), snr_bound=x2_cut.snr_bound
        )
        stats_table.calculate_table(
            start=x2_cut.min_cut,
            stop=max(x2_cut.max_value, x2_cut.max_cut),
            step=x2_cut.cut_step,
        )

        # get contamination and loss for selected cut
        contamination, loss = stats_table.get_contamination_and_loss(x2_cut.max_value)
        self.logger.info(
            f"Applying chi-square cut of {x2_cut.max_value:0.2f} with {contamination:0.2f}% contamination and {loss:0.2f}% loss"
        )

        # apply it
        self.transient.apply_cut(x2_cut)

        return stats_table

    def _apply_ControlLightCurveCut(self, controls_cut: ControlLightCurveCut):
        # TODO
        return

    def _apply_BadDayCut(self, badday_cut: BadDayCut):
        # TODO
        return

    def clean(self, cut_list: CutList, transient: Transient):
        self.transient = transient
        self.transient.match_control_mjds()

        # for cut in cut_list.iterator():
        #     method_name = f"_apply_{cut.__class__.__name__}"
        #     method = getattr(self, method_name, None)
        #     if method is not None:
        #         method(cut)
        #     else:
        #         try:
        #             self.transient.apply_cut(cut)
        #         except Exception as e:
        #             self.logger.error(
        #                 f"Error when trying to apply cut '{cut.name()}': {str(e)}\nSkipping..."
        #             )
