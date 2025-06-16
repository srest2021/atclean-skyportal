#!/usr/bin/env python

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Any, List, Optional, Self, Set, Tuple, Type

import numpy as np
import pandas as pd

from constants import LIMCUTS_COLNAMES
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


class LimCutsTable:
    def __init__(self, lc: LightCurve, snr_bound, indices=None):
        self.logger = CustomLogger(self.__class__.__name__)
        self.t = None

        self.lc = deepcopy(lc)
        if indices is None:
            indices = self.lc.getindices()
        self.indices = indices

        self.good_ix, self.bad_ix = self.get_goodbad_indices(snr_bound)

    def get_goodbad_indices(self, snr_bound):
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

    def get_keptcut_indices(self, x2_max):
        kept_ix = self.lc.ix_inrange(
            colnames=self.lc.colnames.chisquare, uplim=x2_max, indices=self.indices
        )
        cut_ix = AnotB(self.indices, kept_ix)
        return kept_ix, cut_ix

    def calculate_row(self, x2_max, kept_ix=None, cut_ix=None):
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

    def calculate_table(self, cut_start, cut_stop, cut_step):
        self.logger.info(
            f"Calculating loss and contamination for chi-square cuts from {cut_start} to {cut_stop}"
        )
        self.t = pd.DataFrame(columns=LIMCUTS_COLNAMES)

        # for different x2 cuts decreasing from 50
        for cut in range(cut_start, cut_stop + 1, cut_step):
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

    def _apply_UncertaintyEstimation(
        self, uncert_est: UncertaintyEstimation, transient: Transient
    ):
        stats = transient.get_uncert_est_stats(uncert_est)
        final_sigma_extra = np.median(stats["sigma_extra"])

        sigma_typical_old = np.median(stats["median_dflux"])
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

    def _apply_ChiSquareCut(self, x2_cut, transient: Transient, snr_bound: float = 3.0):
        limcuts = LimCutsTable(transient.get_sn(), snr_bound)
        # TODO
        return transient, limcuts

    def _apply_ControlLightCurveCut(
        self, controls_cut: ControlLightCurveCut, transient: Transient
    ):
        # TODO
        return transient

    def _apply_BadDayCut(self, badday_cut: BadDayCut, transient: Transient):
        # TODO
        return transient

    def clean(self, cut_list: CutList, transient: Transient):
        transient.match_control_mjds()

        for cut in cut_list.iterator():
            if isinstance(cut, CustomCut):
                transient = transient.apply_cut(cut)
            else:
                method_name = f"_apply_{cut.__class__.__name__}"
                method = getattr(self, method_name, None)
                if method is not None:
                    transient = method(cut, transient)
                else:
                    self.logger.warning(f"No handler for cut type: {cut.name()}")
