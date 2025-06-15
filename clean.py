#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Optional, Self, Set, Tuple, Type

import numpy as np

from lightcurve import LightCurve, Transient
from utils import (
    ChiSquareCut,
    CustomLogger,
    Cut,
    CutList,
    UncertaintyCut,
    ChiSquareCut,
    UncertaintyEstimation,
    ControlLightCurveCut,
    BadDayCut,
)


class LightCurveCleaner:
    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger(verbose=verbose)

    def _apply_UncertaintyEstimation(
        self, uncert_est: UncertaintyEstimation, transient: Transient
    ):
        # TODO
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
            if cut.can_apply_directly():
                transient = transient.apply_cut(cut)
            else:
                method_name = f"_apply_{cut.__class__.__name__}"
                method = getattr(self, method_name, None)
                if method is not None:
                    transient = method(cut, transient)
                else:
                    self.logger.warning(f"No handler for cut type: {cut.name()}")
