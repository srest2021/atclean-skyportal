#!/usr/bin/env python

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Optional, Self, Set, Tuple, Type

from utils import CustomLogger, combine_flags


class Cut(ABC):
    def __init__(
        self,
        column: Optional[str] = None,
        flag: Optional[int] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.column = column
        self.flag = flag

        if min_value is not None and max_value is not None and min_value >= max_value:
            raise ValueError(
                f"Min value {min_value} cannot be greater than or equal to max value {max_value}"
            )
        self.min_value = min_value
        self.max_value = max_value

    def can_apply_directly(self) -> bool:
        return (
            self.flag is not None
            and self.column is not None
            and self.column != ""
            and (self.min_value is not None or self.max_value is not None)
        )

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    def get_flags(self, return_keys=False) -> Dict[str, int] | List[int]:
        """Extract all attributes that end in '_flag' or are 'flag'."""
        flags = {
            key: value
            for key, value in vars(self).items()
            if isinstance(value, int) and (key.endswith("_flag") or key == "flag")
        }
        return flags if return_keys else list(flags.values())

    def __str__(self) -> str:
        details = []
        if self.column:
            details.append(f"column={self.column}")
        if self.min_value is not None:
            details.append(f"min_value={self.min_value}")
        if self.max_value is not None:
            details.append(f"max_value={self.max_value}")

        flags = self.get_flags(return_keys=True)
        if flags:
            for flag_name, flag_value in flags.items():
                details.append(f"{flag_name}={hex(flag_value)}")

        return (f"{self.name()}: " + ", ".join(details)) if details else self.name()


class CustomCut(Cut):
    def __init__(
        self,
        column: str,
        flag: int,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        super().__init__(
            column=column, flag=flag, min_value=min_value, max_value=max_value
        )
        if not self.can_apply_directly():
            raise ValueError("Please pass a min value, max value, or both")

    def name(self) -> str:
        """Generate a unique name based on all attributes that define this cut."""
        params = [f"column={self.column}"]
        if self.flag is not None:
            params.append(f"flag={hex(self.flag)}")
        if self.min_value is not None:
            params.append(f"min={self.min_value}")
        if self.max_value is not None:
            params.append(f"max={self.max_value}")
        return "Custom Cut (" + ", ".join(params) + ")"

    def __str__(self):
        return self.name()


class UncertaintyCut(Cut):
    def __init__(self, column: str, flag: int = 0x2, max_value: float = 160.0):
        super().__init__(column=column, flag=flag, max_value=max_value)

    @staticmethod
    def name() -> str:
        return "Uncertainty Cut"


class UncertaintyEstimation(Cut):
    def __init__(self, temp_x2_max_value: float = 20, uncert_cut_flag: int = 0x2):
        super().__init__()
        self.temp_x2_max_value = temp_x2_max_value
        self.uncert_cut_flag = uncert_cut_flag

    @staticmethod
    def name() -> str:
        return "Uncertainty Estimation"


class ChiSquareCut(Cut):
    def __init__(
        self,
        column: str,
        flag: int = 0x1,
        max_value: float = 10,
        snr_bound: float = 3,
        min_cut: int = 3,
        max_cut: int = 50,
        cut_step: int = 1,
        use_pre_mjd0_lc: bool = False,
    ):
        super().__init__(column=column, flag=flag, max_value=max_value)

        self.snr_bound = snr_bound

        if min_cut >= max_cut:
            raise ValueError(
                f"Min cut {min_cut} cannot be greater than or equal to max cut {max_cut}"
            )
        if cut_step > (max_cut - min_cut):
            raise ValueError(
                f"Cut step {cut_step} cannot be greater than the difference between max and min cut {max_cut - min_cut}"
            )
        self.min_cut = min_cut
        self.max_cut = max_cut
        self.cut_step = cut_step

        self.use_pre_mjd0_lc = use_pre_mjd0_lc

    @staticmethod
    def name() -> str:
        return "Chi-Square Cut"


class ControlLightCurveCut(Cut):
    def __init__(
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
        super().__init__(flag=flag)

        if flag == questionable_flag:
            raise ValueError(
                f"Bad measurements flag {flag} cannot be equal to questionable measurements flag {questionable_flag}"
            )
        self.questionable_flag = questionable_flag

        self.x2_max = x2_max
        self.x2_flag = x2_flag

        self.snr_max = snr_max
        self.snr_flag = snr_flag

        self.Nclip_max = Nclip_max
        self.Nclip_flag = Nclip_flag

        self.Ngood_min = Ngood_min
        self.Ngood_flag = Ngood_flag

    @staticmethod
    def name() -> str:
        return "Control Light Curve Cut"


class BadDayCut(Cut):
    def __init__(
        self,
        flag: int = 0x800000,
        mjd_bin_size: float = 1.0,
        x2_max: float = 4.0,
        Nclip_max: int = 1,
        Ngood_min: int = 2,
        ixclip_flag: int = 0x1000,
        smallnum_flag: int = 0x2000,
    ):
        super().__init__(flag=flag)
        self.mjd_bin_size = mjd_bin_size
        self.x2_max = x2_max
        self.Nclip_max = Nclip_max
        self.Ngood_min = Ngood_min
        self.ixclip_flag = ixclip_flag
        self.smallnum_flag = smallnum_flag

    @staticmethod
    def name() -> str:
        return "Bad Day Cut"


class CutList:
    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger(verbose=verbose)
        self.list: Dict[str, Cut] = {}

    def add(self, cut: Cut):
        if cut.name() in self.list:
            self.logger.warning(
                f"Cut by the name {cut.name()} already exists; overwriting", dots=True
            )
        self.list[cut.name()] = cut

    def get(self, name: str) -> Cut | None:
        if not name in self.list:
            return None
        return self.list[name]

    def remove(self, name: str):
        if self.has(name):
            del self.list[name]

    def remove_many(self, names: List[str]):
        for name in names:
            self.remove(name)

    def remove_by_flag(self, flag: int):
        """
        Removes any Cut object from self.list that has a matching Cut.flag value.
        """
        to_remove = [
            name
            for name, cut in self.list.items()
            if not isinstance(cut, UncertaintyEstimation)
            and cut.flag is not None
            and cut.flag == flag
        ]
        for name in to_remove:
            del self.list[name]

    def has(self, name: str):
        return name in self.list

    def can_apply_directly(self, name: str):
        return self.list[name].can_apply_directly()

    def get_flag_duplicates(self) -> List[int]:
        if len(self.list) < 1:
            return

        unique_flags = set()
        duplicate_flags = []

        for cut in self.list.values():
            if isinstance(cut, UncertaintyEstimation):
                continue

            flags = cut.get_flags()

            for flag in flags:
                if flag in unique_flags:
                    duplicate_flags.append(flag)
                else:
                    unique_flags.add(flag)

        return duplicate_flags

    def get_custom_cuts(self) -> Dict[str, CustomCut]:
        custom_cuts = {}
        for name, cut in self.list.items():
            if isinstance(cut, CustomCut):
                custom_cuts[name] = cut
        return custom_cuts

    def get_all_flags(self):
        mask = 0
        for cut in self.list.values():
            if not isinstance(cut, UncertaintyEstimation):
                flags = cut.get_flags()
                if len(flags) > 0:
                    combined_flags = combine_flags(flags)
                    mask = mask | combined_flags
        return mask

    def get_all_default_flags(self):
        mask = 0
        for cut in self.list.values():
            if not isinstance(cut, UncertaintyEstimation) and cut.flag is not None:
                mask = mask | cut.flag
        return mask

    def get_previous_flags(self, current_cut_name: str):
        skip_names: List = (
            [
                UncertaintyEstimation.name(),
                BadDayCut.name(),
            ]
            + list(self.get_custom_cuts().keys())
            + [
                ControlLightCurveCut.name(),
                ChiSquareCut.name(),
                UncertaintyCut.name(),
            ]
        )

        try:
            current_cut_index = skip_names.index(current_cut_name)
        except:
            raise ValueError(f"No cut by name {current_cut_name} found")
        skip_names = skip_names[: current_cut_index + 1]
        mask = 0
        for name in self.list:
            flag = self.list[name].flag
            if not name in skip_names and flag is not None:
                mask = mask | flag
        return mask

    def __str__(self):
        output = []
        for name in self.list:
            output.append("• " + self.list[name].__str__())
        return "\n".join(output)
