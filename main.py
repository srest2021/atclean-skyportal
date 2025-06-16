#!/usr/bin/env python

# define command line arguments
import argparse
from typing import Dict

from download import AtlasLightCurveDownloader, ControlCoordinatesTable
from lightcurve import LightCurve
from utils import (
    RA,
    BadDayCut,
    ChiSquareCut,
    ControlLightCurveCut,
    Coordinates,
    CustomLogger,
    CutList,
    Dec,
    UncertaintyCut,
    UncertaintyEstimation,
    parse_arg_coords,
)


def define_args(parser=None, usage=None, conflict_handler="resolve"):
    if parser is None:
        parser = argparse.ArgumentParser(usage=usage, conflict_handler=conflict_handler)

    # for downloading
    parser.add_argument("ra", type=RA, help="RA (degrees or sexagesimal)")
    parser.add_argument("dec", type=Dec, help="Dec (degrees or sexagesimal)")
    parser.add_argument(
        "-l", "--lookbacktime", default=None, type=int, help="lookback time (MJD)"
    )
    parser.add_argument(
        "--max_mjd", default=None, type=float, help="maximum MJD to download"
    )
    parser.add_argument(
        "-n",
        "--num_controls",
        default=8,
        type=int,
        help="number of control light curves per SN",
    )
    parser.add_argument(
        "-r",
        "--radius",
        default=17.0,
        type=float,
        help="radius of control light curve circle pattern around SN",
    )
    parser.add_argument(
        "--flux2mag_sigmalimit",
        default=3.0,
        type=float,
        help="sigma limit when converting flux to magnitude (magnitudes are limits when dmagnitudes are NaN)",
    )
    parser.add_argument(
        "--center_coords",
        type=parse_arg_coords,
        default=None,
        help="comma-separated RA and Dec (degrees or sexagesimal) of a nearby bright object interfering with the light curve to become center of circle pattern",
    )
    parser.add_argument(
        "--sn_min_dist",
        default=3.0,
        type=float,
        help="minimum distance (arcseconds) of control location from SN location (only used if using different circle pattern center coordinates)",
    )

    # ATLAS credentials
    parser.add_argument(
        "--atlas_username", type=str, default=None, help="ATLAS API username"
    )
    parser.add_argument(
        "--atlas_password", type=str, default=None, help="ATLAS API password"
    )

    # cleaning
    # TODO

    # miscellaneous
    parser.add_argument(
        "-v", "--verbose", type=bool, default=False, help="verbosity level"
    )

    return parser


if __name__ == "__main__":
    logger = CustomLogger()
    args = define_args().parse_args()

    # download
    control_coords_table = ControlCoordinatesTable(
        Coordinates(args.ra, args.dec),
        center_coords=args.center_coords,
        sn_min_dist=args.sn_min_dist,
        radius=args.radius,
        num_controls=args.num_controls,
        verbose=args.verbose,
    )
    downloader = AtlasLightCurveDownloader(
        args.atlas_username, args.atlas_password, verbose=args.verbose
    )
    transient_o, transient_c = downloader.download(
        control_coords_table,
        lookbacktime=args.lookbacktime,
        max_mjd=args.max_mjd,
        flux2mag_sigmalimit=args.flux2mag_sigmalimit,
    )

    # clean
    # TODO
    cut_list = CutList(verbose=args.verbose)

    # # define which cuts to apply
    # # -- remove or add from this list at will
    # cut_list.add_many(
    #     [
    #         UncertaintyEstimation(),
    #         UncertaintyCut(transient.colnames.dflux),
    #         ChiSquareCut(transient.colnames.x2),
    #         ControlLightCurveCut(),
    #         BadDayCut(),
    #     ]
    # )
