#!/usr/bin/env python

# column names for table containing coordinates info about SN and control light curves
CTRL_COORDINATES_TABLE_COLNAMES = [
    "transient_name",
    "control_index",
    "ra",
    "dec",
    "ra_offset",
    "dec_offset",
    "radius_arcsec",
    "n_detec",
    "n_detec_c",
    "n_detec_o",
]

# column names for table containing contamination and loss for a range of possible chi-square cuts
CHISQUARECUTS_TABLE_COLNAMES = [
    "PSF Chi-Square Cut",
    "N",
    "Ngood",
    "Nbad",
    "Nkept",
    "Ncut",
    "Ngood,kept",
    "Ngood,cut",
    "Nbad,kept",
    "Nbad,cut",
    "Pgood,kept",
    "Pgood,cut",
    "Pbad,kept",
    "Pbad,cut",
    "Ngood,kept/Ngood",
    "Ploss",
    "Pcontamination",
]

# column names for raw light curve returned from ATLAS API
ATLAS_API_COLNAMES = [
    "MJD",
    "m",
    "dm",
    "uJy",
    "duJy",
    "F",
    "err",
    "chi/N",
    "RA",
    "Dec",
    "x",
    "y",
    "maj",
    "min",
    "phi",
    "apfit",
    "Sky",
    "ZP",
    "Obs",
    "Mask",
]
