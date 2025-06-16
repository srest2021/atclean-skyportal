#!/usr/bin/env python

from dataclasses import dataclass
import io
import re
import time
from typing import Dict, List, Optional, Type
import os, sys, requests, argparse, configparser, math
import pandas as pd
import numpy as np
from getpass import getpass
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time

from constants import CTRL_COORDINATES_COLNAMES
from lightcurve import LightCurve, Transient
from utils import ATLAS_API_COLNAMES, Coordinates, CustomLogger, new_row


def build_payload(coords: Coordinates, min_mjd: float, max_mjd: float) -> Dict:
    return {
        "ra": coords.get_RA_str(),
        "dec": coords.get_Dec_str(),
        "send_email": False,
        "mjd_min": min_mjd,
        "mjd_max": max_mjd,
    }


def query_atlas(
    headers, coords: Coordinates, min_mjd: float, max_mjd: float, verbose: bool = False
) -> pd.DataFrame:
    logger = CustomLogger(verbose=verbose)

    baseurl = "https://fallingstar-data.com/forcedphot"
    task_url = None
    while not task_url:
        with requests.Session() as s:
            resp = s.post(
                f"{baseurl}/queue/",
                headers=headers,
                data=build_payload(coords, min_mjd, max_mjd),
            )
            if resp.status_code == 201:
                task_url = resp.json()["url"]
                logger.info(f"Task url: {task_url}")
            elif resp.status_code == 429:
                message = resp.json()["detail"]
                logger.info(f"{resp.status_code} {message}")
                t_sec = re.findall(r"available in (\d+) seconds", message)
                t_min = re.findall(r"available in (\d+) minutes", message)
                if t_sec:
                    waittime = int(t_sec[0])
                elif t_min:
                    waittime = int(t_min[0]) * 60
                else:
                    waittime = 10
                logger.info(f"Waiting {waittime} seconds")
                time.sleep(waittime)
            else:
                logger.error(f"{resp.status_code}")
                logger.info(resp.text)
                sys.exit()

    result_url = None
    taskstarted_printed = False

    logger.info("Waiting for job to start...")
    while not result_url:
        with requests.Session() as s:
            resp = s.get(task_url, headers=headers)
            if resp.status_code == 200:
                if not (resp.json()["finishtimestamp"] is None):
                    result_url = resp.json()["result_url"]
                    logger.success(
                        f"Task is complete with results available at {result_url}"
                    )
                    break
                elif resp.json()["starttimestamp"]:
                    if not taskstarted_printed:
                        logger.info(
                            f"Task is running (started at {resp.json()['starttimestamp']})"
                        )
                        taskstarted_printed = True
                    time.sleep(2)
                else:
                    # print(f"Waiting for job to start (queued at {resp.json()['timestamp']})")
                    time.sleep(4)
            else:
                logger.error(f"{resp.status_code}")
                logger.info(resp.text)
                sys.exit()

    with requests.Session() as s:
        if result_url is None:
            logger.warning("Empty light curve (no data within the queried MJD range)")
            dfresult = pd.DataFrame(columns=ATLAS_API_COLNAMES)
        else:
            result = s.get(result_url, headers=headers).text
            dfresult = pd.read_csv(io.StringIO(result.replace("###", "")), sep="\s+")

    return dfresult


class ControlCoordinatesTable:
    def __init__(
        self,
        sn_coords: Coordinates,
        center_coords: Optional[Coordinates] = None,
        sn_min_dist: float = 3,
        radius: float = 17,
        num_controls: int = 8,
        verbose: bool = False,
    ):
        self.logger = CustomLogger(verbose=verbose)

        self.t: Optional[pd.DataFrame] = None

        self.sn_coords = sn_coords
        self.center_coords = center_coords
        self.sn_min_dist = sn_min_dist
        self.radius = radius
        self.num_controls = num_controls

        # are we getting controls around SN or around a different location?
        self._closebright = center_coords is not None

        self.construct()

    def update_filt_lens(self, control_index: int, o_len: int, c_len: int):
        if self.t is None:
            raise RuntimeError("Table (self.t) cannot be None")

        indices = np.where(self.t["control_index"] == control_index)[0]
        if len(indices) == 0:
            raise RuntimeError(
                f"Cannot update row in control coordinates table for control index {control_index}: no matching rows."
            )
        elif len(indices) > 1:
            raise RuntimeError(
                f"Cannot update row in control coordinates table for control index {control_index}: duplicate rows."
            )
        index = indices[0]

        # update corresponding row in table with total and filter counts
        self.t.at[index, "n_detec"] = o_len + c_len
        self.t.at[index, f"n_detec_o"] = o_len
        self.t.at[index, f"n_detec_c"] = c_len

    def add_row(
        self,
        control_index: int,
        coords: Coordinates,
        ra_offset: float | Angle = 0.0,
        dec_offset: float | Angle = 0.0,
        radius: float | Angle = 0.0,
        n_detec: int = 0,
        filt_lens: Optional[Dict[str, int]] = None,
        transient_name: str = str(np.nan),
    ):
        row = {
            "tnsname": transient_name,
            "control_index": control_index,
            "ra": coords.get_RA_str(),
            "dec": coords.get_Dec_str(),
            "ra_offset": (
                f"{ra_offset.degree:0.14f}"
                if isinstance(ra_offset, Angle)
                else ra_offset
            ),
            "dec_offset": (
                f"{dec_offset.degree:0.14f}"
                if isinstance(dec_offset, Angle)
                else dec_offset
            ),
            "radius_arcsec": radius.arcsecond if isinstance(radius, Angle) else radius,
            "n_detec": n_detec,
        }

        if not filt_lens is None:
            for filt in filt_lens:
                row[f"n_detec_{filt}"] = filt_lens[filt]

        self.t = new_row(self.t, row)

    def calculate_and_add_row(self, control_index: int):
        angle = Angle(control_index * 360.0 / self.num_controls, u.degree)

        # calculate ra
        ra_distance = Angle(self.radius.degree * math.cos(angle.radian), u.degree)
        ra_offset = Angle(
            ra_distance.degree * (1.0 / math.cos(self.center_coords.dec.angle.radian)),
            u.degree,
        )
        ra = Angle(self.center_coords.ra.angle.degree + ra_offset.degree, u.degree)

        # calculate dec
        dec_offset = Angle(self.radius.degree * math.sin(angle.radian), u.degree)
        dec = Angle(self.center_coords.dec.angle.degree + dec_offset.degree, u.degree)

        coords = Coordinates(ra, dec)

        if self._closebright:
            # check to see if control light curve location is within minimum distance from SN location
            offset_sep = self.sn_coords.get_distance(coords).arcsecond
            if offset_sep < self.sn_min_dist:
                self.logger.warning(
                    f'Control light curve {control_index:3d} too close to SN location ({offset_sep}" away) with minimum distance to SN as {self.sn_min_dist}"; skipping'
                )
                return

        self.add_row(
            control_index,
            coords,
            ra_offset=ra_offset,
            dec_offset=dec_offset,
            radius=self.radius,
        )

    def construct(self, transient_name: str = "unnamed_transient"):
        if self.center_coords is None:
            self.logger.info(
                f'Setting circle pattern of {self.num_controls} control light curves around SN location with radius of {self.radius}" from center'
            )

            # center coordinates are the SN location
            self.center_coords = self.sn_coords

            self.add_row(0, self.sn_coords, transient_name=transient_name)

        else:
            self.logger.info(
                f'Setting circle pattern of {self.num_controls} control light curves around center location {self.center_coords} with radius of {self.radius}" and minimum {self.sn_min_dist}" distance from SN'
            )
            # circle pattern radius is distance between SN and close bright object
            self.radius = self.sn_coords.get_distance(self.center_coords)

            self.add_row(
                0,
                self.sn_coords,
                ra_offset=np.nan,
                dec_offset=np.nan,
                radius=self.radius,
                transient_name=transient_name,
            )

        # add row for each control light curve
        for i in range(1, self.num_controls + 1):
            self.calculate_and_add_row(i)

        self.logger.success(
            f"Control light curve coordinates generated: \n{self.__str__()}"
        )

    def iterator(self, include_sn: bool = False):
        """
        Yields tuples of (control_index, coordinates) from the control coordinates table.

        :param include_sn (bool): If True, include the first row (SN). Defaults to False.
        """
        if self.t is None:
            raise RuntimeError("ControlCoordinatesTable is empty (self.t is None)")

        df = self.t if include_sn else self.t.iloc[1:]
        for _, row in df.iterrows():
            yield row["control_index"], Coordinates(row["ra"], row["dec"])


class AtlasAuthenticator:
    @staticmethod
    def authenticate(
        username: str, password: str, verbose: bool = False
    ) -> Dict[str, str]:
        logger = CustomLogger(verbose=verbose)
        logger.info("Connecting to ATLAS API", newline=True)
        resp = requests.post(
            url=f"https://fallingstar-data.com/forcedphot/api-token-auth/",
            data={"username": username, "password": password},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Authentication failed: {resp.status_code}")
        token = resp.json()["token"]
        logger.info(f"Token: {token}")
        headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
        return headers


class AtlasLightCurveDownloader:
    def __init__(
        self,
        atlas_username: str,
        atlas_password: str,
        verbose: bool = False,
    ):
        self.logger = CustomLogger(verbose=verbose)
        self.headers = AtlasAuthenticator.authenticate(atlas_username, atlas_password)
        if self.headers is None:
            raise RuntimeError("No token header")

    def download_lc(
        self,
        coords: Coordinates,
        lookbacktime: Optional[float] = None,
        max_mjd: Optional[float] = None,
    ) -> pd.DataFrame:
        if lookbacktime is not None:
            min_mjd = float(Time.now().mjd - lookbacktime)
        else:
            min_mjd = 50000.0
        if not max_mjd:
            max_mjd = float(Time.now().mjd)

        self.logger.info(
            f"Downloading ATLAS light curve at {coords} from {min_mjd} MJD to {max_mjd} MJD"
        )
        if min_mjd > max_mjd:
            raise RuntimeError(f"max MJD {max_mjd} cannot be than min MJD {min_mjd}.")

        while True:
            try:
                result = query_atlas(
                    self.headers,
                    coords.ra.angle.degree,
                    coords.dec.angle.degree,
                    min_mjd,
                    max_mjd,
                    verbose=self.logger.verbose,
                )
                return result
            except Exception as e:
                self.logger.warning("Exception caught: " + str(e))
                self.logger.info("Trying again in 20 seconds! Waiting...")
                time.sleep(20)
                continue

    def make_lc(
        self,
        control_index: int,
        coords: Coordinates,
        lookbacktime: Optional[float] = None,
        max_mjd: Optional[float] = None,
        flux2mag_sigmalimit: float = 3.0,
    ):
        result = self.download_lc(coords, lookbacktime=lookbacktime, max_mjd=max_mjd)
        lc = LightCurve(control_index, coords, verbose=self.logger.verbose)
        lc.set(result, flux2mag_sigmalimit=flux2mag_sigmalimit)
        return lc

    def download(
        self,
        control_coords_table: ControlCoordinatesTable,
        lookbacktime: Optional[float] = None,
        max_mjd: Optional[float] = None,
        flux2mag_sigmalimit: float = 3.0,
    ) -> tuple[Transient, Transient]:
        # make a multi-filter transient object
        transient = Transient(verbose=self.logger.verbose)

        for control_index, coords in control_coords_table.iterator():
            self.logger.info(
                f"Making light curve for control index {control_index}", newline=True
            )
            lc = self.make_lc(
                control_index,
                coords,
                lookbacktime=lookbacktime,
                max_mjd=max_mjd,
                flux2mag_sigmalimit=flux2mag_sigmalimit,
            )
            transient.add(lc, deep=False)

            self.logger.info("Updating control coordinates table with filter counts")
            control_coords_table.update_filt_lens(control_index, *lc.get_filt_lens())

            self.logger.success()

        # control_coords_table now contains all control coordinates, filter counts, and other info
        # -- can return it if needed!

        # return a transient object for each filter, o and c
        return transient.split_by_filt()
