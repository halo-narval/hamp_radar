import numpy as np
import xarray as xr
from typing import Optional


def get_decoders(ppar: Optional[xr.Dataset]):
    """
    Decoders for IQ data as in Meteorological Ka-Band Cloud Radar MIRA35 Manual,
    section 2.3.3.2 'Embedded chain type 2; Data chain'. Note these decoders are
    specific to the Ka radar currently in operation on HALO.
    (last checked: 24th Septermber 2024).
    """
    return {
        "SRVI": decode_srvi,
        "SNRD": decode_moment("SNRD"),
        "VELD": decode_moment("VELD"),
        "HNED": decode_moment("HNED"),
        "RMSD": decode_moment("RMSD"),
        "FFTD": decode_iq(ppar),
    }


def decode_ppar(rawdata):
    rawdata = rawdata[0]  # PPAR subblock is only 1 "frame"
    return {
        "prf": (
            (),
            rawdata[0:4].view("<i4")[0],
            {"long_name": "pulse repetition frequency"},
        ),
        "pdr": (
            (),
            rawdata[4:8].view("<i4")[0],
            {"long_name": "pulse duration"},
        ),
        "sft": (
            (),
            rawdata[8:12].view("<i4")[0],
            {"long_name": "FFT Length"},
        ),
        "avc": (
            (),
            rawdata[12:16].view("<i4")[0],
            {"long_name": "number of spectral (in-coherent) averages"},
        ),
        "ihp": (
            (),
            rawdata[16:20].view("<i4")[0],
            {"long_name": "number of lowest range gate (for moment estimation)"},
        ),
        "chg": (
            (),
            rawdata[20:24].view("<i4")[0],
            {"long_name": "count of gates (for moment estimation)"},
        ),
        "pol": (
            (),
            rawdata[24:28].view("<i4")[0],
            {"long_name": "on/off polarimetric measurements"},
        ),
        "att": (
            (),
            rawdata[28:32].view("<i4")[0],
            {"long_name": "on/off STC attenuation"},
        ),
        "tx": (
            (),
            rawdata[32:36].view("<i4")[0],
            {"long_name": "first gate with full sensitivity in STC mode"},
        ),
        "wnd": (
            (),
            rawdata[44:48].view("<i4")[0],
            {"long_name": "debug mode if not 0."},
        ),
        "pos": (
            (),
            rawdata[48:52].view("<i4")[0],
            {
                "long_name": "delay between sync and tx pulse for phase corr",
                "units": "ns",
            },
        ),
        "add": (
            (),
            rawdata[52:56].view("<i4")[0],
            {"long_name": "add to pulse"},
        ),
        "of0": (
            (),
            rawdata[68:72].view("<i4")[0],
            {"long_name": "detection threshold"},
        ),
        "swt": (
            (),
            rawdata[76:80].view("<i4")[0],
            {"long_name": "2nd moment estimation threshold"},
        ),
        "osc": (
            (),
            rawdata[84:88].view("<i4")[0],
            {"long_name": "flag - oscillosgram mode"},
        ),
        "HSn": (
            (),
            rawdata[100:104].view("<i4")[0],
            {"long_name": "flag - Hildebrand div noise detection in noise gate"},
        ),
        "HSa": (
            (),
            rawdata[104:108].view("<f4"),
            {"long_name": "flag - Hildebrand div noise detection in all gates"},
        ),
        "Raw_Gate1": (
            (),
            rawdata[124:128].view("<i4")[0],
            {"long_name": "lowest range gate for spectra saving"},
        ),
        "Raw_Gate2": (
            (),
            rawdata[128:132].view("<i4")[0],
            {"long_name": "range gates with atmospheric signal"},
        ),
        "Raw": (
            (),
            rawdata[132:136].view("<i4")[0],
            {"long_name": "flag - IQ or spectra saving on/off"},
        ),
        "Prc": (
            (),
            rawdata[136:140].view("<i4")[0],
            {"long_name": "flag - Moment estimation switched on/off"},
        ),
    }


def decode_srvi(rawdata):
    return {
        "frm": (
            ("frame",),
            rawdata[:, 0:4].view("<u4")[:, 0],
            {"long_name": "data frame number"},
        ),
        "Tm": (("frame",), rawdata[:, 4:8].view("<u4")[:, 0]),
        "TPow": (
            ("frame",),
            rawdata[:, 8:12].view("<f4")[:, 0],
            {"long_name": "avg transmit power"},
        ),
        "NPw": (
            ("frame", "cocx"),
            rawdata[:, 12:20].view("<f4"),
            {"long_name": "noise power pin-mod in save position"},
        ),
        "CPw": (
            ("frame", "cocx"),
            rawdata[:, 20:28].view("<f4"),
            {"long_name": "noise power int. source"},
        ),
        "PS_Stat": (("frame",), rawdata[:, 28:32].view("<u4")[:, 0]),
        "RC_Err": (("frame",), rawdata[:, 32:36].view("<u4")[:, 0]),
        "TR_Err": (("frame",), rawdata[:, 36:40].view("<u4")[:, 0]),
        "dwSTAT": (("frame",), rawdata[:, 40:44].view("<u4")[:, 0]),
        "dwGRST": (("frame",), rawdata[:, 44:48].view("<u4")[:, 0]),
        "AzmPos": (("frame",), rawdata[:, 48:52].view("<f4")[:, 0]),
        "AzmVel": (("frame",), rawdata[:, 52:56].view("<f4")[:, 0]),
        "ElvPos": (("frame",), rawdata[:, 56:60].view("<f4")[:, 0]),
        "ElvVel": (("frame",), rawdata[:, 60:64].view("<f4")[:, 0]),
        "NorthAngle": (("frame",), rawdata[:, 64:68].view("<f4")[:, 0]),
        "time_milli": (("frame",), rawdata[:, 68:72].view("<u4")[:, 0]),
        "PD_DataQuality": (("frame",), rawdata[:, 72:76].view("<u4")[:, 0]),
        "LO_Frequency": (("frame",), rawdata[:, 76:80].view("<f4")[:, 0]),
        "DetuneFine": (("frame",), rawdata[:, 80:84].view("<f4")[:, 0]),
    }


def decode_moment(name):
    def _decode(rawdata):
        # TODO(ALL): HACK: this arbitrarily reduces the range dimension to 512 to fit with the IQ output
        return {
            name: (
                ("frame", "range", "cocx"),
                rawdata.view("<f4").reshape(rawdata.shape[0], -1, 2)[:, :512, :],
            )
        }

    return _decode


def decode_iq(ppar: Optional[xr.Dataset]):
    nfft = 256
    if ppar is not None:
        nfft = ppar.sft.values  # assumed to be xr.Dataset

    def _decode(rawdata):
        return {
            "FFTD": (
                ("frame", "range", "cocx", "fft", "iq"),
                rawdata.view("<i2").reshape(rawdata.shape[0], -1, 2, nfft, 2),
            )
        }

    return _decode


def decode_time(ds):
    """
    Replaces 'Tm' and 'time_milli' variables in dataset 'ds' with decoded time.

    Parameters:
    ds (xarray.Dataset): The input dataset containing 'Tm' and 'time_milli'
                         variables.

    Returns:
    xarray.Dataset: The dataset with 'Tm' and 'time_milli' variables replaced by
                    new 'time' variable added.
    """
    time = (
        np.datetime64("1970-01-01")
        + ds.Tm * np.timedelta64(1000000000, "ns")
        + ds.time_milli
        * np.timedelta64(1000, "ns")  # [sic] 'time_milli' is microseconds
    )
    return ds.drop_vars(["Tm", "time_milli"]).assign(time=time)
