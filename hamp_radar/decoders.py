import numpy as np


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


def decode_iq(rawdata):
    # TODO(ALL) HACK: the 256 (for nfft) just appears, it probably should be read from somewhere else in the data
    return {
        "FFTD": (
            ("frame", "range", "cocx", "fft", "iq"),
            rawdata.view("<i2").reshape(rawdata.shape[0], -1, 2, 256, 2),
        )
    }


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
