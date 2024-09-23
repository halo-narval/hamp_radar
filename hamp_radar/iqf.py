from iquick import read_pds


def untangle_iqf(
    files,
    zarr_path,
    pulse_dt=208e-9,
    wait1_dt=1.0e-6,
):
    """
    Takes a list of pds files and returns zarr files for frame and pulse data respectivel.

    files: path to pds data
    zarr_path: directory in which zarr files are to be written
    (pulse_dt, wait1_dt): radar parameters for determining range
    """
    import xarray as xr
    import numpy as np
    import pandas as pd

    c_air = 299792458.0 / 1.0003  # speed of light in air

    first = read_pds(files[0])

    tbeg = first.time[0]
    fbeg = first.frm[0]
    nfft = first.sizes["fft"]
    ngate = first.sizes["range"]

    last = read_pds(files[-1])
    tau_frame = (last.time[-1] - tbeg) / (last.frm[-1] - fbeg + 1)

    tau_pulse = tau_frame / nfft
    dtfft = tau_pulse * xr.DataArray(np.arange(nfft), dims=("fft",))

    for file in files:
        radar = read_pds(file)
        if radar.sizes["fft"] != nfft:
            raise ValueError("fft dimension cannot change among files")
        if radar.sizes["range"] != ngate:
            raise ValueError("range dimension cannot change among files")
        if not np.all(np.diff(radar.time) > 0):
            raise ValueError("frames do not increment monotonically")
        time_frame = tbeg + tau_frame * (radar.frm - fbeg)
        print(f"First frame {radar.frm[0].values} of file {file}")
        ds1 = xr.Dataset(
            {
                "TPow": radar["TPow"].assign_attrs({"long_name": "transmit power"}),
                "co_NPw": radar["NPw"]
                .isel(cocx=0)
                .assign_attrs({"long_name": "co-channel power from noise source"}),
                "cx_NPw": radar["NPw"]
                .isel(cocx=1)
                .assign_attrs(
                    {"long_name": "cross-channel power from calibration source"}
                ),
                "co_CPw": radar["CPw"]
                .isel(cocx=0)
                .assign_attrs({"long_name": "co-channel power from noise source"}),
                "cx_CPw": radar["CPw"]
                .isel(cocx=1)
                .assign_attrs(
                    {"long_name": "cross-channel power from calibration source"}
                ),
            },
            coords={
                "radar_time": radar.time.assign_attrs(
                    {
                        "long_name": "timestamp for frame",
                        "provenance": "assigned by radar",
                    }
                ),
                "frame": radar.frm.assign_attrs(
                    {"long_name": "timestamp for frame", "method": "assigned by radar"}
                ),
            },
        )
        ds1.to_zarr(
            f"{zarr_path}/HALO-{pd.to_datetime(radar.time[0].values):%Y%m%da-%H%M%S}-frms.zarr"
        )
        ds2 = (
            (
                xr.Dataset(
                    {
                        "co": radar["FFTD"]
                        .isel(cocx=0)
                        .assign_attrs(
                            {"long_name": "iq data from co-polarazied receiver"}
                        ),
                        "cx": radar["FFTD"]
                        .isel(cocx=1)
                        .assign_attrs(
                            {"long_name": "iq data from cross-polarazied receiver"}
                        ),
                    },
                    coords={
                        "range": (
                            ("range",),
                            (wait1_dt + np.arange(ngate) * pulse_dt) * c_air * 0.5,
                            {
                                "long_name": "range",
                                "units": "m",
                                "provenance": "estimated using pulse length",
                            },
                        ),
                        "iq": (
                            ("iq",),
                            ["i", "q"],
                            {"long_name": "signal phase (inphase or quadrature)"},
                        ),
                        "time": (time_frame + dtfft).assign_attrs(
                            {
                                "long_name": "time",
                                "provenance": "calculated in post procssing using geometry (frame_number, prf)",
                            }
                        ),
                    },
                )
            )
            .stack(time=["frame", "fft"], create_index=False)
            .transpose("iq", "range", "time")
        )
        ds2.to_zarr(
            f"{zarr_path}/HALO-{pd.to_datetime(radar.time[0].values):%Y%m%da-%H%M%S}-cocx.zarr"
        )
    return


def merge_iqf(cocx_files, frms_files):
    """
    Merges pulse data (cocx_files) and frame data (frame_files) as zarr files into a single zarr dataset
    """
    import xarray as xr

    return xr.open_mfdataset(frms_files, engine="zarr").merge(
        xr.open_mfdataset(cocx_files, engine="zarr")
    )
