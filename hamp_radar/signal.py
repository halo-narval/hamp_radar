# %%
def get_iq(radar, channel="co"):
    """
    forms a complex float of the form i +jq from separate i and q input for the specified channel
    """

    return radar[channel][0, :, :].astype(float) + 1j * radar[channel][1, :, :].astype(
        float
    )


def get_pwr(radar, channel="co"):
    """
    returns the power in the specified channel.
    """
    import numpy as np

    return np.square(
        np.abs(
            radar[channel][0, :, :].astype(float)
            + 1j * radar[channel][1, :, :].astype(float)
        )
    )


def denoise(da, threshold=1.0):
    """
    sets value of power less than estimated noise (times a threhsold) to nan.
    """
    import numpy as np

    return da.where(da > da[510, :].std() * threshold, np.nan)


def rng_to_alt(da, surface_gate):
    """
    adds a height coordinate based on range and specified surface gate and
    returns data between surface and nearest useable gate (near_gate)
    """

    alt = da.range[surface_gate] / 1000.0
    return da.assign_coords(height=alt - da.range / 1000.0)


def pwr_to_dbz(da, radar_constant=-115):
    """
    converts power to radar reflectivity using specified radar constant and range
    normalized to 5 km, or 5000 m.
    """

    import numpy as np

    range_scaling = (da.range / 5000.0) ** 2
    return 10.0 * np.log(da * range_scaling) + radar_constant


def smooth(da, kernel="gauss", width=64, window=256, dask="allowed"):
    """
    smooths data by convolving da DataArray with the ('range','time') shape against a kernel to smooth the data
    """
    import scipy
    import numpy as np
    import xarray as xr

    if da.dims != ("range", "time"):
        raise TypeError(
            f"input array has wrong shape {da.dims}, expected ('range', 'time')"
        )
    if kernel == "gauss":
        g = scipy.signal.windows.gaussian(width, window)
        k = g / g.sum()
    elif kernel == "square":
        k = np.ones(len(width)) / len(width)
    elif kernel == "hann":
        k = scipy.signal.windows.hann(width)
    else:
        raise TypeError(f"{kernel} not supported")

    #   K = xr.DataArray(k,dims=("kernel",)).expand_dims(dim={da.dims[0]:da.sizes["range"]})
    #   xr.apply_ufunc(scipy.signal.fftconvolve,da,K
    #                           ,input_core_dims=[["time"],["kernel"]]
    #                           ,output_core_dims=[["time"]]
    #                           ,kwargs={"mode":"same","axes":-1}
    #                           ,dask=dask
    #                           )

    return xr.DataArray(
        scipy.signal.fftconvolve(da, k[np.newaxis, :], mode="same", axes=-1),
        dims=da.dims,
        coords=da.coords,
    )


def taper(da, kernel="hann", nfft=256):
    """
    tapers a window for processing, e.g., as fft.
    """
    import scipy
    import numpy as np
    import xarray as xr

    if da.dims != ("range", "time"):
        raise TypeError(
            f"input array has wrong shape {da.dims}, expected ('range', 'time')"
        )

    if kernel == "hann":
        w = xr.DataArray(scipy.signal.windows.hann(nfft), dims=["nfft"])
    else:
        raise TypeError(f"{kernel} not supported")

    nfrms = np.floor_divide(len(da.time), nfft)
    return np.reshape(da.data[: nfrms * nfft], (nfrms, nfft)) * w


def doppler_spectrum(da, nfft=256, prf=7500.0, rfreq=35.2e9):
    """
    returns doppler power spectrum and assignes velocity as coordinate
    """

    import scipy
    import numpy as np
    import xarray as xr

    c_air = 299792458.0 / 1.0003  # speed of light in air

    nfrms = np.floor_divide(len(da.time), nfft)
    vel = scipy.fft.fftfreq(nfft, d=1 / prf) * 0.5 / (rfreq * c_air)

    co_td = np.asarray(np.reshape(da.data[: nfrms * nfft], (nfrms, nfft)))
    return xr.DataArray(
        scipy.fft.fft(co_td),
        dims=[
            "vel",
        ],
    ).assign_coords(vel=("vel", vel))


# %%
