import numpy as np
import xarray as xr


def postprocess_iq(ds: xr.Dataset):
    return ds.pipe(postprocess_time)


def postprocess_time(ds):
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
