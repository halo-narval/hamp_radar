import xarray as xr
from pathlib import Path
from typing import Optional


"""chunks for arrays with these dimensions such that chunksizes
are about 250MB, e.g. each chunk contains 65536000 4 byte integers,
(ideal for dask arrays) and such that dask chunk lengths along
each dimension are >= length of ideal_zarrchunks"""
ideal_daskchunks = {
    (): ({}),
    ("frame",): {
        "frame": 65536000,
    },
    ("frame", "cocx"): {
        "frame": 32768000,
        "cocx": 2,
    },
    ("frame", "range", "cocx"): {
        "frame": 512000,
        "range": 64,
        "cocx": 2,
    },
    ("frame", "range", "cocx", "fft", "iq"): {
        "frame": 2048,
        "range": 64,
        "cocx": 2,
        "fft": 256,
        "iq": 2,
    },
}


def ideal_zarrchunks(nframe: int):
    """returns ideal chunks for zarr for arrays with these dimensions such that
    chunksizes are about 10MB, e.g. a chunk contains 2621440 4 byte integers,
    and such that cocx dimension (size=2) is split into 2 seperate chunks and
    range dimension (size=512) is sub-divided into smaller chunks (size=32)"""
    import warnings

    if nframe > 2621440:
        warnings.warn(
            "Warning: number of frames is very large, consider changing ideal chunkshapes",
            UserWarning,
        )
    return {
        (): (),
        ("frame",): (nframe,),
        ("frame", "cocx"): (nframe, 1),
        ("frame", "range", "cocx"): (81920, 32, 1),
        ("frame", "range", "cocx", "fft", "iq"): (512, 32, 1, 256, 2),
    }


def ideal_zarrchunks_encoding(ds: xr.Dataset):
    chunks = ideal_zarrchunks(ds.sizes["frame"])
    encoding = {
        x: {"chunks": chunks[ds[x].dims]} for x in list(ds.coords) + list(ds.data_vars)
    }
    return encoding


def merge_encodings(e1: dict, e2: dict):
    e3 = {**e1}
    for x in set(e1.keys()) & set(e2.keys()):
        if set(e2[x]) & set(e3[x]) == set():
            e3[x].update(e2[x])
        else:
            raise KeyError(f"encodings have conflicting keys for {x}")
    for x in set(e2.keys()) - set(e1.keys()):
        e3[x] = e2[x]
    return e3


def zarr_encoding(ds: xr.Dataset, encoding: Optional[dict] = None):
    if encoding is None:
        return ideal_zarrchunks_encoding(ds)
    else:
        return merge_encodings(encoding, ideal_zarrchunks_encoding(ds))


def write_iqdataset(ds: xr.Dataset, filename: Path):
    encoding = zarr_encoding(ds)
    ds.to_zarr(filename, encoding=encoding)
