import xarray as xr
from pathlib import Path
from typing import Optional


def ideal_zarrchunks(ds: xr.Dataset):
    """returns ideal size of chunks for variables with these dimensions
    so that chuunks are about 1MB. Addiontally cocx dimension (size=2)
    is split into 2 seperate chunks and range dimension (size=512)
    is sub-divided into 16 smaller chunks (size=32)"""
    import warnings

    frame = ds.sizes["frame"]
    if frame > 2621440:
        warnings.warn(
            "Warning: number of frames is very large, consider changing ideal chunkshapes",
            UserWarning,
        )
    return {
        (): (),
        ("frame",): (frame,),
        ("frame", "cocx"): (frame, 1),
        ("frame", "range", "cocx"): (8192, 32, 1),
        ("frame", "range", "cocx", "fft", "iq"): (512, 32, 1, 256, 2),
    }


def zarr_chunks_encoding(ds: xr.Dataset):
    ideal_chunks = ideal_zarrchunks(ds)
    encoding = {
        x: {"chunks": ideal_chunks[ds[x].dims]}
        for x in list(ds.coords) + list(ds.data_vars)
        if ds[x].chunks
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
        return zarr_chunks_encoding(ds)
    else:
        return merge_encodings(encoding, zarr_chunks_encoding(ds))


def prepare_daskarrays_for_zarrchunks(ds: xr.Dataset, encoding: dict):
    def safe_chunks(current_chunks, desired_chunks):
        return [max(i, c) for i, c in zip(current_chunks, desired_chunks)]

    for x in list(ds.coords) + list(ds.data_vars):
        if ds[x].chunks:
            current_chunks = (c[0] for c in ds[x].chunks)
            desired_chunks = encoding[x]["chunks"]
            ds[x] = ds[x].chunk(safe_chunks(current_chunks, desired_chunks))
    return ds


def write_iqdataset(ds: xr.Dataset, filename: Path):
    encoding = zarr_encoding(ds)
    ds = prepare_daskarrays_for_zarrchunks(ds, encoding)
    ds.to_zarr(filename, encoding=encoding)
