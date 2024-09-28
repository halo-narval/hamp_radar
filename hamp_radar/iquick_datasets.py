from collections.abc import Iterable
from pathlib import Path
from typing import Optional
import xarray as xr
import numpy as np

from iquick import extract_raw_arrays
from decoders import pds_decode
from geometries import CollectionGeometry, DatasetGeometry
from postprocess import postprocess_iq
from serde_collection import get_collection_geometry


def read_datasetblock(
    datablock: DatasetGeometry,
    ppar: Optional[xr.Dataset] = None,
):
    data = np.memmap(datablock.filename, mode="r")
    raw_arrays = extract_raw_arrays(data, datablock.mainblock)
    radar_tag = datablock.mainblock.tag
    ds = xr.Dataset(
        {
            k: v
            for _, tag, array in raw_arrays
            for k, v in pds_decode(tag, array, radar_tag, ppar).items()
        }
    )
    return ds


def read_dataset(dsgeom: DatasetGeometry, postprocess: bool):
    import warnings

    ds_ppar = read_datasetblock(dsgeom.ppar)

    chunks = {
        "cocx": 2,
        "range": 512,  # TODO(ALL): see HACK about range dimension = 512
        "fft": 256,
        "iq": 2,
    }
    dataset = [read_datasetblock(d, ds_ppar).chunk(chunks) for d in dsgeom.data]
    if dataset != []:
        dataset = xr.concat(dataset, dim="frame")
        dataset = dataset.merge(ds_ppar)
    else:
        warnings.warn(
            "Warning: no data blocks in dataset",
            UserWarning,
        )
        dataset = ds_ppar

    dataset = dataset.assign_attrs(radar_tag=dsgeom.ppar.mainblock.tag)

    # TODO(ALL): rechunk dataset for better data loading
    if postprocess:
        dataset = dataset.pipe(postprocess_iq)
    return dataset


def read_collection(geom: CollectionGeometry, postprocess=True) -> Iterable[xr.Dataset]:
    """
    Converts data from .pds files accorinding to the CollectionGeometry,
    into several xarray datasets (one for each DSP configuration). Currently
    only functioning with geometry of pds files and decoders for IQ data of
    Ka radar currently operational on HALO (last checked: 13th Septermber 2024).

    Parameters:
        geom (CollectionGeometry): The geometry of the collection of datasets, e.g. for a flight.
        postprocess (bool): Whether to apply post-processing to the dataset. Default is True.

    Returns:
       Iterable[xarray.Dataset]: iterable to the next IQ data dataset in the collection.
    """
    for dsgeom in geom.datasets:
        yield read_dataset(dsgeom, postprocess).assign_attrs(name=geom.name)


def main():
    """
    e.g.
    python iquick_datasets.py ./bin/jsons/HALO-20240818a-test.json
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filenames",
        nargs="+",
        help=".json or .pds file(s) for flight, default assumes .pds file(s)",
        type=Path,
    )
    parser.add_argument(
        "--flightname",
        default=None,
        help="name of flight to label datasets with",
        type=str,
    )
    parser.add_argument(
        "--is_pdsjsons",
        default=False,
        help="True = filenames is for serialized .pds files of flight",
        type=bool,
    )
    parser.add_argument(
        "--is_flightjson",
        default=False,
        help="True = filenames is for serialized flight .json",
        type=bool,
    )
    parser.add_argument(
        "-w",
        "--writedir",
        default=None,
        help="True = directory for writing flight datasets",
        type=Path,
    )
    args = parser.parse_args()

    if args.is_flightjson:
        if len(args.filenames) != 1:
            raise ValueError("please provide one flight geometry file")
        filenames = args.filenames[0]
    else:
        filenames = args.filenames

    geom = get_collection_geometry(
        filenames,
        collectionname=args.flightname,
        is_collectionjson=args.is_flightjson,
        is_pdsjsons=args.is_pdsjsons,
    )

    flight_datasets = read_collection(geom)

    if args.writedir:
        for i, ds in enumerate(flight_datasets):
            filename = args.writedir / f"{ds.name}_{i}.zarr"
            print(f"writing {filename}")
            ds.to_zarr(f"{filename}")
    else:
        for ds in flight_datasets:
            print(ds)


if __name__ == "__main__":
    exit(main())
