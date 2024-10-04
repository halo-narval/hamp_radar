from collections.abc import Iterable
from pathlib import Path
from typing import Optional, List
import xarray as xr
import numpy as np

from iquick import extract_raw_arrays
from decoders import pds_decode
from geometries import CollectionGeometry, DatasetGeometry
from postprocess import postprocess_iq
from serde_collection import get_collection_geometry
from writezarr import write_iqdataset, ideal_daskchunks


def read_datasetblock(
    datablock: DatasetGeometry,
    ppar: Optional[xr.Dataset] = None,
    use_autochunks: bool = False,
):
    def decode_arrays(raw_arrays, radar_tag, ppar):
        for _, tag, array in raw_arrays:
            for k, v in pds_decode(tag, array, radar_tag, ppar).items():
                yield k, v

    def daskchunk_array(v):
        dims, data = v[:2]
        try:
            attrs = v[2]
        except IndexError:
            attrs = None
        return xr.DataArray(data=data, dims=dims, attrs=attrs).chunk(
            ideal_daskchunks[dims]
        )

    data = np.memmap(datablock.filename, mode="r")
    raw_arrays = extract_raw_arrays(data, datablock.mainblock)
    radar_tag = datablock.mainblock.tag
    if use_autochunks:
        return xr.Dataset(
            {k: v for k, v in decode_arrays(raw_arrays, radar_tag, ppar)}
        ).chunk("auto")
    else:
        return xr.Dataset(
            {
                k: daskchunk_array(v)
                for k, v in decode_arrays(raw_arrays, radar_tag, ppar)
            }
        )


def read_concat_datasetblocks(
    datablocks: List[DatasetGeometry],
    ds_ppar: Optional[xr.Dataset] = None,
    use_autochunks: bool = False,
):
    """reads dataset blocks then concatenates them along frame dimension
    to create a single dataset. If not using auto chunking, frame dimension
    of each variable may be rechunked."""
    dataset = [
        read_datasetblock(d, ds_ppar, use_autochunks=use_autochunks) for d in datablocks
    ]

    if dataset == []:
        return None

    ds = xr.concat(dataset, dim="frame")

    if not use_autochunks:
        for x in list(ds.coords) + list(ds.data_vars):
            if "frame" in ds[x].dims:
                ds[x] = ds[x].chunk(frame=ideal_daskchunks[ds[x].dims]["frame"])
    return ds


def read_dataset(
    dsgeom: DatasetGeometry, postprocess: bool, use_autochunks: bool = False
):
    import warnings

    ds_ppar = read_datasetblock(dsgeom.ppar)
    ds_data = read_concat_datasetblocks(
        dsgeom.data, ds_ppar, use_autochunks=use_autochunks
    )

    if ds_data is None:
        warnings.warn(
            "Warning: no data blocks in dataset",
            UserWarning,
        )
        dataset = ds_ppar
    else:
        dataset = ds_data.merge(ds_ppar)

    dataset = dataset.assign_attrs(radar_tag=dsgeom.ppar.mainblock.tag)

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
            write_iqdataset(ds, filename)
    else:
        for ds in flight_datasets:
            print(ds)


if __name__ == "__main__":
    exit(main())
