from collections.abc import Iterable
from typing import List
from pathlib import Path

from geometries import (
    MultiMainBlockGeometry,
    PdsFileGeometry,
    DatasetBlockGeometry,
    DatasetGeometry,
    FlightGeometry,
)
from serde_pds_files import (
    get_pdsfile_geometries,
    serialize_write_data,
)


def ppar_for_dataset(
    filename: Path, mmbg: MultiMainBlockGeometry
) -> DatasetBlockGeometry:
    """
    Creates a DatasetBlockGeometry object with only a single PPAR from a MultiMainBlockGeometry
    instance. Requires mmbg.subblocks[0].tag == "PPAR" and len(subblocks) == 1.

    If mmbg.count > 1 a new mmbg is created containing only the last PPAR in the mmbg.

    Args:
        filename (Path): The path to the .pds file associated with the mmbg.
        mmbg (MultiMainBlockGeometry): The multi-main block geometry object.

    Returns:
        DatasetBlockGeometry: A new dataset block geometry object with only the latest PPAR in the mmbg.
    """
    assert mmbg.subblocks[0].tag == "PPAR", "mmbg subblocks should be PPARs"
    assert (
        len(mmbg.subblocks) == 1
    ), "Consecutive PPAR subblocks should never occuur, corrupt data?"

    if mmbg.count > 1:
        # only use last PPAR for DSP configuration
        offset = mmbg.offset + mmbg.step * (mmbg.count - 1)
        count = 1
        mmbg = MultiMainBlockGeometry(
            mmbg.tag, offset, count, mmbg.step, mmbg.subblocks
        )

    ppar = DatasetBlockGeometry(filename, mmbg)
    return ppar


def convert_to_datasetgeometries(
    pfgs: Iterable[PdsFileGeometry],
) -> List[DatasetGeometry]:
    """
    Converts an iterable of PdsFileGeometry instances into a list of DatasetGeometry instances.

    This function converts a collection of PdsFileGeometry instances into
    DatasetGeometry instances. It identifies "PPAR" tags to start new datasets and appends
    data blocks to the most recent dataset. If consecutibe PPAR tags are found withou
    data in between, the ppar of the current dataset is updated to match the last given PPAR.

    Args:
        pfgs (Iterable[PdsFileGeometry]): An iterable of PdsFileGeometry instances, e.g. for a flight.

    Returns:
        List[DatasetGeometry]: A list of DatasetGeometry instances, e.g. for each dataset in a flight.
    """
    import warnings

    datasets = []
    is_ppar = False
    for pfg in pfgs:
        for mmbg in pfg.mainblocks:
            if mmbg.subblocks[0].tag == "PPAR":
                ppar = ppar_for_dataset(pfg.filename, mmbg)
                if is_ppar:
                    # update ppar of existing dataset
                    datasets[-1].ppar = ppar
                else:
                    # start new dataset
                    datasets.append(DatasetGeometry(ppar, []))
                    is_ppar = True
            else:
                if datasets == []:
                    # skip data if no PPAR has yet been found in pfg.mainblocks
                    warnings.warn(
                        "Warning: data found before first PPAR is being skipped",
                        UserWarning,
                    )
                else:
                    # append to latest dataset
                    is_ppar = False
                    datablock = DatasetBlockGeometry(pfg.filename, mmbg)
                    datasets[-1].data.append(datablock)
    return datasets


def scan_flight(flightname: str, pfgs: Iterable[PdsFileGeometry]):
    datasets = convert_to_datasetgeometries(pfgs)
    return FlightGeometry(flightname, datasets)


def write_flight_geometry(geom: FlightGeometry, geomsdir: Path):
    import time

    start = time.time()

    writefile = geomsdir / Path(geom.name).with_suffix(".json")
    serialize_write_data(writefile, geom)

    end = time.time()
    print(f"serializing {geom.name}: {end - start:.5f}s")


def main():
    """
    e.g.
    python serde_flight.py -g dummy_flight_jsons dummy_flight_pds/*.pds --flightname dummyflight100
    or
    python serde_flight.py -g dummy_flight_jsons dummy_flight_pds/*.jsons --flightname dummyflight100 --is_jsons True
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--geomsdir",
        required=True,
        help="directory for writing .json files",
        type=Path,
    )
    parser.add_argument("filenames", nargs="+", help="names of .pds files", type=Path)
    parser.add_argument(
        "-fn", "--flightname", default="testflight", help="name of flight to serialize"
    )
    parser.add_argument(
        "--is_jsons",
        default=False,
        help="True = filenames are for serialized pds .jsons",
        type=bool,
    )
    args = parser.parse_args()

    pfgs = get_pdsfile_geometries(args.filenames, is_jsons=args.is_jsons)
    geom = scan_flight(args.flightname, pfgs)
    write_flight_geometry(geom, args.geomsdir)


if __name__ == "__main__":
    exit(main())
