from collections.abc import Iterable
from typing import List, Optional, Union
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
    deserialize_data,
)


def is_ppar(block):
    return block.mainblock.subblocks[0].tag == "PPAR"


def pparblock_for_dataset(
    filename: Path, mmbg: MultiMainBlockGeometry
) -> DatasetBlockGeometry:
    """
    Creates a DatasetBlockGeometry object with only a single PPAR from a MultiMainBlockGeometry
    instance. Requires mmbg.subblocks[0].tag == "PPAR" and len(subblocks) == 1.

    If mmbg.count > 1 a new mmbg is created containing only the last PPAR in the
    mmbg (i.e. the latest DSP configuration).

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
        offset = mmbg.offset + mmbg.step * (mmbg.count - 1)
        count = 1
        mmbg = MultiMainBlockGeometry(
            mmbg.tag, offset, count, mmbg.step, mmbg.subblocks
        )

    return DatasetBlockGeometry(filename, mmbg)


def generate_dbgs(pfgs: Iterable[PdsFileGeometry]) -> Iterable[DatasetBlockGeometry]:
    """
    Generates DatasetBlockGeometry intsances from a series of PdsFileGeometry instances

    Args:
        pfgs (Iterable[PdsFileGeometry]): An iterable of PdsFileGeometry instances.

    Yields:
        DatasetBlockGeometry: A DatasetBlockGeometry object for each mainblock in
                              the PdsFileGeometry iterable.
    """
    for pfg in pfgs:
        for mmbg in pfg.mainblocks:
            if mmbg.subblocks[0].tag == "PPAR":
                yield pparblock_for_dataset(pfg.filename, mmbg)
            else:
                yield DatasetBlockGeometry(pfg.filename, mmbg)


def next_pparblock(blocks: Iterable[DatasetBlockGeometry]) -> DatasetBlockGeometry:
    """
    Finds and returns the next PPAR block from an (peekable) iterable of dataset blocks.

    Function finds the "last of the first" PPAR block, i.e. it finds the first
    occurance a PPAR in the list of DatasetBlockGeometry instances and if more
    PPAR blocks follow consecutively after this one then the latest PPAR block
    in the consecutive PPARs is returned.

    Args:
        blocks (Iterable[DatasetBlockGeometry]): Peekable iterable of dataset blocks to search through.

    Returns:
        DatasetBlockGeometry: The next PPAR block found in the iterable.

    Raises:
        IndexError: If no PPAR block is found by iterating over the dataset blocks.
    """
    import warnings

    try:
        while not is_ppar(blocks.peek()):
            # skip data if no PPAR block has yet been found
            warnings.warn(
                "Warning: data blocks before PPAR skipped",
                UserWarning,
            )
            skipblock = next(blocks)
    except IndexError as e:
        e.add_note(f"No PPAR found in list of dataset blocks for {skipblock.filename}")
        raise

    while is_ppar(blocks.peek()):
        ppar = next(blocks)
        if not blocks:
            warnings.warn(
                "Warning: no data blocks appear after PPAR",
                UserWarning,
            )
            break

    return ppar


def next_datablocks(
    blocks: Iterable[DatasetBlockGeometry],
) -> List[DatasetBlockGeometry]:
    """
    Returns a list of dataset blocks from an (peekable) iterable for all the blocks
    until a PPAR block is found of the iterable is exhausted.

    Args:
        blocks (Iterable[DatasetBlockGeometry]): An iterable of dataset blocks to search through.

    Returns:
        List[DatasetBlockGeometry]: The dataset blocks up to the next PPAR / end of the iterable.
    """

    def gen_datablocks():
        while blocks and not is_ppar(blocks.peek()):
            yield next(blocks)

    return list(gen_datablocks())


def convert_to_datasetgeometries(
    pfgs: Iterable[PdsFileGeometry],
) -> Iterable[DatasetGeometry]:
    """
    Converts an iterable of PdsFileGeometry instances into an iterable of DatasetGeometry instances.

    This function converts a collection of PdsFileGeometry instances into
    DatasetGeometry instances. It identifies the next "PPAR" tag in the remaining mainblocks from
    the PdsFileGeometry instances and the associated data thereafter in order to return the
    geometry of the next dataset from iterating over PdsFileGeometry instances.

    Args:
        pfgs (Iterable[PdsFileGeometry]): An iterable of PdsFileGeometry instances, e.g. for a flight.

    Returns:
        Iterable[DatasetGeometry]: An iterable list of DatasetGeometry instances, e.g. for each dataset in a flight.
    """
    from more_itertools import peekable

    blocks = peekable(generate_dbgs(pfgs))
    while blocks:
        ppar = next_pparblock(blocks)
        data = next_datablocks(blocks)
        yield DatasetGeometry(ppar, data)


def scan_flight(flightname: str, pfgs: Iterable[PdsFileGeometry]):
    datasets = list(convert_to_datasetgeometries(pfgs))
    return FlightGeometry(flightname, datasets)


def write_flight_geometry(geom: FlightGeometry, geomsdir: Path):
    import time

    start = time.time()

    writefile = geomsdir / Path(geom.name).with_suffix(".json")
    serialize_write_data(writefile, geom)

    end = time.time()
    print(f"serializing {geom.name}: {end - start:.5f}s")


def get_flight_geometry(
    filenames: Union[Path, List[Path]],
    flightname: Optional[str] = None,
    is_flightjson=False,
    is_pdsjsons=False,
) -> FlightGeometry:
    if is_flightjson:
        return deserialize_data(filenames, FlightGeometry)
    else:
        if flightname is None:
            raise ValueError("please provide a flightname for the flight geometry")
        pfgs = get_pdsfile_geometries(filenames, is_jsons=is_pdsjsons)
        return scan_flight(flightname, pfgs)


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
