import numpy as np
from collections.abc import Iterable
from typing import List
from pathlib import Path

from geometries import PdsFileGeometry
from iquick import get_geometry


def scan_pdsfile(filename: Path):
    data = np.memmap(filename, mode="r")
    mmbgs = list(get_geometry(data))
    return PdsFileGeometry(filename, mmbgs)


def serialize_write_data(writefile: Path, data):
    from serde.json import to_json

    with open(writefile, "w") as json_file:
        json_file.write(to_json(data))


def write_pdsfile_geometries(filenames: List[Path], geomsdir: Path):
    import time

    if geomsdir.exists():
        for filename in filenames:
            start = time.time()

            geom = scan_pdsfile(filename)
            writefile = geomsdir / Path(filename.name).with_suffix(".json")
            serialize_write_data(writefile, geom)

            end = time.time()
            print(f"serializing {filename}: {round(end - start, 5)} s")
    else:
        raise ValueError("'" + str(geomsdir) + "' directory for writing doesn't exist")


def deserialize_data(filename: Path, obj):
    from serde.json import from_json

    with open(filename, "r") as json_file:
        return from_json(obj, json_file.read())


def get_pdsfile_geometries(
    filenames: List[Path], is_jsons=False
) -> Iterable[PdsFileGeometry]:
    if is_jsons:
        for filename in filenames:
            yield deserialize_data(filename, PdsFileGeometry)
    else:
        for filename in filenames:
            yield scan_pdsfile(filename)


def main():
    """
    e.g. python serde_pds_files.py -g dummy_flight_jsons dummy_flight_pds/*.pds
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
    args = parser.parse_args()

    write_pdsfile_geometries(args.filenames, args.geomsdir)


if __name__ == "__main__":
    exit(main())
