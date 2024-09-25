import numpy as np
from collections.abc import Iterable
from typing import List
from pathlib import Path

from geometries import PdsFileGeometry
from iquick import get_geometry


def serialize_pdsfile(filename: Path):
    from serde.json import to_json

    data = np.memmap(filename, mode="r")
    mmbgs = list(get_geometry(data))
    pfg = PdsFileGeometry(filename, mmbgs)
    return to_json(pfg)


def write_serialized_geometry(writefile: Path, geom):
    import json

    with open(writefile, "w") as json_file:
        parsed = json.loads(geom)  # for nice formatting
        json.dump(parsed, json_file, indent=2)
    return parsed


def write_pdsfile_geometries(filenames: List[Path], geomsdir: Path):
    import time

    if geomsdir.exists():
        for filename in filenames:
            start = time.time()

            geom = serialize_pdsfile(filename)
            writefile = geomsdir / Path(filename.name).with_suffix(".json")
            write_serialized_geometry(writefile, geom)

            end = time.time()
            print(f"serializing {filename}: {round(end - start, 5)} s")
    else:
        raise ValueError("'" + str(geomsdir) + "' directory for writing doesn't exist")


def deserialize_file_geometry(filename: Path, obj):
    import json
    from serde.json import from_json

    with open(filename, "r") as file:
        data = json.load(file)
    data = json.dumps(data)
    return from_json(obj, data)


def get_pdsfile_geometries(
    filenames: List[Path], is_jsons=False
) -> Iterable[PdsFileGeometry]:
    if is_jsons:
        for filename in filenames:
            yield deserialize_file_geometry(filename, PdsFileGeometry)
    else:
        for filename in filenames:
            data = np.memmap(filename, mode="r")
            mmbgs = list(get_geometry(data))
            yield PdsFileGeometry(filename, mmbgs)


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
