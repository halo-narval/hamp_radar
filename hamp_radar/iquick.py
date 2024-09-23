from collections.abc import Iterable

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import as_strided

from geometries import (
    SingleSubBlockGeometry,
    SingleMainBlockGeometry,
    MultiMainBlockGeometry,
)
from decoders import decode_srvi, decode_moment, decode_iq, decode_time


def main_ofs(mainblock):
    # blocktype and blocksize assumed to by 4 bytes long
    """
    Returns the list of SingleSubBlockGeometry instances for a main block of
    data, as in Meteorological Ka-Band Cloud Radar MIRA35 Manual, section 2.3.2
    'File structure'.

    Args:
        mainblock (bytes): The main block data ('main chunk' in manual).

    Returns:
        list: A list of SingleSubBlockGeometry instances, each representing a sub-block
              within the main block.
    """
    o = 0
    ofs = []
    while o + 8 < len(mainblock):
        blocktype = bytes(mainblock[o : o + 4])
        blocksize = mainblock[o + 4 : o + 8].view("<i4")[0]
        ofs.append(SingleSubBlockGeometry(blocktype, o + 8, blocksize))
        o += 8 + blocksize
    return ofs


def get_tag_size(data):
    """
    Extracts the tag (also known as signature) and the pointer for size
    from 'data' assuming layout as in Meteorological Ka-Band Cloud Radar
    MIRA35 Manual, section 2.3.1 'Definition of chunk common structure'
    where the tag is the first 4 bytes of data and the pointer is the
    next 4 bytes.

    Args:
        data (bytes): The data assumed to conform with Meteorological Ka-Band
                      Cloud Radar MIRA35 Manual, section 2.3.1 'Definition of
                      chunk common structure'

    Returns:
        Tuple[bytes, int]: The tag and the pointer for size.
    """
    return bytes(data[:4]), data[4:8].view("<i4")[0]


def get_geometry(data):
    """
    Interprets and returns the geometry of the 'data' assuming it is a
    memmap of a PDS file (or an open PDS file) for radar IQ data with structure
    as in Meteorological Ka-Band Cloud Radar MIRA35 Manual, section 2.3.2
    'File structure'.

    First attempt will read the main tag and size from first 8 bytes
    of the data. If the data length is insufficient, or the main tag is not
    found by attempting to return to the start of the file from main_size+8,
    it shifts the offset by 1024 bytes (to skip file header) and tries again.
    If the main tag is still not found, it raises a ValueError.

    If get_tag_size is sucessful, the function defines a generator
    `main_blocks` that iterates through the data, yielding
    `SingleMainBlockGeometry` instances for each block (i.e. for each main chunk
    as defined in section 2.3.2)

    The function returns a list of different MultiMainBlockGeometry instances
    obtained from compacting together simliar SingleMainBlockGeometry from the
    main_blocks multiple SingleMainBlockGeometry instances.

    Parameters:
    data (bytes): The binary data from which to extract geometry information.

    Returns:
    list: A list of compacted geometry information extracted from the data.

    Raises:
    ValueError: If the main tag cannot be found in the data, indicating that the data may not be a PDS file.
    """
    o = 0
    main_tag, main_size = get_tag_size(data[:8])

    if (
        len(data) < main_size + 8
        or get_tag_size(data[o + 8 + main_size :])[0] != main_tag
    ):
        o = 1024
        main_tag, main_size = get_tag_size(data[o:])
        if get_tag_size(data[o + 8 + main_size :])[0] != main_tag:
            raise ValueError("Could not find main tag, is this a PDS file?")

    def main_blocks(data, o):
        while o + 8 < len(data):
            tag, size = get_tag_size(data[o:])
            yield SingleMainBlockGeometry(
                tag, o + 8, size, main_ofs(data[o + 8 : o + 8 + size])
            )
            o += 8 + size

    return compact_geometry(main_blocks(data, o))


def compact_geometry(
    main_blocks: Iterable[SingleMainBlockGeometry],
) -> Iterable[MultiMainBlockGeometry]:
    """
    Compacts a sequence of SingleMainBlockGeometry instances into a sequence of
    MultiMainBlockGeometry instances.

    This function takes an iterable of SingleMainBlockGeometry objects and
    combines sequential ones that are similar enough ('compatible') into a
    single MultiMainBlockGeometry instance. When the previous
    SingleMainBlockGeometry instance is not compatible to the current one, a new
    MultiMainBlockGeometry instance is started. A list of all the
    MultiMainBlockGeometry instances is then returned.

    Args:
        main_blocks (Iterable[SingleMainBlockGeometry]): An iterable of
                                                         SingleMainBlockGeometry
                                                         instances.

    Yields:
        Iterable[MultiMainBlockGeometry]: An iterable of MultiMainBlockGeometry
                                          instances.
    """
    base_offset = None
    prev_offset = None
    prev_distance = None
    prev_subblocks = None
    prev_tag = None
    count = 0
    for mb in main_blocks:
        is_compatible = True
        if prev_offset is not None:
            distance = mb.offset - prev_offset
            if prev_distance is not None and prev_distance != distance:
                is_compatible = False
        else:
            distance = None

        if prev_subblocks != mb.subblocks:
            is_compatible = False

        if prev_tag != mb.tag:
            is_compatible = False

        if is_compatible:
            prev_distance = distance
            count += 1
        else:
            if base_offset is not None and prev_subblocks is not None and count > 0:
                yield MultiMainBlockGeometry(
                    prev_tag, base_offset, count, prev_distance, prev_subblocks
                )
            base_offset = mb.offset
            prev_distance = None
            count = 1

        prev_offset = mb.offset
        prev_subblocks = mb.subblocks
        prev_tag = mb.tag

    yield MultiMainBlockGeometry(
        prev_tag, base_offset, count, prev_distance, prev_subblocks
    )


def extract_raw_arrays(data, mmbgs: Iterable[MultiMainBlockGeometry]):
    """
    Generator function to extract arrays from 'data' based on the
    geometry given by the iterator over MultiMainBlockGeometry instances.

    Iterating over a list of this generator yields a tuple for the subblocks
    across all the MultiMainBlockGeometry instances sequentially.

    Optimisation uses NumPy library as_strided function to create a view of the
    original data array interpreted with different shape and strides. Shape will
    have (nrows, ncols) = (mmbg.count, block.size). Stride will be 1 unless
    mmbg.count > 1, in which case mmbg.step is used to advance to the next
    required subblock (skipping past other subblocks with different tags)

    Args:
        data: The input data (memory map or open file) from which to extract the
              arrays.
        mmbgs (Iterable[MultiMainBlockGeometry]): An iterable of
              MultiMainBlockGeometry instances.

    Yields:
        tuple: A tuple containing:
            - mmbg.tag: The tag/signature of the main block.
            - block.tag: The tag/signature of the subblock.
            - ndarray: A view of the data array corresponding to the subblock.
    """
    for mmbg in mmbgs:
        for block in mmbg.subblocks:
            yield (
                mmbg.tag,
                block.tag,
                as_strided(
                    data[mmbg.offset + block.offset :],
                    shape=(mmbg.count, block.size),
                    strides=(mmbg.step if mmbg.count > 1 else 1, 1),
                    subok=True,
                    writeable=False,
                ),
            )


def single_dspparams_data(data):
    """
    Returns raw arrays of data from the first occurrence of DSP parameters
    up to the next occurrence or up to the end of the data.

    The first value in the returned list of raw arrays is the DSP parameters
    configuration (tag == "PPAR").

    Raises warning if no PPAR tags are found in the data.

    Parameters:
    data (any): The input data (memory map or open file) from which to extract the
                arrays associated with a single DSP parameter configuration.

    Returns:
    list: A list of raw arrays extracted from the data.
    """

    mmbgs = get_geometry(data)

    start = None
    end = -1
    for i, mmbg in enumerate(mmbgs):
        if mmbg.tag == "PPAR":
            if start is None:
                start = i
            else:
                end = i
                break

    if start is None:
        start = 0
        print(
            "Warning: No PPAR tags found, using data from entire file which isn't associated with any DSP configuration"
        )

    single_dspparams_mmbgs = mmbgs[start:end]

    return extract_raw_arrays(data, single_dspparams_mmbgs)


def read_pds(filename, postprocess=True):
    """
    Converts data from a file called 'filename', into an xarray Dataset. Currently
    only functioning with geometry of pds files and decoders for IQ data of
    Ka radar currently operational on HALO (last checked: 13th Septermber 2024).

    Parameters:
    filename (str): The path to the file containing the IQ data.
    postprocess (bool): Whether to apply post-processing to the dataset. Default is True.

    Returns:
    xarray.Dataset: The IQ data dataset.
    """

    # Decoders for IQ data as in Meteorological Ka-Band Cloud Radar MIRA35 Manual,
    # section 2.3.3.2 'Embedded chain type 2; Data chain'. Note these decoders are
    # specific to the Ka radar currently in operation on HALO.
    # (last checked: 13th Septermber 2024).
    decoders = {
        b"SRVI": decode_srvi,
        b"SNRD": decode_moment("SNRD"),
        b"VELD": decode_moment("VELD"),
        b"HNED": decode_moment("HNED"),
        b"RMSD": decode_moment("RMSD"),
        b"FFTD": decode_iq,  # TODO(ALL) HACK: FFTD may or may not be IQ data. This is configured in PPAR
    }

    data = np.memmap(filename, mode="r")
    raw_arrays = single_dspparams_data(data)
    ds = xr.Dataset(
        {
            k: v
            for _, tag, array in raw_arrays
            if tag in decoders
            for k, v in decoders[tag](array).items()
        }
    )
    if postprocess:
        ds = ds.pipe(postprocess_iq)
    return ds


def postprocess_iq(ds):
    # TODO(ALL): move to new file
    return ds.pipe(decode_time)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    print(read_pds(args.filename).pipe(decode_time))


if __name__ == "__main__":
    exit(main())
