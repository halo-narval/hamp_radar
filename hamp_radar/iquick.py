from collections.abc import Iterable

from numpy.lib.stride_tricks import as_strided

from geometries import (
    SingleSubBlockGeometry,
    SingleMainBlockGeometry,
    MultiMainBlockGeometry,
)


def get_tag_size(data):
    """
    Extracts the tag (also known as signature) and the pointer for size
    from 'data' assuming layout as in Meteorological Ka-Band Cloud Radar
    MIRA35 Manual, section 2.3.1 'Definition of chunk common structure'

    The tag is the first 4 bytes of data and the pointer for size is the
    next 4 bytes. Bytes for tag are decoded to ascii string unless no valid
    tag can be formed (see below). Bytes for size are interpreted as an integer

    If the first 4 bytes cannot be decoded as an ASCII string, tag=None and size=0 are returned.

    Args:
        data (bytes): The data assumed to conform with Meteorological Ka-Band
                      Cloud Radar MIRA35 Manual, section 2.3.1 'Definition of
                      chunk common structure'

    Returns:
        Tuple[Optional[str], int]: The tag and the pointer for size.
    """
    try:
        return bytes(data[:4]).decode("ascii"), int(data[4:8].view("<i4")[0])
    except UnicodeDecodeError:
        return None, 0


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
        blocktype, blocksize = get_tag_size(mainblock[o : o + 8])
        ofs.append(SingleSubBlockGeometry(blocktype, o + 8, blocksize))
        o += 8 + blocksize
    return ofs


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


def extract_raw_arrays(data, mmbg: MultiMainBlockGeometry):
    """
    Generator function to extract arrays from 'data' based on the
    geometry given by the MultiMainBlockGeometry instance.

    Iterating over a list of this generator yields a tuple for the subblocks
    from the MultiMainBlockGeometry instance sequentially.

    Optimisation uses NumPy library as_strided function to create a view of the
    original data array interpreted with different shape and strides. Shape will
    have (nrows, ncols) = (mmbg.count, block.size). Stride will be 1 unless
    mmbg.count > 1, in which case mmbg.step is used to advance to the next
    required subblock (skipping past other subblocks with different tags)

    Args:
        data: The input data (memory map or open file) from which to extract the
              arrays.
        mmbg (MultiMainBlockGeometry): A MultiMainBlockGeometry instance.

    Yields:
        tuple: A tuple containing:
            - mmbg.tag: The tag/signature of the main block.
            - block.tag: The tag/signature of the subblock.
            - ndarray: A view of the data array corresponding to the subblock.
    """
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
