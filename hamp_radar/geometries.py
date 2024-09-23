from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SingleSubBlockGeometry:
    """
    Representation of a single sub-block geometry, as explained in
    Meteorological Ka-Band Cloud Radar MIRA35 Manual, section 2.3.1
    'Definition of chunk common structure', e.g. for a chunk in an
    Embedded chain type 2 (see section 2.3.3.2).

    Attributes:
        tag (str): The tag/signature representing the chunk type.
        offset (int): Pointer to start of sub-block's data block relative to the enclosing main block.
        size (int): The size of the sub-block's data block (bytes).
    """

    tag: str
    offset: int
    size: int


@dataclass
class SingleMainBlockGeometry:
    """
    Representation of a single main block geometry as list of
    SingleSubBlockGeometry instances with some overarching metadata.

    This represents one 'main chunk' as in Meteorological Ka-Band Cloud Radar
    MIRA35 Manual, section 2.3.2 'File structure', and should be chunk as in
    section 2.3.3 'Main chunk structure'.

    The sub-blocks of a SingleMainBlockGeometry correspond to each of the blocks
    in the embedded chunk chain which composes the main chunk.

    Attributes:
        tag (str): The tag / frame header describing the data in the sub-blocks.
        offset (int): Pointer to start of the main block's sub-blocks, relative to the file.
        size (int): The size of the main block's sub-blocks (bytes).
        subblocks (List[SingleSubBlockGeometry]): A list of sub-blocks within
                                                  the main block.
    """

    tag: str
    offset: int
    size: int
    subblocks: List[SingleSubBlockGeometry]


@dataclass
class MultiMainBlockGeometry:
    """
    Representation of multiple main blocks as list of SingleSubBlockGeometry
    instances with some overarching metadata. This can be multiple 'main chunks'
    as in Meteorological Ka-Band Cloud Radar MIRA35 Manual, section 2.3.2
    'File structure'.

    Allows for 'count' number of SingleMainBlockGeometry instances to be
    combined, assuming they are similar enough (e.g. same subblocks, same offset,
    same tag etc.). (See compact_geometry function.)

    Attributes:
        tag (str): The tag/signature describing the main blocks.
        offset (int): Pointer to the start of the first main block.
        count (int): The number of SingleMainBlockGeometry instances
                     combined to make the MultiMainBlockGeometry instance.
        step (Optional[int]): The distance between each SingleMainBlockGeometry
                              instance, required if count > 1.
        subblocks (List[SingleSubBlockGeometry]): The sub-blocks in each main block.
    """

    tag: str
    offset: int
    count: int
    step: Optional[int]
    subblocks: List[SingleSubBlockGeometry]
