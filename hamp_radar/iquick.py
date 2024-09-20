from dataclasses import dataclass
from typing import List, Optional
from collections.abc import Iterable

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import as_strided


@dataclass
class SingleSubBlockGeometry:
    tag: str
    offset: int
    size: int


@dataclass
class SingleMainBlockGeometry:
    tag: str
    offset: int
    size: int
    subblocks: List[SingleSubBlockGeometry]


@dataclass
class MultiMainBlockGeometry:
    tag: str
    offset: int
    count: int
    step: Optional[int]
    subblocks: List[SingleSubBlockGeometry]


def main_ofs(mainblock):
    o = 0
    ofs = []
    while o + 8 < len(mainblock):
        blocktype = bytes(mainblock[o : o + 4])
        blocksize = mainblock[o + 4 : o + 8].view("<i4")[0]
        ofs.append(SingleSubBlockGeometry(blocktype, o + 8, blocksize))
        o += 8 + blocksize
    return ofs


def get_tag_size(data):
    return bytes(data[:4]), data[4:8].view("<i4")[0]


def get_geometry(data):
    o = 0
    main_tag, main_size = get_tag_size(data)

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

    return list(compact_geometry(main_blocks(data, o)))


def compact_geometry(
    main_blocks: Iterable[SingleMainBlockGeometry],
) -> Iterable[MultiMainBlockGeometry]:
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
    for mmbg in mmbgs:
        for block in mmbg.subblocks:
            yield (
                mmbg.tag,
                block.tag,
                as_strided(
                    data[mmbg.offset + block.offset :],
                    (mmbg.count, block.size),
                    (mmbg.step if mmbg.count > 1 else 1, 1),
                    subok=True,
                    writeable=False,
                ),
            )


def decode_srvi(rawdata):
    return {
        "frm": (
            ("frame",),
            rawdata[:, 0:4].view("<u4")[:, 0],
            {"long_name": "data frame number"},
        ),
        "Tm": (("frame",), rawdata[:, 4:8].view("<u4")[:, 0]),
        "TPow": (
            ("frame",),
            rawdata[:, 8:12].view("<f4")[:, 0],
            {"long_name": "avg transmit power"},
        ),
        "NPw": (
            ("frame", "cocx"),
            rawdata[:, 12:20].view("<f4"),
            {"long_name": "noise power pin-mod in save position"},
        ),
        "CPw": (
            ("frame", "cocx"),
            rawdata[:, 20:28].view("<f4"),
            {"long_name": "noise power int. source"},
        ),
        "PS_Stat": (("frame",), rawdata[:, 28:32].view("<u4")[:, 0]),
        "RC_Err": (("frame",), rawdata[:, 32:36].view("<u4")[:, 0]),
        "TR_Err": (("frame",), rawdata[:, 36:40].view("<u4")[:, 0]),
        "dwSTAT": (("frame",), rawdata[:, 40:44].view("<u4")[:, 0]),
        "dwGRST": (("frame",), rawdata[:, 44:48].view("<u4")[:, 0]),
        "AzmPos": (("frame",), rawdata[:, 48:52].view("<f4")[:, 0]),
        "AzmVel": (("frame",), rawdata[:, 52:56].view("<f4")[:, 0]),
        "ElvPos": (("frame",), rawdata[:, 56:60].view("<f4")[:, 0]),
        "ElvVel": (("frame",), rawdata[:, 60:64].view("<f4")[:, 0]),
        "NorthAngle": (("frame",), rawdata[:, 64:68].view("<f4")[:, 0]),
        "time_milli": (("frame",), rawdata[:, 68:72].view("<u4")[:, 0]),
        "PD_DataQuality": (("frame",), rawdata[:, 72:76].view("<u4")[:, 0]),
        "LO_Frequency": (("frame",), rawdata[:, 76:80].view("<f4")[:, 0]),
        "DetuneFine": (("frame",), rawdata[:, 80:84].view("<f4")[:, 0]),
    }


def decode_moment(name):
    def _decode(rawdata):
        # TODO: HACK: this arbitrarily reduces the range dimension to 512 to fit with the IQ output
        return {
            name: (
                ("frame", "range", "cocx"),
                rawdata.view("<f4").reshape(rawdata.shape[0], -1, 2)[:, :512, :],
            )
        }

    return _decode


def decode_iq(rawdata):
    # TODO HACK: the 256 (for nfft) just appears, it probably should be read from somewhere else in the data
    return {
        "FFTD": (
            ("frame", "range", "cocx", "fft", "iq"),
            rawdata.view("<i2").reshape(rawdata.shape[0], -1, 2, 256, 2),
        )
    }


decoders = {
    b"SRVI": decode_srvi,
    b"SNRD": decode_moment("SNRD"),
    b"VELD": decode_moment("VELD"),
    b"HNED": decode_moment("HNED"),
    b"RMSD": decode_moment("RMSD"),
    b"FFTD": decode_iq,  # TODO HACK: FFTD may or may not be IQ data. This is configured in PPAR
}


def read_iq(filename, postprocess=True):
    data = np.memmap(filename)
    raw_arrays = list(extract_raw_arrays(data, get_geometry(data)))
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


def decode_time(ds):
    time = (
        np.datetime64("1970-01-01")
        + ds.Tm * np.timedelta64(1000000000, "ns")
        + ds.time_milli * np.timedelta64(1000, "ns")
    )
    return ds.drop_vars(["Tm", "time_milli"]).assign(time=time)


def postprocess_iq(ds):
    return ds.pipe(decode_time)

def get_pdsinfo(pds_fname,blocks_of_output=0):

    """
    parses pds files and reports on structure. it is
    hopefully a rather simple minded implementation
    that helps make the geometry of the input files
    clear.

    pds_fname: the path to a pds file to parse
    blocks_of_output: number of blocks with verbose output
    """
    import datetime
    import struct
    import sys
    import os

    svri_format = '<2I5f5I5f2I2f'
    iblk = 0
    ipos = 0
    f = open(pds_fname, "rb")

    while ipos < os.stat(pds_fname).st_size-8:
        f.seek(ipos)
        header, block_size = struct.unpack("<4si",f.read(8))
        if (header == b'SRVI'):
            f.seek(ipos+8)
            svri     = struct.unpack(svri_format,f.read(struct.calcsize(svri_format)))
            t = datetime.datetime(1970,1,1)+datetime.timedelta(seconds=svri[1])+datetime.timedelta(microseconds=svri[17])
            if ( iblk < blocks_of_output):
                print (f'Frame number {svri[0]}, at {t} and {svri[6]}')
            if iblk == 0 :
                frm0 = svri[0]
                t0   = datetime.datetime(1970,1,1)+datetime.timedelta(seconds=svri[1])+datetime.timedelta(microseconds=svri[17])
            iblk +=1
        if header == b'HAXC' :
            ipos += 8
        else:
            ipos += block_size + 8

    if (svri[0]-frm0 != iblk-1): sys.exit('error, frames do not increment montonically')

    return t0,t,frm0,svri[0],iblk


def untangle_iqf(files,
                zarr_path,
                c_light = 299792458.,
                ref_air=1.0003,
                pulse_length = 208e-9,
                d_gate1 = 150.
                ):

    """
    Takes a list of pds files and returns zarr files
    separating (untangling) frame data from pulse (cocx) data.

    files: path to pds data
    zarr_path: directory in which zarr files are to be written
    (c_light, ref_air, pulse_length, d_gate1): parameter for
    determining the range information passed as a coordinate to
    the pulse dataset
    """
    import sys
    import xarray as xr
    import numpy as np
    import pandas as pd

    first = read_iq(files[0])
    last  = read_iq(files[-1])
    tbeg  = first.time[0]
    fbeg  = first.frm[0]
    nfrms = last.frm[-1]-fbeg+1
    nfft  = last.sizes['fft']
    ngate = last.sizes['range']

    tau_frame = (last.time[-1] - tbeg)/nfrms
    tau_pulse = tau_frame/nfft
    dtfft     = tau_pulse * xr.DataArray(np.arange(nfft), dims=("fft",))
    len_gate  = c_light/ref_air * pulse_length / 2.

    for file in files:
        radar = read_iq(file)
        print (f"First frame {radar.frm[0].values} of file {file}")
        if (np.all(np.diff(radar.time)>0)) :
            time_frame = tbeg + tau_frame*(radar.frm-fbeg)
            ds1 = (radar[["TPow","NPw","CPw"]]
                   .assign_coords(frame=radar.frm,frame_time=radar.time)
                   )
            fout  = f'{zarr_path}/HALO-{pd.to_datetime(radar.time[0].values):%Y%m%da-%H%M%S}-frms.zarr'
            ds1.to_zarr(fout)

            ds2 = (xr.Dataset({
                "co": radar['FFTD'].isel(cocx=0),
                "cx": radar['FFTD'].isel(cocx=1),
            })
            .assign_coords(pulse_time=lambda ds2: time_frame + dtfft)
            .stack(pulse_time=["frame", "fft"], create_index=False)
            .transpose("iq","range","pulse_time")
            .assign_coords(range=("range",np.arange(ngate)*len_gate+d_gate1))
            .assign_coords(iq=("iq",["i","q"]))
            )
            ds2["range"].attrs["long_name"] = "distance from aircraft"
            ds2["range"].attrs["units"] = "m"
            ds2["iq"].attrs["long_name"] = "signal phase (inphase or quadrature)"
            fout  = f'{zarr_path}/HALO-{pd.to_datetime(radar.time[0].values):%Y%m%da-%H%M%S}-cocx.zarr'
            ds2.to_zarr(fout)
        else:
            sys.exit('error frames do not increment monotonically')
    return ds_cocx,ds_frms


def merge_iqf(cocx_files,frms_files):

    """
    Takes a list of pulse data (cocx_files) and rame data (frame_files)
    merges them into a single data set including additional metadata, which
    the function returns, ideally to be written to a zarr dataset.

    cocx_files: list of zarr files with pulse data
    frms_files: list of zarr files with frame data

    ex.,
    zarr_path = './radar_zarr_files'
    files = [pds_path + x for x in sorted(os.listdir(path))]
    untangle_iqf(files,zarr_path)

    cocx_files = sorted(glob.glob(zarr_path + '*cocx.zarr'))
    frms_files = sorted(glob.glob(zarr_path + '*frms.zarr'))
    ds = merge_radar_zarr(cocx_files,frms_files)
    ds.to_zarr(fout)
    """
    import numpy as np
    import xarray as xr
    ds_frms = xr.open_mfdataset(frms_files,engine='zarr')
    dx = (ds_frms
          .merge(ds_frms.NPw.to_dataset(dim='cocx').rename({0:"co_NPw"}).rename({1:"cx_NPw"}))
          .merge(ds_frms.CPw.to_dataset(dim='cocx').rename({0:"co_CPw"}).rename({1:"cx_CPw"}))
          .drop_vars(["NPw","CPw"])
    )
    dx["co_NPw"].attrs['long_name'] = 'co-channel noise power pin-mod in save position'
    dx["cx_NPw"].attrs['long_name'] = 'cross channel noise power pin-mod in save position'
    dx["co_CPw"].attrs['long_name'] = 'co-channel noise power int. source'
    dx["cx_CPw"].attrs['long_name'] = 'cross channel noise power int. source'
    dx["frame_time"].attrs['long_name'] = "radar timestamp for frame"

    dy= xr.open_mfdataset(cocx_files,engine='zarr')
    dy["co"].attrs['long_name'] = 'iq data from co-polarazied receiver'
    dy["cx"].attrs['long_name'] = 'iq data from cross-polarazied receiver'
    dy["pulse_time"].attrs['long_name'] = 'estimated time of pulse from calculated prf'
    dy["range"].attrs["long_name"] = "distance from aircraft"
    dy["range"].attrs["units"] = "m"
    dy["iq"].attrs["long_name"] = "signal phase (inphase or quadrature)"

    return dx.merge(dy)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    print(read_iq(args.filename).pipe(decode_time))


if __name__ == "__main__":
    exit(main())
