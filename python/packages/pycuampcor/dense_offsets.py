"""
pycuampcor.dense_offsets: estimate the dense offset field between two SAR images with pycuampcor

The dense offsets tool (formerly cuDenseOffsets.py of ISCE2, used with MintPy for offset time
series): the offsets of the windows on a regular grid of the reference image, written as
band-interleaved binary files (offset.bip, offset_snr.bip, ...) with their headers, and optionally
the geometry (lat, lon, ...) on the same grid (--full-geom, --out-geom).

The images are raw binary files (single band, complex64 or float32, e.g., ISCE SLCs), or
rasters read with GDAL (GeoTIFF, VRT, ...: the first band, any pixel type, converted to float32,
or to complex64 if complex; if pycuampcor is built with GDAL). The size and pixel type of a raw
image are given with --length, --width, --data-type, or read from its ISCE .xml (if ISCE2 is
installed), or from GDAL (the .vrt next to the file, or the file itself); those of a GDAL raster
from GDAL. The output headers are ISCE .xml (with .vrt) if ISCE2 is installed, ENVI .hdr and .vrt
otherwise. ISCE2 is optional; GDAL is needed to read the image metadata without ISCE2 (or the
options), and for the geometry.
"""

import argparse
import array
import os
import sys
import time

import pycuampcor

EXAMPLE = '''example:
  python -m pycuampcor.dense_offsets -r ./SLC/20151120/20151120.slc.full -s ./SLC/20151214/20151214.slc.full
  python -m pycuampcor.dense_offsets -r ./SLC/20151120/20151120.slc.full -s ./SLC/20151214/20151214.slc.full --outprefix ./offsets/20151120_20151214/offset --ww 256 --wh 256 --sw 8 --sh 8 --oo 32 --kw 300 --kh 100 --gpuid 0

  # offset and its geometry
  # tip: re-run with --full/out-geom and without --redo to generate geometry only
  python -m pycuampcor.dense_offsets -r ./SLC/20151120/20151120.slc.full -s ./SLC/20151214/20151214.slc.full --outprefix ./offsets/20151120_20151214/offset --ww 256 --wh 256 --sw 8 --sh 8 --oo 32 --kw 300 --kh 100 --gpuid 0 --full-geom ./geom_reference --out-geom ./offset/geom_reference

  # GeoTIFF images (e.g., amplitudes of UInt16, read as float32)
  python -m pycuampcor.dense_offsets -r ref.tif -s sec.tif --op offsets/offset

  # images without ISCE/GDAL metadata, on the CPU
  python -m pycuampcor.dense_offsets -r ref.slc -s sec.slc --length 2000 --width 1000 --data-type complex64 --op offsets/offset --cpu
'''

# the pixel types of the images: name -> (bytes per pixel, GDAL names)
DATA_TYPES = {'complex64': (8, ('CFloat32',)), 'float32': (4, ('Float32',))}
# the extensions of the images read with GDAL by the auto reader
GDAL_EXTENSIONS = ('.tif', '.tiff', '.vrt')


def create_parser():
    """The command-line parser"""
    parser = argparse.ArgumentParser(prog='python -m pycuampcor.dense_offsets',
                                     description='Generate the offset field between two SAR images',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    # input/output
    parser.add_argument('-r', '--reference', type=str, dest='reference', required=True,
                        help='Reference image (raw binary file)')
    parser.add_argument('-s', '--secondary', type=str, dest='secondary', required=True,
                        help='Secondary image (raw binary file)')
    parser.add_argument('--length', type=int, dest='length', default=None,
                        help='Length (lines) of the images (default: from the image metadata).')
    parser.add_argument('--width', type=int, dest='width', default=None,
                        help='Width (samples) of the images (default: from the image metadata).')
    parser.add_argument('--data-type', dest='data_type', choices=sorted(DATA_TYPES), default=None,
                        help='Pixel type of the images (default: from the image metadata).')
    parser.add_argument('--reader', dest='reader', choices=['auto', 'raw', 'gdal'], default='auto',
                        help='How to read the images: raw binary files, or rasters with GDAL (auto: GDAL\n'
                             f'for the names ending with {", ".join(GDAL_EXTENSIONS)}, raw otherwise) (default: %(default)s).')
    parser.add_argument('--fix-xml', '--fix-image-xml', dest='fixImageXml', action='store_true',
                        help='Fix the image file path in the XML file (ISCE2). Enable this if input files have been moved.')
    parser.add_argument('--fix-vrt', '--fix-image-vrt', dest='fixImageVrt', action='store_true',
                        help='Fix the image file path in the VRT file (ISCE2). Enable this if input files have VRT pointing to non-existing burst files')

    parser.add_argument('--op', '--outprefix', '--output-prefix', type=str, dest='outprefix',
                        default='offset', required=True,
                        help='Output prefix (default: %(default)s).')
    parser.add_argument('--os', '--outsuffix', type=str, dest='outsuffix', default='',
                        help='Output suffix (default: %(default)s).')

    # window size settings
    parser.add_argument('--ww', type=int, dest='winwidth', default=64,
                        help='Window width (default: %(default)s).')
    parser.add_argument('--wh', type=int, dest='winhgt', default=64,
                        help='Window height (default: %(default)s).')
    parser.add_argument('--sw', type=int, dest='srcwidth', default=20,
                        help='Half search range along width, (default: %(default)s, recommend: 4-32).')
    parser.add_argument('--sh', type=int, dest='srchgt', default=20,
                        help='Half search range along height (default: %(default)s, recommend: 4-32).')
    parser.add_argument('--kw', type=int, dest='skipwidth', default=64,
                        help='Skip across (default: %(default)s).')
    parser.add_argument('--kh', type=int, dest='skiphgt', default=64,
                        help='Skip down (default: %(default)s).')

    # the windows: given by the starting pixel and the number of windows, or determined from
    # the margin, the search range and the image size
    parser.add_argument('--mm', type=int, dest='margin', default=0,
                        help='Margin (default: %(default)s).')
    parser.add_argument('--nwa', type=int, dest='numWinAcross', default=-1,
                        help='Number of windows across (default: %(default)s to be auto-determined).')
    parser.add_argument('--nwd', type=int, dest='numWinDown', default=-1,
                        help='Number of windows down (default: %(default)s to be auto-determined).')
    parser.add_argument('--startpixelac', dest='startpixelac', type=int, default=None,
                        help='Starting pixel across of the reference image. ' +
                             'Default: %(default)s to be determined by the margin and the search range.')
    parser.add_argument('--startpixeldw', dest='startpixeldw', type=int, default=None,
                        help='Starting pixel down of the reference image. ' +
                             'Default: %(default)s to be determined by the margin and the search range.')
    parser.add_argument('--use-center', dest='use_center', type=int, default=0,
                        help='Whether the starting/ending pixels are the centers of the windows. ' +
                             'Default: %(default)s (the first/last pixels of the windows).')
    parser.add_argument('--endpixelac', dest='endpixelac', type=int, default=None,
                        help='Ending pixel across of the reference image. ' +
                             'Default: %(default)s to be determined by the image width.')
    parser.add_argument('--endpixeldw', dest='endpixeldw', type=int, default=None,
                        help='Ending pixel down of the reference image. ' +
                             'Default: %(default)s to be determined by the margin, the search range and the image length.')

    # cross-correlation algorithm
    parser.add_argument('--wf', '--workflow', dest='workflow', type=int, default=0,
                        help='workflow (0: two passes, first pass without antialiasing oversampling, 1: one pass with antialiasing oversampling) (default: %(default)s).')
    parser.add_argument('--alg', '--algorithm', dest='algorithm', type=int, default=0,
                        help='cross-correlation algorithm (0 = frequency domain, 1 = time domain) (default: %(default)s).')
    parser.add_argument('--raw-osf', '--raw-over-samp-factor', type=int, dest='raw_oversample',
                        default=2, choices=range(2, 5),
                        help='anti-aliasing oversampling factor, equivalent to i_ovs in ROI_PAC (default: %(default)s).')
    parser.add_argument('--drmp', '--deramp', dest='deramp', type=int, default=0,
                        help='remove phase ramp of complex signals before antialiasing oversampling (0: use magnitude instead (for TOPS), 1: linear phase ramp, 2: skip deramping) (default: %(default)s).')
    parser.add_argument('--drmp-axis', '--deramp-axis', dest='deramp_axis', type=int, default=2,
                        help='axis to remove phase (0: along column/azimuth only, 1: along row/range only, 2: along both axes) (default: %(default)s).')

    # gross offset
    gross = parser.add_argument_group('Initial gross offset')
    gross.add_argument('-g', '--gross', type=int, dest='gross', default=0,
                       help='Use varying gross offset or not')
    gross.add_argument('--aa', type=int, dest='azshift', default=0,
                       help='Gross azimuth offset (default: %(default)s).')
    gross.add_argument('--rr', type=int, dest='rgshift', default=0,
                       help='Gross range offset (default: %(default)s).')
    gross.add_argument('--gf', '--gross-file', type=str, dest='gross_offset_file',
                       help='Varying gross offset input file (int32 pairs (down, across), one per window)')
    gross.add_argument('--mg', '--merge-gross-offset', type=int, dest='merge_gross_offset', default=0,
                       help='Whether to merge gross offset to the output offset image (default: %(default)s).')

    corr = parser.add_argument_group('Correlation surface')
    corr.add_argument('--corr-stat-size', type=int, dest='corr_stat_win_size', default=21,
                      help='Zoom-in window size of the correlation surface for statistics(snr/variance) (default: %(default)s).')
    corr.add_argument('--corr-srch-size', type=int, dest='corr_srch_size', default=4,
                      help='(half) Zoom-in window size of the correlation surface for oversampling, '
                      'equivalent to i_srcp in ROI_PAC (default: %(default)s).')
    corr.add_argument('--corr-osf', '--oo', '--corr-over-samp-factor', type=int, dest='corr_oversample', default=32,
                      help='Oversampling factor of the zoom-in correlation surface (default: %(default)s).')
    corr.add_argument('--corr-osm', '--corr-over-samp-method', type=int, dest='corr_oversamplemethod', default=0,
                      help='Oversampling method for the correlation surface 0=fft, 1=sinc (default: %(default)s).')

    geom = parser.add_argument_group('Geometry', 'generate corresponding geometry datasets (requires GDAL)')
    geom.add_argument('--full-geom', dest='full_geometry_dir', type=str,
                      help='(Input) Directory of geometry files in full resolution.')
    geom.add_argument('--out-geom', dest='out_geometry_dir', type=str,
                      help='(Output) Directory of geometry files corresponding to the offset field.')

    # processing settings
    proc = parser.add_argument_group('Processing parameters')
    proc.add_argument('--gpuid', '--gid', '--gpu-id', dest='gpuid', type=int, default=0,
                      help='GPU ID (default: %(default)s).')
    proc.add_argument('--nstreams', dest='nstreams', type=int, default=2,
                      help='Number of cuda streams (default: %(default)s).')
    proc.add_argument('--cpu', dest='cpu', action='store_true',
                      help='Use the CPU implementation instead of GPU (also used if pycuampcor has no CUDA support).')
    proc.add_argument('--nthreads', dest='nthreads', type=int, default=0,
                      help='Number of CPU threads, 0 for the OpenMP default (default: %(default)s).')
    proc.add_argument('--usemmap', dest='usemmap', type=int, default=1,
                      help='Unused (raw images are always read with a memory map); kept for compatibility.')
    proc.add_argument('--mmapsize', dest='mmapsize', type=int, default=8,
                      help='The memory map buffer size in GB (default: %(default)s).')
    proc.add_argument('--nwac', type=int, dest='numWinAcrossInChunk', default=0,
                      help='Number of windows across in a chunk/batch, 0 for automatic (default: %(default)s).')
    proc.add_argument('--nwdc', type=int, dest='numWinDownInChunk', default=0,
                      help='Number of windows down in a chunk/batch, 0 for automatic (default: %(default)s).')

    proc.add_argument('--redo', dest='redo', action='store_true',
                      help='To redo by force (ignore the existing offset fields).')

    return parser


def cmd_line_parse(iargs=None):
    """The command-line options"""
    return create_parser().parse_args(args=iargs)


def isce_available():
    """Whether ISCE2 is installed (for the .xml metadata)"""
    try:
        import isce  # noqa: F401
        import isceobj  # noqa: F401
        return True
    except ImportError:
        return False


def use_gdal(fname, inps):
    """Whether the image is a raster read with GDAL (by --reader, or by the extension of its name)"""
    return inps.reader == 'gdal' or (inps.reader == 'auto' and fname.lower().endswith(GDAL_EXTENSIONS))


def open_gdal(fname):
    """Open a raster with GDAL (the Python bindings, osgeo) to read its metadata"""
    try:
        from osgeo import gdal
    except ImportError:
        raise RuntimeError(f'No metadata for {fname}: give --length, --width (and --data-type), '
                           'or install ISCE2 (.xml) or GDAL (.vrt)') from None
    gdal.UseExceptions()
    return gdal.Open(fname, gdal.GA_ReadOnly)


def image_info(fname, inps):
    """
    The length, width and pixel type of an image: from the options, or else from GDAL for a GDAL
    raster (float32, or complex64 if complex), and for a raw image from the ISCE .xml (if ISCE2 is
    installed) or GDAL (the .vrt next to the image, or the image itself)
    """
    if inps.length and inps.width:
        return inps.length, inps.width, inps.data_type or 'complex64'
    if use_gdal(fname, inps):
        from osgeo import gdal
        ds = open_gdal(fname)
        data_type = 'complex64' if gdal.DataTypeIsComplex(ds.GetRasterBand(1).DataType) else 'float32'
        return ds.RasterYSize, ds.RasterXSize, inps.data_type or data_type
    if os.path.exists(fname + '.xml') and isce_available():
        import isceobj
        img = isceobj.createImage()
        img.load(fname + '.xml')
        data_type = 'complex64' if img.getDataType().upper().startswith('C') else 'float32'
        return img.getLength(), img.getWidth(), inps.data_type or data_type
    from osgeo import gdal
    ds = open_gdal(fname + '.vrt' if os.path.exists(fname + '.vrt') else fname)
    gdal_type = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    data_type = next((name for name, (_, gdal_names) in DATA_TYPES.items() if gdal_type in gdal_names), None)
    if data_type is None and not inps.data_type:
        raise RuntimeError(f'Unsupported pixel type {gdal_type} of {fname}: complex64 or float32 raw images only '
                           '(other types with --reader gdal)')
    return ds.RasterYSize, ds.RasterXSize, inps.data_type or data_type


def check_raw_image(fname, length, width, data_type):
    """Check that the image is a raw binary file of length x width pixels of the data type"""
    expected = length * width * DATA_TYPES[data_type][0]
    size = os.path.getsize(fname)
    if size != expected:
        raise RuntimeError(f'{fname}: {size} bytes, not a raw {data_type} image of {length} x {width} '
                           f'({expected} bytes)')


def fix_isce_metadata(fnames, fix_xml, fix_vrt):
    """Fix the file paths in the ISCE .xml or .vrt files of the images (ISCE2)"""
    if not (fix_xml or fix_vrt):
        return
    if not isce_available():
        raise RuntimeError('--fix-xml and --fix-vrt require ISCE2')
    from isceobj.Util.ImageUtil import ImageLib as IML
    for fname in fnames:
        fname = os.path.abspath(fname)
        img = IML.loadImage(fname)[0]
        if fix_xml:
            img.filename = fname
            img.setAccessMode('READ')
            img.renderHdr()
        if fix_vrt:
            img.renderVRT()


def write_header(fname, width, length, bands, double):
    """
    Write the header of a band-interleaved (BIP) output image: ISCE .xml (and .vrt) if ISCE2 is
    installed, ENVI .hdr and .vrt otherwise
    """
    if isce_available():
        import isceobj
        img = isceobj.createImage()
        img.setDataType('DOUBLE' if double else 'FLOAT')
        img.setFilename(fname)
        img.setBands(bands)
        img.scheme = 'BIP'
        img.setWidth(width)
        img.setLength(length)
        img.setAccessMode('read')
        img.renderHdr()
        return
    size = 8 if double else 4
    with open(fname + '.hdr', 'w') as f:
        f.write('ENVI\n'
                'description = {pycuampcor offsets}\n'
                f'samples = {width}\nlines = {length}\nbands = {bands}\n'
                'header offset = 0\nfile type = ENVI Standard\n'
                f'data type = {5 if double else 4}\ninterleave = bip\nbyte order = 0\n')
    vrt_bands = ''.join(
        f'  <VRTRasterBand dataType="{"Float64" if double else "Float32"}" band="{b + 1}" subClass="VRTRawRasterBand">\n'
        f'    <SourceFilename relativeToVRT="1">{os.path.basename(fname)}</SourceFilename>\n'
        f'    <ByteOrder>LSB</ByteOrder>\n'
        f'    <ImageOffset>{b * size}</ImageOffset>\n'
        f'    <PixelOffset>{bands * size}</PixelOffset>\n'
        f'    <LineOffset>{bands * size * width}</LineOffset>\n'
        f'  </VRTRasterBand>\n' for b in range(bands))
    with open(fname + '.vrt', 'w') as f:
        f.write(f'<VRTDataset rasterXSize="{width}" rasterYSize="{length}">\n{vrt_bands}</VRTDataset>\n')


def estimate_offset_field(reference, secondary, inps):
    """
    Estimate the offset field with pycuampcor
    @param reference, secondary the paths of the reference and secondary images
    @param inps the command-line options
    @return the ampcor object, and the grid of the offset field in the reference image (x/y_start:
        the center of the first window, x/y_step, x/y_win_num) for the geometry
    """
    fix_isce_metadata([reference, secondary], inps.fixImageXml, inps.fixImageVrt)

    length, width, ref_type = image_info(reference, inps)
    sec_length, sec_width, sec_type = image_info(secondary, inps)
    readers = ['gdal' if use_gdal(fname, inps) else 'raw' for fname in (reference, secondary)]
    for fname, reader, size in zip((reference, secondary), readers,
                                   ((length, width, ref_type), (sec_length, sec_width, sec_type))):
        if reader == 'raw':
            check_raw_image(fname, *size)
    print(f'image length, width: {length}, {width} ({ref_type}, {sec_type})')

    # if using a gross offset, enlarge the margin
    margin = max(inps.margin, abs(inps.azshift), abs(inps.rgshift))

    # the starting pixels of the windows in the reference image: given (their first pixels, or
    # their centers), or after the margin and the search range
    if inps.startpixeldw is not None:
        start_down = inps.startpixeldw - (inps.winhgt // 2 if inps.use_center else 0)
    else:
        start_down = margin + inps.srchgt
    if inps.startpixelac is not None:
        start_across = inps.startpixelac - (inps.winwidth // 2 if inps.use_center else 0)
    else:
        start_across = margin + inps.srcwidth
    print(f'the first pixel in reference image is: ({start_down}, {start_across})')
    print(f'the center pixel of the first window is : ({start_down + inps.winhgt // 2}, '
          f'{start_across + inps.winwidth // 2})')
    if start_down < inps.srchgt:
        print(f'Warning: the starting pixel down {start_down} is less than the search range {inps.srchgt}. Zeros will be patched')
    if start_across < inps.srcwidth:
        print(f'Warning: the starting pixel across {start_across} is less than the search range {inps.srcwidth}. Zeros will be patched')

    # the ending pixels: given, or before the search range and the margin
    if inps.endpixeldw is not None:
        end_down = inps.endpixeldw + (inps.winhgt // 2 if inps.use_center else 0)
    else:
        end_down = length - margin - inps.srchgt
    if inps.endpixelac is not None:
        end_across = inps.endpixelac + (inps.winwidth // 2 if inps.use_center else 0)
    else:
        end_across = width - margin - inps.srcwidth

    # the number of windows down and across: the size of the offset field
    n_down = inps.numWinDown if inps.numWinDown > 0 else (end_down - start_down - inps.winhgt) // inps.skiphgt
    n_across = inps.numWinAcross if inps.numWinAcross > 0 else (end_across - start_across - inps.winwidth) // inps.skipwidth
    print(f'the number of windows: {n_down} by {n_across}')
    last_down = start_down + (n_down - 1) * inps.skiphgt + inps.winhgt
    last_across = start_across + (n_across - 1) * inps.skipwidth + inps.winwidth
    print(f'the last pixel in reference image is: ({last_down - 1}, {last_across - 1})')
    if last_down > length - inps.srchgt:
        print(f'Warning: the ending pixel down {last_down} is more than length {length} - search range {inps.srchgt}')
    if last_across > width - inps.srcwidth:
        print(f'Warning: the ending pixel across {last_across} is more than width {width} - search range {inps.srcwidth}')

    # the gross offsets: constant, or one per window from a file of int32 pairs (down, across)
    if inps.gross == 0:
        print(f'Set constant grossOffset ({inps.azshift}, {inps.rgshift})')
        gross_down, gross_across = inps.azshift, inps.rgshift
    else:
        print(f'Set varying grossOffset from file {inps.gross_offset_file}')
        gross = array.array('i')
        with open(inps.gross_offset_file, 'rb') as f:
            gross.frombytes(f.read())
        if len(gross) != 2 * n_down * n_across:
            raise RuntimeError(f'The gross offsets in {inps.gross_offset_file} do not match the number of '
                               f'windows: {n_down} by {n_across} pairs of int32')
        gross_down, gross_across = list(gross[0::2]), list(gross[1::2])

    # the outputs
    fbase = f'{inps.outprefix}{inps.outsuffix}'
    outputs = dict(offsetImageName=fbase + '.bip', grossOffsetImageName=fbase + '_gross.bip',
                   snrImageName=fbase + '_snr.bip', covImageName=fbase + '_cov.bip',
                   peakValueImageName=fbase + '_peak.bip')
    for name, fname in outputs.items():
        print(f'{name}: {fname}')

    # the ampcor object (GPU, or CPU if asked or without CUDA support), set up to run
    params = dict(
        workflow=inps.workflow, algorithm=inps.algorithm, derampMethod=inps.deramp,
        derampAxis=inps.deramp_axis,
        referenceImageName=reference, referenceImageHeight=length, referenceImageWidth=width,
        referenceImageDataType=ref_type, referenceImageReader=readers[0],
        secondaryImageName=secondary, secondaryImageHeight=sec_length, secondaryImageWidth=sec_width,
        secondaryImageDataType=sec_type, secondaryImageReader=readers[1],
        windowSizeHeight=inps.winhgt, windowSizeWidth=inps.winwidth,
        halfSearchRangeDown=inps.srchgt, halfSearchRangeAcross=inps.srcwidth,
        skipSampleDown=inps.skiphgt, skipSampleAcross=inps.skipwidth,
        referenceStartPixelDownStatic=start_down, referenceStartPixelAcrossStatic=start_across,
        numberWindowDown=n_down, numberWindowAcross=n_across,
        rawDataOversamplingFactor=inps.raw_oversample, corrStatWindowSize=inps.corr_stat_win_size,
        corrSurfaceZoomInWindow=2 * inps.corr_srch_size * inps.raw_oversample,
        corrSurfaceOverSamplingMethod=inps.corr_oversamplemethod,
        corrSurfaceOverSamplingFactor=inps.corr_oversample,
        mergeGrossOffset=inps.merge_gross_offset,
        numberWindowDownInChunk=inps.numWinDownInChunk, numberWindowAcrossInChunk=inps.numWinAcrossInChunk,
        mmapSize=inps.mmapsize,
        grossOffsetDown=gross_down, grossOffsetAcross=gross_across,
        **outputs)
    if inps.cpu or not pycuampcor.has_cuda:
        ampcor = pycuampcor.PyCPUAmpcor(nThreads=inps.nthreads, **params)
    else:
        ampcor = pycuampcor.PyCuAmpcor(deviceID=inps.gpuid, nStreams=inps.nstreams, **params)
    print('deramp method:', ampcor.derampMethod, ', axis:', ampcor.derampAxis)
    print('correlation surface zoom-in window size:', ampcor.corrSurfaceZoomInWindow)
    ampcor.checkPixelInImageRange()

    # the grid of the offset field in the reference image (centers of the windows)
    geometry = dict(x_start=start_across + inps.winwidth // 2, y_start=start_down + inps.winhgt // 2,
                    x_step=inps.skipwidth, y_step=inps.skiphgt, x_win_num=n_across, y_win_num=n_down)

    if not inps.redo and os.path.exists(outputs['offsetImageName']):
        print(f"offset field file: {outputs['offsetImageName']} exists and w/o redo, skip re-estimation.")
        return ampcor, geometry

    double = ampcor.isDoublePrecision()
    print(f"Running {type(ampcor).__name__} in {'DOUBLE' if double else 'FLOAT'} precision")
    ampcor.runAmpcor()
    print('Finished')

    # the headers of the outputs
    for name, bands in (('offsetImageName', 2), ('grossOffsetImageName', 2), ('snrImageName', 1),
                        ('covImageName', 3), ('peakValueImageName', 1)):
        write_header(outputs[name], n_across, n_down, bands, double)
    return ampcor, geometry


def prepare_geometry(full_dir, out_dir, x_start, y_start, x_step, y_step, x_win_num, y_win_num,
                     fbases=('lat', 'lon', 'los', 'hgt', 'z', 'shadowMask', 'waterMask')):
    """
    Generate the geometry datasets on the grid of the offset field from the full resolution
    geometry datasets (with GDAL)
    @param full_dir the input geometry directory in full resolution
    @param out_dir the output geometry directory
    @param x_start,y_start the starting column/row (the center of the first window)
    @param x_step,y_step the output pixel step in column/row direction
    @param x_win_num,y_win_num the number of columns/rows
    """
    from osgeo import gdal
    gdal.UseExceptions()
    full_dir = os.path.abspath(full_dir)
    out_dir = os.path.abspath(out_dir)

    # grab the file extension for full resolution file
    full_exts = ['.rdr.full', '.rdr'] if full_dir != out_dir else ['.rdr.full']
    full_exts = [e for e in full_exts if os.path.isfile(os.path.join(full_dir, f'{fbases[0]}{e}'))]
    if len(full_exts) == 0:
        raise ValueError(f'No full resolution {fbases[0]}.rdr* file found in: {full_dir}')
    full_ext = full_exts[0]

    print('-' * 50)
    print('generate the corresponding multi-looked geometry datasets using gdal ...')
    in_files = [os.path.join(full_dir, f'{f}{full_ext}') for f in fbases]
    in_files = [i for i in in_files if os.path.isfile(i)]
    fbases = [os.path.basename(i).split('.')[0] for i in in_files]
    out_files = [os.path.join(out_dir, f'{i}.rdr') for i in fbases]
    os.makedirs(out_dir, exist_ok=True)

    for in_file, out_file in zip(in_files, out_files):
        ds = gdal.Open(in_file, gdal.GA_ReadOnly)
        # starting column/row and column/row number
        src_win = [x_start, y_start, x_win_num * x_step, y_win_num * y_step]
        print(f'read {src_win} from file: {in_file}')

        # write binary data file
        print(f'write file: {out_file}')
        opts = gdal.TranslateOptions(format='ENVI', width=x_win_num, height=y_win_num,
                                     srcWin=src_win, noData=0)
        gdal.Translate(out_file, ds, options=opts)
        ds = None

        # write VRT file
        print(f'write file: {out_file}.vrt')
        ds = gdal.Open(out_file, gdal.GA_ReadOnly)
        gdal.Translate(out_file + '.vrt', ds, options=gdal.TranslateOptions(format='VRT'))
        ds = None


def main(iargs=None):
    """Estimate the offset field (and its geometry) with the command-line options iargs"""
    inps = cmd_line_parse(iargs)
    start_time = time.time()

    out_dir = os.path.dirname(inps.outprefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # estimate offset
    geometry = estimate_offset_field(inps.reference, inps.secondary, inps)[1]

    # generate geometry
    if inps.full_geometry_dir and inps.out_geometry_dir:
        prepare_geometry(inps.full_geometry_dir, inps.out_geometry_dir, **geometry)

    m, s = divmod(time.time() - start_time, 60)
    print(f'time used: {m:02.0f} mins {s:02.1f} secs.\n')


if __name__ == '__main__':
    main(sys.argv[1:])
