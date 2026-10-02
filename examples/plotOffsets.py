import glob
import os
import sys

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

gdal.UseExceptions()

# Find the data file of an ENVI header (GDAL opens ENVI datasets by the data file, not the .hdr)
# X.bip.hdr -> X.bip, or X.hdr -> X.<ext>
def envi_data_file(header_path):
    stem = header_path[:-len('.hdr')]
    if os.path.isfile(stem):
        return stem
    candidates = [f for f in glob.glob(glob.escape(stem) + '.*')
                  if not f.endswith(('.hdr', '.aux.xml', '.vrt'))]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Could not find a unique data file for the ENVI header {header_path}")
    return candidates[0]

# Function to open a GDAL-readable offset file (VRT, ENVI .bip, or .hdr) and read two bands
def read_offset_file(file_path):
    if file_path.endswith('.hdr'):
        file_path = envi_data_file(file_path)
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"Could not open {file_path}")

    # Read the raster bands (assuming two bands: Azimuth Offset and Range Offset)
    band1 = dataset.GetRasterBand(1)
    band2 = dataset.GetRasterBand(2)

    azimuth_data = band1.ReadAsArray()
    range_data = band2.ReadAsArray()

    return azimuth_data, range_data

# Function to plot Azimuth Offset and Range Offset data as 2D color maps
def plot_color_maps(azimuth_data, range_data, vmin_azimuth=None, vmax_azimuth=None, vmin_range=None, vmax_range=None, cmap="viridis"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the Azimuth Offset band
    im_azimuth = axes[0].imshow(azimuth_data, cmap=cmap, vmin=vmin_azimuth, vmax=vmax_azimuth)
    axes[0].set_title("Azimuth Offset")
    plt.colorbar(im_azimuth, ax=axes[0])

    # Plot the Range Offset band
    im_range = axes[1].imshow(range_data, cmap=cmap, vmin=vmin_range, vmax=vmax_range)
    axes[1].set_title("Range Offset")
    plt.colorbar(im_range, ax=axes[1])

    plt.tight_layout()
    plt.show()

# Function to plot 3D surface plots for Azimuth Offset and Range Offset
def plot_3d_surfaces(azimuth_data, range_data, vmin_azimuth=None, vmax_azimuth=None, vmin_range=None, vmax_range=None, cmap="viridis"):

    # exclude points out of the plot range
    azimuth_data = np.where((azimuth_data < vmin_azimuth) | (azimuth_data > vmax_azimuth), np.nan, azimuth_data)
    range_data = np.where((range_data < vmin_range) | (range_data > vmax_range), np.nan, range_data)

    y = np.arange(azimuth_data.shape[1])
    x = np.arange(azimuth_data.shape[0])
    x, y = np.meshgrid(x, y)

    ratio = azimuth_data.shape[1]/azimuth_data.shape[0]
    if ratio >1 :
        box_aspect = (1/ratio, 1, 1)
    else:
        box_aspect = (1, ratio, 1)
    print(ratio, box_aspect)


    fig = plt.figure(figsize=(14, 7))

    # Plot the Azimuth Offset 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(x, y, azimuth_data.T, cmap=cmap, edgecolor='none', vmin=vmin_azimuth, vmax=vmax_azimuth)
    ax1.set_title("Azimuth Offset - 3D Surface Plot")
    # ax1.view_init(elev=30, azim=-60)  # Adjust view angle for Azimuth Offset
    ax1.set_zlim(vmin_azimuth, vmax_azimuth)
    ax1.set_box_aspect(box_aspect)
    plt.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Plot the Range Offset 3D surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(x, y, range_data.T, cmap=cmap, edgecolor='none', vmin=vmin_range, vmax=vmax_range)
    ax2.set_title("Range Offset - 3D Surface Plot")
    # ax2.view_init(elev=30, azim=-60)  # Adjust view angle for Range Offset
    ax2.set_zlim(vmin_range, vmax_range)
    ax2.set_box_aspect(box_aspect)
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()


# Function to plot the offset magnitude with the offset vector field
def plot_velocity(azimuth_data, range_data, grid=20, vmin_azimuth=None, vmax_azimuth=None, vmin_range=None, vmax_range=None, cmap="viridis"):

    # exclude points out of the plot range
    azimuth_data = np.where((azimuth_data < vmin_azimuth) | (azimuth_data > vmax_azimuth), np.nan, azimuth_data)
    range_data = np.where((range_data < vmin_range) | (range_data > vmax_range), np.nan, range_data)

    # velocity
    velocity = np.sqrt(np.square(azimuth_data)+np.square(range_data))


    height, width = azimuth_data.shape

    y, x = np.mgrid[0:height:grid, 0:width:grid]

    u = azimuth_data[::grid, ::grid]
    v = range_data[::grid, ::grid]

    ratio = azimuth_data.shape[1] / azimuth_data.shape[0]

    fig = plt.figure(figsize=(7, 14))

    # Plot the offset magnitude and the offset vectors
    # (x: range/column, y: azimuth/row; angles='xy' follows the image's downward row axis)
    ax1 = fig.add_subplot(111)
    ax1.set_title("Offset vector Plot")
    vmap = ax1.imshow(velocity, cmap=cmap, vmin=0, vmax=np.nanmax(velocity))
    plt.colorbar(vmap, ax=ax1)
    ax1.quiver(x, y, v, u, color='r', angles='xy')

    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Azimuth Offset and Range Offset from an offset file (VRT, ENVI .bip, or .hdr) - cmap, 3d surface, or velocity map.")
    parser.add_argument("offset_file", type=str, help="Path to the offset file (.vrt, .bip, or .hdr)")
    parser.add_argument("--plot_3d", action="store_true", help="3D surface plot for both offsets")
    parser.add_argument("--plot_velocity", action="store_true", help="Velocity map for the offsets")
    parser.add_argument("--velocity_grid", type=int, default=50, help="The velocity field sparsity")
    parser.add_argument("--range", type=float, default=None, help="One Range(+/-) for both Offset colormaps")
    parser.add_argument("--pixel_unit_azimuth", type=float, default=1, help="Pixel size along azimuth")
    parser.add_argument("--pixel_unit_range", type=float, default=1, help="Pixel size along range")
    parser.add_argument("--vmin_azimuth", type=float, default=None, help="Minimum value for Azimuth Offset colormap")
    parser.add_argument("--vmax_azimuth", type=float, default=None, help="Maximum value for Azimuth Offset colormap")
    parser.add_argument("--vmin_range", type=float, default=None, help="Minimum value for Range Offset colormap")
    parser.add_argument("--vmax_range", type=float, default=None, help="Maximum value for Range Offset colormap")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name (default: viridis)")


    args = parser.parse_args()

    try:
        azimuth_data, range_data = read_offset_file(args.offset_file)
    except Exception as e:
        sys.exit(f"Error: {e}")

    # scale the offsets with pixel size
    if args.pixel_unit_azimuth != 1:
        azimuth_data *= args.pixel_unit_azimuth
    if args.pixel_unit_range != 1:
        range_data *= args.pixel_unit_range

    # set the plot/colormap range
    if args.range is not None:
        vmin_azimuth = -args.range
        vmax_azimuth = args.range
        vmin_range = -args.range
        vmax_range = args.range
    else:
        vmin_azimuth = args.vmin_azimuth if args.vmin_azimuth is not None else np.nanmin(azimuth_data)
        vmax_azimuth = args.vmax_azimuth if args.vmax_azimuth is not None else np.nanmax(azimuth_data)
        vmin_range = args.vmin_range if args.vmin_range is not None else np.nanmin(range_data)
        vmax_range = args.vmax_range if args.vmax_range is not None else np.nanmax(range_data)

    # plot surface 3d when requested
    if args.plot_3d:
        plot_3d_surfaces(
            azimuth_data,
            range_data,
            vmin_azimuth,
            vmax_azimuth,
            vmin_range,
            vmax_range,
            args.cmap
        )
    # plot the velocity intensity map and vector field
    elif args.plot_velocity:
        plot_velocity(
            azimuth_data,
            range_data,
            args.velocity_grid,
            vmin_azimuth,
            vmax_azimuth,
            vmin_range,
            vmax_range,
            args.cmap
        )
    else:
        # Plot the color maps
        plot_color_maps(
            azimuth_data,
            range_data,
            vmin_azimuth, vmax_azimuth, vmin_range, vmax_range,
            args.cmap
        )
