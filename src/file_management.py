import os
import glob
import numpy as np
import rasterio # type: ignore
from rasterio.merge import merge # type: ignore
from rasterio.windows import Window # type: ignore
from rasterio.transform import Affine # type: ignore
from PIL import Image
import csv 
from beartype import beartype
from pathlib import Path
from typing import Union

@beartype
def tile_satellite_images(
    input_dir: Union[str, Path],
    output_tile_dir: Union[str, Path],
    output_png_dir: Union[str, Path],
    metadata_csv_path: Union[str, Path],
    tile_size: int = 1024,
    tile_overlap: int = 0,
    nodata_threshold: float = 0.95
):
    """
    Split georeferenced satellite TIFFs into tiles, convert to PNG, and export metadata CSV.
    """
    os.makedirs(output_tile_dir, exist_ok=True)
    os.makedirs(output_png_dir, exist_ok=True)

    # STEP 1: Read and merge all input TIFFs
    tif_paths = glob.glob(os.path.join(input_dir, "*.tif"))
    if not tif_paths:
        raise RuntimeError(f"No TIFF files found in {input_dir}")

    src_files_to_mosaic = [rasterio.open(p) for p in tif_paths]
    mosaic, out_transform = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    # STEP 2: Tile and export
    tile_metadata = []
    height, width = mosaic.shape[1], mosaic.shape[2]
    tile_count = 0

    for i in range(0, height, tile_size - tile_overlap):
        for j in range(0, width, tile_size - tile_overlap):
            tile = mosaic[:, i:i+tile_size, j:j+tile_size]

            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                continue  # skip incomplete tiles

            # Check for transparency using alpha if present
            if tile.shape[0] == 4:  # RGBA
                alpha = tile[3]
                if np.mean(alpha == 0) > nodata_threshold:
                    continue
            else:
                # Fall back to checking "all channels zero" if no alpha
                if np.mean(np.all(tile == 0, axis=0)) > nodata_threshold:
                    continue


            window = Window(j, i, tile_size, tile_size) # type: ignore
            transform = rasterio.windows.transform(window, out_transform) # type: ignore

            tile_filename = f"tile_{i}_{j}.tif"
            tile_path = os.path.join(output_tile_dir, tile_filename)

            tile_meta = out_meta.copy()
            tile_meta.update({
                "height": tile_size,
                "width": tile_size,
                "transform": transform
            })

            with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                dst.write(tile)

            # Convert to PNG for CVAT
            # PNG conversion
            if tile.shape[0] >= 4:
                # RGBA image with transparency
                tile_np = tile[:4]
                tile_np = np.clip(tile_np, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                Image.fromarray(tile_np, mode="RGBA").save(output_png_dir)
            else:
                # RGB image — add opaque alpha if needed
                tile_rgb = tile[:3] if tile.shape[0] >= 3 else np.tile(tile[0], (3, 1, 1))
                tile_rgb = np.clip(tile_rgb, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                alpha = np.full((tile_rgb.shape[0], tile_rgb.shape[1], 1), 255, dtype=np.uint8)
                tile_rgba = np.concatenate((tile_rgb, alpha), axis=-1)
                Image.fromarray(tile_rgba, mode="RGBA").save(output_png_dir)

            # Record metadata
            affine: Affine = transform
            tile_metadata.append({
                "filename": tile_filename,
                "x_origin": affine.c,
                "y_origin": affine.f,
                "pixel_width": affine.a,
                "pixel_height": affine.e,
                "row_offset": i,
                "col_offset": j
            })

            tile_count += 1

    # STEP 3: Write metadata CSV
    with open(metadata_csv_path, 'w', newline='') as csvfile:
        fieldnames = ["filename", "x_origin", "y_origin", "pixel_width", "pixel_height", "row_offset", "col_offset"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tile_metadata)

    print(f"✅ Done: Saved {tile_count} tiles.")
    print(f"🗂️ Tiles in: {output_tile_dir}")
    print(f"🖼️ CVAT-ready PNGs in: {output_png_dir}")
    print(f"📄 Metadata CSV: {metadata_csv_path}")
