import os
import glob
import rasterio
from rasterio.windows import Window
import numpy as np

# --- Configuration ---
SATELLITE_IMG_DIR = "path/to/your/satellite_images"
MASK_IMG_DIR = "path/to/your/mask_images"
OUTPUT_DIR = "path/to/your/output_crops"

CROP_SIZE = 1024
STRIDE = 512

# Define the pixel value for the 'white' class.
# For 8-bit unsigned images, 255 is standard for white.
# This value will be used directly for single-band masks,
# and for each channel (R,G,B) in 3-band masks if white is (255,255,255).
WHITE_PIXEL_VALUE = 255

# Thresholds for categorizing crops
MINIMAL_WHITE_THRESHOLD_PERCENT = 5.0 # Up to 5% white is considered "minimal"

# Output folder names
CONTAINS_WHITE_DIR = os.path.join(OUTPUT_DIR, "contains_white")
CONTAINS_MINIMAL_WHITE_DIR = os.path.join(OUTPUT_DIR, "contains_minimal_white")
NO_WHITE_DIR = os.path.join(OUTPUT_DIR, "no_white")
# --- End Configuration ---

def create_output_dirs():
    """Creates the output directories if they don't exist."""
    os.makedirs(CONTAINS_WHITE_DIR, exist_ok=True)
    os.makedirs(CONTAINS_MINIMAL_WHITE_DIR, exist_ok=True)
    os.makedirs(NO_WHITE_DIR, exist_ok=True)

def process_images():
    """Processes all satellite and mask TIFFs in the input directories."""
    create_output_dirs()
    
    satellite_image_paths = glob.glob(os.path.join(SATELLITE_IMG_DIR, "*.tif"))
    if not satellite_image_paths:
        print(f"No TIFF files found in satellite image directory: {SATELLITE_IMG_DIR}")
        return

    for sat_img_path in satellite_image_paths:
        basename = os.path.basename(sat_img_path)
        mask_basename_parts = os.path.splitext(basename)
        # Attempt to find a common mask naming convention (e.g., image.tif -> image_mask.tif)
        # Prioritize exact match, then try common suffices like _mask, _msk, etc.
        potential_mask_names = [
            f"{mask_basename_parts[0]}_mask{mask_basename_parts[1]}",
            f"{mask_basename_parts[0]}_msk{mask_basename_parts[1]}",
            f"mask_{mask_basename_parts[0]}{mask_basename_parts[1]}",
        ]
        mask_img_path = None
        for pmn in potential_mask_names:
            current_mask_path = os.path.join(MASK_IMG_DIR, pmn)
            if os.path.exists(current_mask_path):
                mask_img_path = current_mask_path
                break
        
        if not mask_img_path: # Fallback to a broader search if specific names not found
            core_name = mask_basename_parts[0] 
            # More generic search, be cautious if names are very ambiguous
            potential_masks_glob = glob.glob(os.path.join(MASK_IMG_DIR, f"*{core_name}*mask*{mask_basename_parts[1]}")) + \
                                   glob.glob(os.path.join(MASK_IMG_DIR, f"*{core_name}*msk*{mask_basename_parts[1]}"))
            if potential_masks_glob:
                 mask_img_path = potential_masks_glob[0] # Take the first match
                 if len(potential_masks_glob) > 1:
                     print(f"Warning: Multiple potential masks found for {basename}, using {os.path.basename(mask_img_path)}")
            
        if not mask_img_path or not os.path.exists(mask_img_path):
            print(f"Warning: Mask file not found for satellite image {sat_img_path} using common naming patterns. Skipping.")
            continue

        print(f"Processing: {basename} (Mask: {os.path.basename(mask_img_path)})")

        try:
            with rasterio.open(sat_img_path) as sat_src, rasterio.open(mask_img_path) as mask_src:
                if sat_src.width != mask_src.width or sat_src.height != mask_src.height:
                    print(f"Warning: Dimensions of {basename} ({sat_src.width}x{sat_src.height}) "
                          f"and its mask ({mask_src.width}x{mask_src.height}) do not match. Skipping.")
                    continue
                
                img_width = sat_src.width
                img_height = sat_src.height
                crop_count = 0

                for top in range(0, img_height - STRIDE, STRIDE):
                    for left in range(0, img_width - STRIDE, STRIDE):
                        actual_window_width = min(CROP_SIZE, img_width - left)
                        actual_window_height = min(CROP_SIZE, img_height - top)
                        
                        # Skip if the actual window is too small (e.g. smaller than stride/2 to avoid tiny slivers)
                        # This check can be adjusted or removed based on preference.
                        if actual_window_width < STRIDE / 2 or actual_window_height < STRIDE / 2:
                            if actual_window_width < CROP_SIZE and actual_window_height < CROP_SIZE : #Only skip if both are smaller than CROP_SIZE
                                 continue

                        read_window = Window(left, top, actual_window_width, actual_window_height)

                        sat_crop = sat_src.read(window=read_window)
                        mask_crop = mask_src.read(window=read_window) # Reads all bands: (bands, height, width)

                        if mask_crop.size == 0: continue # Should be caught by earlier width/height check

                        white_pixel_count = 0
                        # total_pixels_for_percentage is the spatial area of the crop
                        total_pixels_for_percentage = actual_window_height * actual_window_width 
                        if total_pixels_for_percentage == 0: continue


                        if mask_src.count == 1:
                            # Single-band mask
                            # mask_crop has shape (1, height, width), so we take the first element
                            mask_data_to_analyze = mask_crop[0]
                            white_pixel_count = np.count_nonzero(mask_data_to_analyze == WHITE_PIXEL_VALUE)
                        
                        elif mask_src.count == 3:
                            # Assuming 3-band RGB mask, white is (WHITE_PIXEL_VALUE, WHITE_PIXEL_VALUE, WHITE_PIXEL_VALUE)
                            # mask_crop has shape (3, height, width)
                            # Create a (3,1,1) array for comparison [R,G,B] with the WHITE_PIXEL_VALUE
                            white_rgb_target = np.array([WHITE_PIXEL_VALUE, WHITE_PIXEL_VALUE, WHITE_PIXEL_VALUE], 
                                                        dtype=mask_crop.dtype)[:, np.newaxis, np.newaxis]
                            is_white_pixel = np.all(mask_crop == white_rgb_target, axis=0)
                            white_pixel_count = np.sum(is_white_pixel)
                        
                        else: # Fallback for other band counts (e.g., 2, 4, or more bands)
                            print(f"  Warning: Mask {os.path.basename(mask_img_path)} for {basename} has {mask_src.count} bands. "
                                  f"Auto-detection for 'white' is set for 1 or 3 (RGB) bands. "
                                  f"Analyzing first band only using WHITE_PIXEL_VALUE={WHITE_PIXEL_VALUE}.")
                            mask_data_to_analyze = mask_crop[0] # Analyze the first band
                            white_pixel_count = np.count_nonzero(mask_data_to_analyze == WHITE_PIXEL_VALUE)

                        percentage_white = (white_pixel_count / total_pixels_for_percentage) * 100
                        
                        output_folder = NO_WHITE_DIR
                        if white_pixel_count > 0:
                            if percentage_white > MINIMAL_WHITE_THRESHOLD_PERCENT:
                                output_folder = CONTAINS_WHITE_DIR
                            else:
                                output_folder = CONTAINS_MINIMAL_WHITE_DIR
                        
                        crop_transform = sat_src.window_transform(read_window)
                        out_meta = sat_src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": sat_crop.shape[1], 
                            "width": sat_crop.shape[2],
                            "transform": crop_transform,
                            "count": sat_src.count 
                        })

                        crop_filename = f"{os.path.splitext(basename)[0]}_crop_y{top}_x{left}.tif" #More descriptive crop name
                        output_path = os.path.join(output_folder, crop_filename)

                        with rasterio.open(output_path, "w", **out_meta) as dst:
                            dst.write(sat_crop)
                        crop_count += 1
                
                print(f"  Finished processing {basename}, generated {crop_count} crops.")

        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening or reading {sat_img_path} or its mask: {e}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {basename}: {e}. Skipping.")

if __name__ == "__main__":
    if SATELLITE_IMG_DIR == "path/to/your/satellite_images" or \
       MASK_IMG_DIR == "path/to/your/mask_images" or \
       OUTPUT_DIR == "path/to/your/output_crops":
        print("Please update SATELLITE_IMG_DIR, MASK_IMG_DIR, and OUTPUT_DIR variables in the script.")
    else:
        process_images()
        print("Processing complete.")