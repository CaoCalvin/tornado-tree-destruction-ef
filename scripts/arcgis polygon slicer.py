import arcpy
import os
import time # For simple timing

# --- Ensure Spatial Analyst extension is available and checked out ---
try:
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
        print("Spatial Analyst extension checked out.")
    else:
        raise Exception("Spatial Analyst extension is not available.")
except Exception as e:
    print(e)
    # import sys; sys.exit()

# --- Configuration ---
polygon_fc = r"C:\Users\kevin\dev\tornado-tree-destruction-ef\tornado-tree-destruction-ef\data\2022_07_18_dugwal\Dugwal ArcGIS\Dugwal Damage.shp" # UPDATE THIS
value_for_polygons = 255  # CHANGED: Set to 255 for WHITE in 8-bit imagery
value_field = "FID" # Field in polygons containing 'value_for_polygons' (must be numeric)

input_raster_folder = r"C:\Users\kevin\dev\tornado-tree-destruction-ef\tornado-tree-destruction-ef\data\2022_07_18_dugwal\TIFFs"
# UPDATE THIS to your desired final output folder
output_mask_folder = r"C:\Users\kevin\dev\tornado-tree-destruction-ef\tornado-tree-destruction-ef\data\2022_07_18_dugwal\TIFFs\out_gdb_temp"

# --- Setup Scratch Workspace (File Geodatabase) ---
# IMPORTANT: Define a path for your scratch geodatabase.
# The script will try to create it if it doesn't exist.
scratch_folder = r"C:\temp\ArcGIS_Scratch" # Or any other suitable scratch location
scratch_gdb_name = "scratch.gdb"
scratch_gdb_path = os.path.join(scratch_folder, scratch_gdb_name)

# Create output and scratch folders if they don't exist
if not os.path.exists(output_mask_folder):
    os.makedirs(output_mask_folder)
    print(f"Created output folder: {output_mask_folder}")
if not os.path.exists(scratch_folder):
    os.makedirs(scratch_folder)
    print(f"Created scratch folder: {scratch_folder}")

# Create scratch GDB if it doesn't exist
if not arcpy.Exists(scratch_gdb_path):
    try:
        arcpy.CreateFileGDB_management(scratch_folder, scratch_gdb_name)
        print(f"Created scratch geodatabase: {scratch_gdb_path}")
    except arcpy.ExecuteError:
        print(f"Error creating scratch GDB: {scratch_gdb_path}")
        print(arcpy.GetMessages())
        # import sys; sys.exit() # Exit if GDB creation fails

arcpy.env.scratchWorkspace = scratch_gdb_path
arcpy.env.overwriteOutput = True # Allow overwriting of temporary data

# Set the workspace for listing rasters
arcpy.env.workspace = input_raster_folder
original_tifs = arcpy.ListRasters("*", "TIF")
# original_tifs = ["22_Dugwal_503000_5379000.tif"] # Test with one specific file
if not original_tifs:
    print(f"No TIF files found in {input_raster_folder}")
else:
    print(f"Found {len(original_tifs)} TIF files to process.")

for tif_name in original_tifs:
    original_tif_path = os.path.join(input_raster_folder, tif_name)
    base_name = os.path.splitext(tif_name)[0]
    # Sanitize base_name for use in GDB (remove characters not allowed in GDB item names if necessary)
    # For GDB, names must start with a letter, and can contain letters, numbers, and underscores.
    # Let's keep them relatively simple for intermediate steps.
    # ArcPy often handles sanitization but being explicit can help.
    sanitized_base_name = arcpy.ValidateTableName(base_name)[:20] # Truncate to avoid overly long names

    print(f"\nProcessing: {original_tif_path} (Sanitized base: {sanitized_base_name})")

    # Define intermediate raster paths (will be in scratch GDB)
    # GDB item names cannot have extensions like .tif
    poly_to_raster_temp_name = f"ptr_{sanitized_base_name}" # Shorter, GDB friendly
    poly_to_raster_temp_path = os.path.join(arcpy.env.scratchWorkspace, poly_to_raster_temp_name)

    con_output_temp_name = f"con_{sanitized_base_name}" # Shorter, GDB friendly
    con_output_temp_path = os.path.join(arcpy.env.scratchWorkspace, con_output_temp_name)

    # Define final output path
    final_mask_name = f"{base_name}_mask_8bit.tif"
    final_mask_path = os.path.join(output_mask_folder, final_mask_name)

    # List of temporary items to delete in this iteration
    temp_items_to_delete = []

    try:
        # --- Get properties from ORIGINAL raster ---
        cell_size_x_result = arcpy.GetRasterProperties_management(original_tif_path, "CELLSIZEX")
        source_cellSize = float(cell_size_x_result.getOutput(0))

        desc_raster = arcpy.Describe(original_tif_path)
        source_extent = desc_raster.extent
        source_spatial_reference = desc_raster.spatialReference # Good to have if needed

        # CRITICAL ENVIRONMENT SETTINGS:
        arcpy.env.snapRaster      = original_tif_path
        arcpy.env.extent          = original_tif_path
        arcpy.env.cellSize        = original_tif_path
        # (OPTIONAL) arcpy.env.outputCoordinateSystem = desc_raster.spatialReference
        
        # Now all downstream raster ops—including PolygonToRaster—will
        # use the same origin, cell size, and grid alignment as the TIFF.


        print(f"  Source Cell Size: {source_cellSize}, Source Extent: {source_extent.XMin} {source_extent.YMin} {source_extent.XMax} {source_extent.YMax}")
        print(f"  Env SnapRaster: {arcpy.env.snapRaster}")
        print(f"  Env Extent: {arcpy.env.extent}")
        print(f"  Env CellSize: {arcpy.env.cellSize}")
        print(f"  Final TIF output: {final_mask_path}")

        # --- Step 1: Create a temporary feature class with constant value ---
        temp_fc_name = f"temp_fc_{sanitized_base_name}"
        temp_fc_path = os.path.join(arcpy.env.scratchWorkspace, temp_fc_name)
        
        if arcpy.Exists(temp_fc_path): arcpy.Delete_management(temp_fc_path)
        
        # Copy the input polygon to create a temporary feature class
        arcpy.CopyFeatures_management(polygon_fc, temp_fc_path)
        
        # Add a field for consistent white value
        arcpy.AddField_management(temp_fc_path, "WHITE_VAL", "SHORT")
        
        # Calculate the field to have value 255 (white)
        arcpy.CalculateField_management(temp_fc_path, "WHITE_VAL", "255")
        
        temp_items_to_delete.append(temp_fc_path)

        # --- Step 2: Polygon to Raster using the WHITE_VAL field ---
        print(f"  Step 1: PolygonToRaster to {poly_to_raster_temp_path}...")
        if arcpy.Exists(poly_to_raster_temp_path): arcpy.Delete_management(poly_to_raster_temp_path)
        arcpy.conversion.PolygonToRaster(
            in_features=temp_fc_path,
            value_field="WHITE_VAL",  # Use the field with constant 255 value
            out_rasterdataset=poly_to_raster_temp_path,
            cell_assignment="CELL_CENTER",
            priority_field="",
            cellsize=source_cellSize # Explicitly pass cell size to the tool as well
        )
        temp_items_to_delete.append(poly_to_raster_temp_path)

        # --- Step 3: Use Con (operates on in_memory raster, result is a raster object) ---
        print(f"  Step 2: Applying Con tool (result as raster object)...")
        in_raster_for_con_obj = arcpy.Raster(poly_to_raster_temp_path)
        # Con(IsNull(in_raster_obj), 0 (for background/black), 255 (constant white for polygons))
        out_con_raster_obj = arcpy.sa.Con(arcpy.sa.IsNull(in_raster_for_con_obj), 0, 255)
        # No .save() of out_con_raster_obj to in_memory, pass the object to CopyRaster

        # --- Step 4: Copy Raster to set final pixel type and format ---
        # Input for CopyRaster is the 'out_con_raster_obj' Raster object
        print(f"  Step 3: Copying raster object to {final_mask_path} with 8_BIT_UNSIGNED...")
        if arcpy.Exists(final_mask_path): arcpy.Delete_management(final_mask_path)
        arcpy.management.CopyRaster(
            in_raster=out_con_raster_obj, # Use the Con output RASTER OBJECT directly
            out_rasterdataset=final_mask_path,
            pixel_type="8_BIT_UNSIGNED",
            nodata_value=None # Assuming 0 and 255 are your only data values
        )
        print(f"  Successfully created final mask: {final_mask_name}")

        # --- Verification of dimensions (optional, for debugging) ---
        desc_final_raster = arcpy.Describe(final_mask_path)
        final_cols = arcpy.GetRasterProperties_management(final_mask_path, "COLUMNCOUNT").getOutput(0)
        final_rows = arcpy.GetRasterProperties_management(final_mask_path, "ROWCOUNT").getOutput(0)
        print(f"  Final Dimensions: {final_cols} cols x {final_rows} rows. CellSize: {desc_final_raster.meanCellWidth}x{desc_final_raster.meanCellHeight}")
        
    except arcpy.ExecuteError as exec_err:
        print(f"  ArcPy ExecuteError processing {tif_name}:")
        arcpy.AddError(arcpy.GetMessages(2))
        print(arcpy.GetMessages(2))
        print(f"  Error details: {exec_err}")

    except Exception as e:
        print(f"  An unexpected error occurred with {tif_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("  Cleaning up temporary items for this iteration...")
        for item_path in temp_items_to_delete:
            if arcpy.Exists(item_path):
                try:
                    arcpy.Delete_management(item_path)
                    print(f"    Deleted intermediate item: {item_path}")
                except Exception as del_e:
                    print(f"    Error deleting {item_path}: {del_e}")

        arcpy.ClearEnvironment("snapRaster")
        arcpy.ClearEnvironment("extent")

# --- Check in Spatial Analyst extension when done ---
arcpy.CheckInExtension("Spatial")
print("Spatial Analyst extension checked in.")
arcpy.ResetEnvironments()
print("\nProcessing complete.")