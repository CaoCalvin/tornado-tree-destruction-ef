# %%
import file_management
# %% [markdown]
# ## Data Loading
# Below we load our dataset and perform initial exploration.

# %%
file_management.tile_satellite_images(
    input_dir="../data/sof_wabasso/sof_wabasso_orig/",
    output_tile_dir="../data/sof_wabasso/sof_wabasso_tile/",
    output_png_dir="../data/sof_wabasso/sof_wabasso_png/",
    metadata_csv_path="../data/sof_wabasso/sof_wabasso_tile/sof_wabasso_tile_metadata.csv",
    tile_size=1024,
    tile_overlap=0,
    nodata_threshold=1.0
)

# %%
