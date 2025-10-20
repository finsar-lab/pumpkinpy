"""Functions to reproject LOS data into it's VH, VV components.

Functions for the reprojection of TS-InSAR LOS data into vertical and
horizontal components by means of a regular spaced grid

@author: Héctor Quiroz
"""
from functools import partial

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import typer
from joblib import Parallel, delayed
from kneed import KneeLocator
from rich.console import Console

from pumpkinpy.config import get_config

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
console = Console()

def preprocess_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess the data frame for reprojection.

    Converts dates into pandas.DateFrame and orders columns into coordinates,
    angles and dates

    Parameters
    ----------
    df : pd.DataFrame
        The data frame to preprocess

    Returns
    -------
    total_dates_df: pandas.DataFrame
        A data frame containing pandas Timestamp objects representing the
        total dates.
    processed_df: pandas.DataFrame
        New data frame with coordinates, angles and total dates as
        pandas.Timestamp list

    """
    coordinates = df[["X", "Y"]]

    date_cols = [col for col in df.columns if col.startswith("D2")]
    total_dates_df = df[date_cols]
    total_dates_df.columns = [pd.to_datetime(col[1:]) for col in date_cols]

    angles = df[["incidence_angle", "azimuth_angle"]]

    processed_df = pd.concat([coordinates, angles, total_dates_df], axis=1)

    return processed_df, total_dates_df


def gridding(total_points: pd.DataFrame, cell_size: int) -> gpd.GeoDataFrame:
    """Create a grid of the specified cell size.

    Parameters
    ----------
    total_points : pandas.DataFrame
        Total points from both datasets (Just X and Y coordinates)
    cell_size : int
        Specified cell size for this grid

    Returns
    -------
    geopandas.GeoDataFrame
        The grid created with cells with the specified cell_size

    """
    gdf = gpd.GeoDataFrame(
            total_points,
            geometry=gpd.points_from_xy(total_points.X, total_points.Y)
    )
    gdf = gdf.drop(columns=["X", "Y"])
    xmin, ymin, xmax, ymax = gdf.total_bounds

    buffer = cell_size * 0.01  # 1% buffer
    xmin -= buffer
    ymin -= buffer
    xmax += buffer
    ymax += buffer

    grid_cells = []

    x_coords = np.arange(xmin, xmax + cell_size, cell_size)
    y_coords = np.arange(ymin, ymax + cell_size, cell_size)

    # Merges the coords into (X0[i], Y0[i])
    x0, y0 = np.meshgrid(x_coords, y_coords)

    x0_flat = x0.ravel()
    y0_flat = y0.ravel()
    x1_flat = x0_flat + cell_size
    y1_flat = y0_flat + cell_size

    # Polygons defining each grid.
    grid_cells = shapely.box(x0_flat, y0_flat, x1_flat, y1_flat)

    # New dataframe made of polygons (The grid)
    return gpd.GeoDataFrame(grid_cells, columns=["geometry"])

def grid_dissolve(
        polygon_grid: gpd.GeoDataFrame,
        asc_df: pd.DataFrame,
        desc_df: pd.DataFrame
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Fit every asc and desc points into the grid.

    Parameters
    ----------
    polygon_grid : geopandas.DataFrame
        The grid with a defined cell size
    asc_df : pandas.DataFrame
        Ascending dataframe
    desc_df : pandas.DataFrame
        Descending dataframe

    Returns
    -------
    asc_within_grid : geopandas.GeoDataFrame
        Every point interesected with the grid for asc geodataframe
    desc_within_grid : geopandas.GeoDataFrame
        Every point interesected with the grid for desc geodataframe

    """
    asc_gdf = gpd.GeoDataFrame(
        asc_df, geometry=gpd.points_from_xy(asc_df.X, asc_df.Y)
    )
    asc_gdf = asc_gdf.drop(columns=["X", "Y"])

    desc_gdf = gpd.GeoDataFrame(
        desc_df, geometry=gpd.points_from_xy(desc_df.X, desc_df.Y)
    )
    desc_gdf = desc_gdf.drop(columns=["X", "Y"])

    asc_within_grid = gpd.sjoin(
        asc_gdf, polygon_grid, how="left", predicate="within"
    )

    desc_within_grid = gpd.sjoin(
        desc_gdf, polygon_grid, how="left", predicate="within"
    )

    return asc_within_grid, desc_within_grid

def process_cellsize(
        cell_size: int,
        total_points: pd.DataFrame,
        asc_df: pd.DataFrame,
        desc_df: pd.DataFrame
    ) -> tuple[int, float]:
    """Process a single cell size.

    Parameters
    ----------
    cell_size : int
        Specified cell size for this grid
    total_points : pandas.DataFrame
        Total points from both datasets (Just X and Y coordinates)
    asc_df : pandas.DataFrame
        Ascending dataframe
    desc_df : pandas.DataFrame
        Descendign dataframe

    Returns
    -------
    geopandas.GeoDataFrame
        The grid created with cells with the specified cell_size

    """
    polygon_grid = gridding(total_points, cell_size)
    asc_within_grid, desc_within_grid = grid_dissolve(
        polygon_grid, asc_df, desc_df
    )

    common_indexes = len(set(asc_within_grid["index_right"]).intersection(
        desc_within_grid["index_right"]
    ))

    totcell = int(polygon_grid.count().iloc[0])
    ratio = totcell / common_indexes
    return totcell, ratio

def get_best_cell_size(asc_df: pd.DataFrame, desc_df: pd.DataFrame) -> int:
    """Get the best cell size possible."""
    config = get_config()
    merged_lats = pd.concat([asc_df["X"], desc_df["X"]])
    merged_longs = pd.concat([asc_df["Y"], desc_df["Y"]])
    total_points = pd.concat([
        merged_lats.to_frame(),
        merged_longs.to_frame()],
        axis=1
    )

    cell_sizes = list(np.arange(15, 216, 10))

    process_func = partial(
        process_cellsize,
        total_points=total_points,
        asc_df=asc_df,
        desc_df=desc_df
    )

    results = Parallel(n_jobs=16, batch_size="auto", verbose=5)(
        delayed(process_func)(e) for e in cell_sizes
    )

    # Unpack results
    raw_totcells, raw_ratios = zip(*results, strict=False)
    totcells: list[int] = list(raw_totcells)
    ratios: list[float] = list(raw_ratios)

    for_grid_size_asses = pd.DataFrame(
        {
            "cellsize": cell_sizes,
            "totcells": totcells,
            "ratios": ratios,
        }
    )

    gridsize = KneeLocator(
        for_grid_size_asses["cellsize"],
        for_grid_size_asses["ratios"],
        curve="convex",
        direction="decreasing",
    )

    plt.savefig(config.paths.image_dir / "knee_plot.png")

    if gridsize.knee is None:
        msg = "Knee point could not be determined."
        raise ValueError(msg)

    return int(gridsize.knee)


def run(asc_filename: str, desc_filename: str) -> None:
    """Reprojection's entry point."""
    config = get_config()
    raw_data_dir = config.paths.raw_data_dir

    asc_df = pd.read_parquet(raw_data_dir / f"{asc_filename}")
    desc_df = pd.read_parquet(raw_data_dir / f"{desc_filename}")

    asc_df, total_dates_asc = preprocess_df(asc_df)
    desc_df, total_dates_desc = preprocess_df(desc_df)

    if config.reprojection.best_cell_size == 0:
        console.print(
            "If best cell size is known, place it under config.toml or pass "
            "it as a parameter to avoid extra calculations",
            style="bold yellow"
        )
        config.reprojection.best_cell_size = get_best_cell_size(asc_df, desc_df)

    console.print(f"Best cell size: [cyan]{config.reprojection.best_cell_size}[/cyan]")










# function resampling Asc and Desc time series by interpolation (choose the interpolation method at line 121)
# the number of time steps is automatically calculated as (total no. of days covered)/(no. of asc and desc acquisitions)
# input: asc formatted dates, desc formatted dates, grid with asc and desc data merged
# output: resampled grid, new interpolated dates
def resampling_ts(datelist_asc, datelist_desc, asc_within_grid, desc_within_grid):
    start_date = max(min(datelist_asc), min(datelist_desc))
    end_date = min(max(datelist_asc), max(datelist_desc))

    asc_trimmed_dates = [date for date in datelist_asc if start_date < date < end_date]
    desc_trimmed_dates = [date for date in datelist_desc if start_date < date < end_date]
    # grouped = asc_within_grid.groupby('index_right')
    # for index_right_value, group in grouped:
    #     print(f"Group: {index_right_value}")
    #     print(group)


    # Convert to int once
    asc_within_grid['index_right'] = asc_within_grid['index_right'].astype(int)
    desc_within_grid['index_right'] = desc_within_grid['index_right'].astype(int)

    # Find common keys
    keys_asc = set(asc_within_grid['index_right'].unique())
    keys_desc = set(desc_within_grid['index_right'].unique())
    common_keys = keys_asc & keys_desc
    common_dates = np.union1d(asc_trimmed_dates, desc_trimmed_dates)

    # Filter to only common keys BEFORE any operations
    asc_filtered = asc_within_grid[asc_within_grid['index_right'].isin(common_keys)]
    desc_filtered = desc_within_grid[desc_within_grid['index_right'].isin(common_keys)]

    def trim_date_columns(df, start_date, end_date, suffix: str):
        cols_to_keep = []
        for col in df.columns:
            try:
                col_date = pd.to_datetime(col)
                if start_date <= col_date <= end_date:
                    cols_to_keep.append(col)
            except:
                cols_to_keep.append(col)
        return df[cols_to_keep]

    asc_filtered = trim_date_columns(asc_filtered, start_date, end_date, "_asc")
    desc_filtered = trim_date_columns(desc_filtered, start_date, end_date, "_desc")

    final_grid = pd.concat([asc_filtered, desc_filtered], axis=0)

    frequency=Timedelta((end_date - start_date)/ int(len(common_dates)))
    frequency=Timedelta.round(frequency, "D")

    # Generates a new datelist from min date to max date each "frecuency". Like np.arange but with dates
    interpolated_dates = pd.date_range(common_dates.min(), common_dates.max(), freq=frequency)

    date_columns = []
    non_date_columns = []

    for col in final_grid.columns:
        try:
            pd.to_datetime(col)
            date_columns.append(col)
        except:
            non_date_columns.append(col)

    df_dates = final_grid[date_columns].copy()
    df_non_dates = final_grid[non_date_columns].copy()

    date_cols_sorted = sorted(date_columns, key=lambda x: pd.to_datetime(x))
    df_dates = df_dates[date_cols_sorted]

    df_dates_interpolated = df_dates.interpolate(method='linear', axis=1)

    # Remove first and last date as they can't be interpolated
    df_dates_interpolated = df_dates_interpolated.iloc[:, 1:-1]

    df_result = pd.concat([df_non_dates, df_dates_interpolated], axis=1)

    return df_result


def plot_interactive_grid(gdf, date_columns):
    """
    Create an interactive plot with a slider to browse through dates
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.15)

    initial_idx = 0
    initial_date = date_columns[initial_idx]

    # Get value ranges for consistent colorbars
    vh_cols = [f"Vh_{col}" for col in date_columns]
    vv_cols = [f"Vv_{col}" for col in date_columns]
    vh_min, vh_max = gdf[vh_cols].min().min(), gdf[vh_cols].max().max()
    vv_min, vv_max = gdf[vv_cols].min().min(), gdf[vv_cols].max().max()

    # Initial plots
    plot1 = gdf.plot(column=f"Vh_{initial_date}",
                     ax=ax1,
                     legend=True,
                     cmap='RdYlBu_r',
                     edgecolor='black',
                     linewidth=0.5,
                     vmin=vh_min,
                     vmax=vh_max,
                     legend_kwds={'label': 'VH Value', 'shrink': 0.8})
    ax1.set_title(f'VH - {initial_date}')

    plot2 = gdf.plot(column=f"Vv_{initial_date}",
                     ax=ax2,
                     legend=True,
                     cmap='RdYlBu_r',
                     edgecolor='black',
                     linewidth=0.5,
                     vmin=vv_min,
                     vmax=vv_max,
                     legend_kwds={'label': 'VV Value', 'shrink': 0.8})
    ax2.set_title(f'VV - {initial_date}')

    # Add slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Date Index',
        valmin=0,
        valmax=len(date_columns) - 1,
        valinit=initial_idx,
        valstep=1
    )

    # Update function
    def update(val):
        idx = int(slider.val)
        date = date_columns[idx]
        ax1.clear()
        ax2.clear()
        gdf.plot(column=f"Vh_{date}",
                 ax=ax1,
                 legend=True,
                 cmap='RdYlBu_r',
                 edgecolor='black',
                 linewidth=0.5,
                 vmin=vh_min,
                 vmax=vh_max,
                 legend_kwds={'label': 'VH Value', 'shrink': 0.8})
        ax1.set_title(f'VH - {date}')

        gdf.plot(column=f"Vv_{date}",
                 ax=ax2,
                 legend=True,
                 cmap='RdYlBu_r',
                 edgecolor='black',
                 linewidth=0.5,
                 vmin=vv_min,
                 vmax=vv_max,
                 legend_kwds={'label': 'VV Value', 'shrink': 0.8})
        ax2.set_title(f'VV - {date}')

        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# fuction for reprojection of Asc and Desc displacements along Vertical and Horizontal components
# input: resampled grid, Asc LOS angle (radians), Desc LOS angle (radians)
def reprojection(final_grid, optimal_grid, horz_az_angle=-90):
    """
    Reprojects Asc and Desc displacements along Horizontal and Vertical components
    Based on MintPy's asc_desc2horz_vert implementation

    Parameters:
    -----------
    resampled_grid : dict
        Dictionary with grid cells containing time series data and angles
    horz_az_angle : float
        Horizontal azimuth angle in degrees (default: -90, meaning East-West)
        Measured from the north with anti-clockwise direction as positive.
    """
    optimal_grid = optimal_grid.copy()
    keys_desc = set(final_grid['index_right'])
    optimal_grid = optimal_grid[optimal_grid.index.isin(keys_desc)]

    date_columns = []
    for col in final_grid.columns:
        try:
            pd.to_datetime(col)
            date_columns.append(col)
        except:
            pass

    n_cells = len(optimal_grid)
    n_dates = len(date_columns)
    vh_array = np.full((n_cells, n_dates), np.nan, dtype=np.float32)
    vv_array = np.full((n_cells, n_dates), np.nan, dtype=np.float32)

    # Index mapping
    index_to_position = {idx: pos for pos, idx in enumerate(optimal_grid.index)}

    cells = final_grid.groupby('index_right')
    for cell_index, points in cells:
        pos = index_to_position.get(cell_index)
        total_points = points.shape[0]

        inc_angles = np.deg2rad(points["incidence_angle"].values)
        az_angles = np.deg2rad(points["azimuth_angle"].values)

        G = np.zeros((total_points, 2), dtype=np.float32)
        G[:, 0] = np.sin(inc_angles) * np.cos(az_angles - np.deg2rad(horz_az_angle))
        G[:, 1] = np.cos(inc_angles)

        displacement_data = points[date_columns].values
        dhv = np.dot(np.linalg.pinv(G), displacement_data)

        vh_array[pos, :] = dhv[0, :]
        vv_array[pos, :] = dhv[1, :]
        print(f"Done with: {cell_index}")

    # Create DataFrames from arrays
    vh_df = pd.DataFrame(
        vh_array,
        index=optimal_grid.index,
        columns=[f"Vh_{col}" for col in date_columns]
    )
    vv_df = pd.DataFrame(
        vv_array,
        index=optimal_grid.index,
        columns=[f"Vv_{col}" for col in date_columns]
    )

    optimal_grid = pd.concat([optimal_grid, vh_df, vv_df], axis=1)
    plot_interactive_grid(optimal_grid, date_columns)
    return print("reprojection successful")
