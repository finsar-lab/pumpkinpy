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
from joblib import Parallel, delayed
from kneed import KneeLocator
from matplotlib.widgets import Slider
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from pumpkinpy.config import get_config

np.set_printoptions(precision=4, suppress=True)
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

    date_cols = df.columns[df.columns.str.startswith("D2")].tolist()
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
        geometry=gpd.points_from_xy(total_points.X, total_points.Y),
        columns=total_points.columns.difference(["X", "Y"])
    )
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

def df_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert DataFrame with X, Y to GeoDataFrame."""
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))
    return gdf.drop(columns=["X", "Y"])

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
    asc_gdf = df_to_gdf(asc_df)
    desc_gdf = df_to_gdf(desc_df)

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
    return totcell, float(ratio)

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

    results = Parallel(n_jobs=28, batch_size="auto", verbose=5)(
        delayed(process_func)(e) for e in cell_sizes
    )

    # Unpack results
    totcells, ratios = zip(*results, strict=False)

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


def resampling_ts(
    datelist_asc: pd.DataFrame,
    datelist_desc: pd.DataFrame,
    asc_within_grid: pd.DataFrame,
    desc_within_grid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Resample ascending and descending time series by interpolation.

    This function resamples time series data from ascending and descending satellite
    acquisitions by performing linear interpolation. The number of time steps is
    automatically calculated as (total number of days covered) / (number of ascending
    and descending acquisitions).

    Parameters
    ----------
    datelist_asc : List[pd.Timestamp]
        List of formatted dates for ascending acquisitions.
    datelist_desc : List[pd.Timestamp]
        List of formatted dates for descending acquisitions.
    asc_within_grid : pd.DataFrame
        DataFrame containing ascending data with 'index_right' column and date columns.
        The 'index_right' column identifies spatial grid points.
    desc_within_grid : pd.DataFrame
        DataFrame containing descending data with 'index_right' column and date columns.
        The 'index_right' column identifies spatial grid points.

    Returns
    -------
    df_result : pd.DataFrame
        Resampled grid with interpolated values. Contains non-date columns from input
        DataFrames plus interpolated date columns (excluding first and last dates).
    interpolated_dates : pd.DatetimeIndex
        New interpolated dates at regular frequency intervals spanning the common
        date range.

    """
    # Identify overlapping date range
    start_date = max(min(datelist_asc), min(datelist_desc))
    end_date = min(max(datelist_asc), max(datelist_desc))

    console.print(f"[cyan]Date range:[/cyan] {start_date.date()} to {end_date.date()}")

    asc_trimmed_dates = [
        date for date in datelist_asc if start_date < date < end_date
    ]
    desc_trimmed_dates = [
        date for date in datelist_desc if start_date < date < end_date
    ]

    console.print(
        f"[cyan]Trimmed acquisitions:[/cyan] {len(asc_trimmed_dates)} "
        f"ASC, {len(desc_trimmed_dates)} DESC"
    )

    # Convert to int once
    asc_within_grid["index_right"] = asc_within_grid["index_right"].astype(int)
    desc_within_grid["index_right"] = desc_within_grid["index_right"].astype(int)

    # Find common keys
    keys_asc = set(asc_within_grid["index_right"].unique())
    keys_desc = set(desc_within_grid["index_right"].unique())
    common_keys = keys_asc & keys_desc
    common_dates = np.union1d(asc_trimmed_dates, desc_trimmed_dates)

    console.print(f"[cyan]Common grid points:[/cyan] {len(common_keys)}")
    console.print(f"[cyan]Total unique dates:[/cyan] {len(common_dates)}")

    # Filter to only common keys BEFORE any operations
    asc_filtered = asc_within_grid[asc_within_grid["index_right"].isin(common_keys)]
    desc_filtered = desc_within_grid[desc_within_grid["index_right"].isin(common_keys)]

    def trim_date_columns(
        df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Trim DataFrame to only include date columns within specified range.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with date-named columns.
        start_date : pd.Timestamp
            Start date for filtering.
        end_date : pd.Timestamp
            End date for filtering.
        suffix : str
            Suffix identifier (not currently used but maintained for compatibility).

        Returns
        -------
        pd.DataFrame
            DataFrame with only columns within date range (plus non-date columns).

        """
        date_cols = []
        non_date_cols = []

        for col in df.columns:
            if isinstance(col, pd.Timestamp):
                if start_date <= col <= end_date:
                    date_cols.append(col)
            else:
                non_date_cols.append(col)

        return df[non_date_cols + date_cols]

    asc_filtered = trim_date_columns(asc_filtered, start_date, end_date)
    desc_filtered = trim_date_columns(desc_filtered, start_date, end_date)

    final_grid = pd.concat([asc_filtered, desc_filtered], axis=0)

    # Calculate frequency for interpolation
    frequency = pd.Timedelta((end_date - start_date) / len(common_dates))
    frequency = pd.Timedelta.round(frequency, "D")

    console.print(f"[cyan]Interpolation frequency:[/cyan] {frequency}")

    # Generate new datelist from min date to max date at regular frequency
    interpolated_dates = pd.date_range(common_dates.min(), common_dates.max(),
                                       freq=frequency)

    console.print(
        f"[cyan]Interpolated dates generated:[/cyan] {len(interpolated_dates)}"
    )

    date_columns = [col for col in final_grid.columns if isinstance(col, pd.Timestamp)]
    non_date_columns = [col for col in final_grid.columns if col not in date_columns]

    df_dates = final_grid[date_columns].copy()
    df_non_dates = final_grid[non_date_columns].copy()

    # Sort date columns chronologically
    date_cols_sorted = sorted(date_columns, key=lambda x: pd.to_datetime(x))
    df_dates = df_dates[date_cols_sorted]

    # Perform linear interpolation across time axis
    df_dates_interpolated = df_dates.interpolate(method="linear", axis=1)

    # Remove first and last date as they can't be reliably interpolated
    df_dates_interpolated = df_dates_interpolated.iloc[:, 1:-1]

    console.print("[green]Interpolation complete[/green]")
    console.print(
        f"[cyan]Final columns:[/cyan] {len(df_dates_interpolated.columns)} "
        f"date columns, {len(non_date_columns)} non-date columns"
    )

    df_result = pd.concat([df_non_dates, df_dates_interpolated], axis=1)

    return df_result, interpolated_dates

def process_single_cell(
    cell_index: int,
    points: pd.DataFrame,
    date_columns: list,
    horz_az_angle: float
) -> tuple[int, np.ndarray, np.ndarray]:
    """Process a single grid cell for reprojection.

    Parameters
    ----------
    cell_index : int
        Grid cell identifier
    points : pd.DataFrame
        Points belonging to this cell
    date_columns : list
        List of date column names
    horz_az_angle : float
        Horizontal azimuth angle in degrees

    Returns
    -------
    tuple[int, np.ndarray, np.ndarray]
        Cell index, horizontal displacements, vertical displacements
    """
    total_points = points.shape[0]

    # Convert angles to radians
    inc_angles = np.deg2rad(points["incidence_angle"].values)
    az_angles = np.deg2rad(points["azimuth_angle"].values)

    # Build geometry matrix G
    G = np.zeros((total_points, 2), dtype=np.float32)
    G[:, 0] = np.sin(inc_angles) * np.cos(az_angles - np.deg2rad(horz_az_angle))
    G[:, 1] = np.cos(inc_angles)

    # Get displacement data for all dates
    displacement_data = points[date_columns].values

    # Solve for horizontal and vertical components
    dhv = np.dot(np.linalg.pinv(G), displacement_data)

    return cell_index, dhv[0, :], dhv[1, :]


def reprojection (
    final_grid: pd.DataFrame,
    optimal_grid: gpd.GeoDataFrame,
    horz_az_angle: float = -90,
    n_jobs: int = -1
) -> gpd.GeoDataFrame:
    """Reprojects Asc and Desc displacements using parallel processing.

    Parameters
    ----------
    final_grid : pd.DataFrame
        DataFrame containing time series data with columns:
        - 'index_right': Grid cell identifiers
        - 'incidence_angle': Satellite incidence angles in degrees
        - 'azimuth_angle': Satellite azimuth angles in degrees
        - Date columns: Time series displacement values
    optimal_grid : gpd.GeoDataFrame
        Target grid DataFrame indexed by cell identifiers.
    horz_az_angle : float, optional
        Horizontal azimuth angle in degrees, default: -90.0 (East-West direction)
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available cores. Default: -1

    Returns
    -------
    optimal_grid : gpd.GeoDataFrame
        Input optimal_grid augmented with horizontal and vertical displacement columns
    """
    console.print("[bold cyan]Starting parallel displacement reprojection...[/bold cyan]")

    optimal_grid = optimal_grid.copy()

    keys_desc = set(final_grid["index_right"])
    optimal_grid = optimal_grid[optimal_grid.index.isin(keys_desc)]

    # Extract date columns
    date_columns = [col for col in final_grid.columns if isinstance(col, pd.Timestamp)]

    console.print(f"[cyan]Time steps:[/cyan] {len(date_columns)}")
    console.print(f"[cyan]Grid cells:[/cyan] {len(optimal_grid)}")
    console.print(f"[cyan]Parallel jobs:[/cyan] {n_jobs if n_jobs > 0 else 'all cores'}")

    # Prepare arrays
    n_cells = len(optimal_grid)
    n_dates = len(date_columns)
    vh_array = np.full((n_cells, n_dates), np.nan, dtype=np.float32)
    vv_array = np.full((n_cells, n_dates), np.nan, dtype=np.float32)

    # Index mapping
    index_to_position = {idx: pos for pos, idx in enumerate(optimal_grid.index)}

    # Group by cell
    cells = list(final_grid.groupby("index_right"))

    # Parallel processing with Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing grid cells...",
            total=len(cells)
        )

        # Process cells and update progress bar in real-time
        results = []
        for result in Parallel(n_jobs=n_jobs, batch_size="auto", verbose=0, return_as="generator")(
            delayed(process_single_cell)(
                cell_index,
                points,
                date_columns,
                horz_az_angle
            )
            for cell_index, points in cells
        ):
            results.append(result)
            progress.update(task, advance=1)

    for cell_index, vh_vals, vv_vals in results:
        pos = index_to_position[cell_index]
        vh_array[pos, :] = vh_vals
        vv_array[pos, :] = vv_vals

    console.print("[green]✓ Reprojection calculations complete[/green]")

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
    console.print(
        f"[cyan]Output columns:[/cyan] {len(vh_df.columns)} "
        f"horizontal + {len(vv_df.columns)} vertical"
    )
    console.print("[bold green]✓ Reprojection successful[/bold green]")

    return optimal_grid

def merge_coordinates(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merge X, Y coordinates from two DataFrames."""
    return pd.DataFrame({
        "X": pd.concat([df1["X"], df2["X"]]),
        "Y": pd.concat([df1["Y"], df2["Y"]])
    })

def substract_first_date(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Do a temporal normalization of each component."""
    vh_cols = gdf.columns[gdf.columns.str.startswith("Vh")].tolist()
    vv_cols = gdf.columns[gdf.columns.str.startswith("Vv")].tolist()

    gdf[vh_cols] = gdf[vh_cols].sub(gdf[vh_cols[0]], axis=0)
    gdf[vv_cols] = gdf[vv_cols].sub(gdf[vv_cols[0]], axis=0)

    return gdf

def plot_displacement_matplotlib(gdf, component='Vh'):
    """
    Create interactive matplotlib plot with slider for displacement time series

    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame with geometry column and time series columns
    component : str
        Either 'Vh' (horizontal) or 'Vv' (vertical)
    """

    # Extract column names
    cols = [col for col in gdf.columns if col.startswith(f'{component}_')]
    cols = sorted(cols)

    # Extract dates
    dates = [col.split('_')[1] for col in cols]

    # Get centroids
    gdf['centroid'] = gdf.geometry.centroid
    gdf['x'] = gdf.centroid.x
    gdf['y'] = gdf.centroid.y

    # Calculate global min/max for consistent color scale
    all_values = gdf[cols].values.flatten()
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    # Make color scale symmetric around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)

    # Initial plot
    scatter = ax.scatter(
        gdf['x'], gdf['y'],
        c=gdf[cols[0]],
        cmap='RdBu_r',
        vmin=vmin, vmax=vmax,
        s=50,
        edgecolors='gray',
        linewidth=0.5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{component} Displacement (m)', rotation=270, labelpad=20)

    # Set title and labels
    title = ax.set_title(f'{component} Displacement - {dates[0]}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Create slider axis
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(
        ax_slider, 'Date',
        0, len(cols) - 1,
        valinit=0,
        valstep=1,
        valfmt='%d'
    )

    def update(val):
        idx = int(slider.val)
        scatter.set_array(gdf[cols[idx]])
        title.set_text(f'{component} Displacement - {dates[idx]}')
        # Update slider label to show actual date
        ax_slider.set_xlabel(f'Date: {dates[idx]}', fontsize=10)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Set initial slider label
    ax_slider.set_xlabel(f'Date: {dates[0]}', fontsize=10)

    plt.show()

    return fig, ax, slider





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
            "it as an argument to avoid extra calculations",
            style="bold yellow"
        )
        config.reprojection.best_cell_size = get_best_cell_size(asc_df, desc_df)
    console.print(f"Best cell size: [cyan]{config.reprojection.best_cell_size}[/cyan]")

    total_points = merge_coordinates(asc_df, desc_df)

    polygon_grid = gridding(total_points, config.reprojection.best_cell_size)
    asc_within_grid, desc_within_grid = grid_dissolve(
        polygon_grid, asc_df, desc_df
    )

    interpolated_points, final_dates = resampling_ts(
        total_dates_asc, total_dates_desc, asc_within_grid, desc_within_grid
    )

    optimal_grid = reprojection(interpolated_points, polygon_grid, n_jobs=20)
    normalized_grid = substract_first_date(optimal_grid)

    fig_vh, ax_vh, slider_vh = plot_displacement_matplotlib(normalized_grid, component='Vh')
    fig_vv, ax_vv, slider_vv = plot_displacement_matplotlib(normalized_grid, component='Vv')
