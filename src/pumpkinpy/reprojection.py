#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for the reprojection of TS-InSAR LOS data into vertical and 
horizontal components by means of a regular spaced grid

@author: Davide Festa
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from pandas import Timedelta
import math as mt


# function creating grid where to interpolate points
# input:dataframe containing lats and lons of both asc and desc points, list of cellsizes
# output:polygonal geodataframe containing cells grid
def gridding(total_points: pd.DataFrame, cell_size: int) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
            total_points,
            geometry=gpd.points_from_xy(total_points.X, total_points.Y)
    )
    gdf = gdf.drop(columns=['X', 'Y'])
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
    X0, Y0 = np.meshgrid(x_coords, y_coords)

    x0_flat = X0.ravel()
    y0_flat = Y0.ravel()
    x1_flat = x0_flat + cell_size
    y1_flat = y0_flat + cell_size

    # Polygons defining each grid.
    grid_cells = shapely.box(x0_flat, y0_flat, x1_flat, y1_flat)

    # New dataframe made of polygons (The grid)
    return gpd.GeoDataFrame(grid_cells, columns=['geometry'])

# function dissolving and mediating Asc and Desc time series within every cell
# input: cells grid, dataframe of coordinates ascending, dataframe of coordinates descending
# output: grid with asc and desc data merged
def grid_dissolve(
        polygon_grid: gpd.GeoDataFrame,
        asc_dataframe: pd.DataFrame,
        desc_dataframe: pd.DataFrame
    ) -> gpd.GeoDataFrame:

    asc_geodataframe = gpd.GeoDataFrame(
        asc_dataframe, geometry=gpd.points_from_xy(asc_dataframe.X, asc_dataframe.Y)
    )
    asc_geodataframe = asc_geodataframe.drop(columns=["X", "Y"])

    desc_geodataframe = gpd.GeoDataFrame(
        desc_dataframe, geometry=gpd.points_from_xy(desc_dataframe.X, desc_dataframe.Y)
    )
    desc_geodataframe = desc_geodataframe.drop(columns=["X", "Y"])

    asc_within_grid = gpd.sjoin(
        asc_geodataframe, polygon_grid, how="left", predicate="within"
    )

    desc_within_grid = gpd.sjoin(
        desc_geodataframe, polygon_grid, how="left", predicate="within"
    )
    # dissolve_asc = asc_within_grid.dissolve(by="index_right", aggfunc="mean") # Agreggates the displacement (NOT GEOMETRY)

    # Drop the geometry of the multipoint, only keep the polygons (Grid)
    #grid_asc = pd.merge(polygon_grid, asc_within_grid, left_index=True, right_index=True, how='inner')
    #cell_dissolve_asc = pd.merge(polygon_grid, dissolve_asc, left_index=True, right_index=True, how='inner').drop(columns=["geometry_y"])


    #dissolve_desc = merge_desc.dissolve(by="index_right", aggfunc="mean")
    # Drop the geometry of the multipoint, only keep the polygons (Grid)
    #cell_dissolve_desc = pd.merge(cell, dissolve_desc, left_index=True, right_index=True, how='inner').drop(columns=["geometry_y"])

    return asc_within_grid, desc_within_grid


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

    print(optimal_grid)
    print("-"*20)
    print("Empleado del mes")
    print("-"*20)
    exit()

    return print("reprojection successful")
