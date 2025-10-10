#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for the reprojection of TS-InSAR LOS data into vertical and 
horizontal components by means of a regular spaced grid

@author: Davide Festa
"""
import numpy as np
import pandas as pd
import geopandas
import shapely
from pandas import Timedelta
import math as mt


# function creating grid where to interpolate points
# input:dataframe containing lats and lons of both asc and desc points, list of cellsizes
# output:polygonal geodataframe containing cells grid
def gridding(Lats_Lons_totalPoints, cell_size):
    gdf = geopandas.GeoDataFrame(Lats_Lons_totalPoints, 
            geometry=geopandas.points_from_xy(Lats_Lons_totalPoints.X, Lats_Lons_totalPoints.Y))
    gdf = gdf.drop(columns=['X', 'Y']) # Geometry with total points (POINT (Lat, Lon))
    xmin, ymin, xmax, ymax= gdf.total_bounds
    grid_cells = []
    for x0 in np.arange(xmin, xmax, cell_size):
        for y0 in np.arange(ymin, ymax, cell_size):
            # bounds
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            grid_cells.append( shapely.geometry.box(x0, y0, x1, y1)  ) # Creates a geometry with this corners
    cell = geopandas.GeoDataFrame(grid_cells, columns=['geometry']) # New dataframe made of polygons (The grid)
    
    return cell
    
# function dissolving and mediating Asc and Desc time series within every cell
# input: cells grid, dataframe of coordinates ascending, dataframe of coordinates descending
# output: grid with asc and desc data merged
def grid_dissolve(cell, dfcoords_asc, dfcoords_desc):
    #dfcoords is a dataframe with coordinates asc (X, Y) and every date
    
    gdf_dfcoords_asc=geopandas.GeoDataFrame(dfcoords_asc, geometry=geopandas.points_from_xy(dfcoords_asc.X, dfcoords_asc.Y))
    gdf_dfcoords_asc=gdf_dfcoords_asc.drop(columns=["X", "Y"])
    
    gdf_dfcoords_desc=geopandas.GeoDataFrame(dfcoords_desc, geometry=geopandas.points_from_xy(dfcoords_desc.X, dfcoords_desc.Y))
    gdf_dfcoords_desc=gdf_dfcoords_desc.drop(columns=["X", "Y"])

    # As before, creates a geodataframe with only point geometry (POINT(Lat, Lon)) and dates
    
    merge_asc = geopandas.sjoin(gdf_dfcoords_asc, cell, how="left", predicate="within")
    #print(merge_asc)
    #import time
    #time.sleep(2)
    dissolve_asc = merge_asc.dissolve(by="index_right", aggfunc="mean") # Agreggates the displacement (NOT GEOMETRY)
    #print(dissolve_asc)
    #exit()

    # Drop the geometry of the multipoint, only keep the polygons (Grid)
    cell_dissolve_asc = pd.merge(cell, dissolve_asc, left_index=True, right_index=True, how='inner').drop(columns=["geometry_y"])
        
    merge_desc = geopandas.sjoin(gdf_dfcoords_desc, cell, how="left", predicate="within")
    dissolve_desc = merge_desc.dissolve(by="index_right", aggfunc="mean")

    # Drop the geometry of the multipoint, only keep the polygons (Grid)
    cell_dissolve_desc = pd.merge(cell, dissolve_desc, left_index=True, right_index=True, how='inner').drop(columns=["geometry_y"])
    
    # Place both of the dissolved datasets into a dictionary
    datasets_names=["data_asc", "data_desc"]
    grid_with_asc_desc = dict(zip(datasets_names, [cell_dissolve_asc, cell_dissolve_desc]))
    
    return grid_with_asc_desc


# function resampling Asc and Desc time series by interpolation (choose the interpolation method at line 121)
# the number of time steps is automatically calculated as (total no. of days covered)/(no. of asc and desc acquisitions)
# input: asc formatted dates, desc formatted dates, grid with asc and desc data merged
# output: resampled grid, new interpolated dates
def resampling_ts(datelist_asc, datelist_desc, grid_with_asc_desc):

    # Datelist is an array of dates. Grid is the dissolved grid
    # grid_wih_asc_desc has the column geometry_x which is a multipolygon (grid)

    # Gets first and last date for both datastes
    t0asc = min(datelist_asc)
    t0desc = min(datelist_desc)
    tlast_asc = max(datelist_asc)
    tlast_desc = max(datelist_desc)

    start_date = max(t0asc, t0desc)
    end_date = min(tlast_asc, tlast_desc)

    # Removes incidence and azimuth for date calculations
    incidence_asc = grid_with_asc_desc["data_asc"]["incidence_angle"]
    azimuth_asc = grid_with_asc_desc["data_asc"]['azimuth_angle']
    grid_asc = grid_with_asc_desc["data_asc"].drop(columns=["incidence_angle", "azimuth_angle", "geometry_x"])

    # Gets the index of each polygon of the data_asc and casts them into ints
    keys_grid_asc=list(grid_with_asc_desc["data_asc"].index)
    keys_grid_asc=[int(x) for x in keys_grid_asc]

    # Gets the grid_asc from the dictionary and drops the geometry_x (Multipolygon)
    # grid_asc = grid_with_asc_desc["data_asc"] ; grid_asc = grid_asc.drop(columns=["geometry_x"])


    # Created a dictionary with the index as keys
    grid_asc=grid_asc.to_dict(orient="index")

    for e in keys_grid_asc:
        grid_asc[e] = pd.DataFrame.from_dict(grid_asc[e], orient='index')
        grid_asc[e].index = pd.to_datetime(grid_asc[e].index)

        # Convert start_date and end_date to pandas datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Filter using boolean mask
        mask = (grid_asc[e].index >= start) & (grid_asc[e].index <= end)
        grid_asc[e] = grid_asc[e][mask]
    
    # Does the same form desc
    incidence_desc = grid_with_asc_desc["data_desc"]["incidence_angle"]
    azimuth_desc = grid_with_asc_desc["data_desc"]['azimuth_angle']
    grid_desc = grid_with_asc_desc["data_desc"].drop(columns=["incidence_angle", "azimuth_angle", "geometry_x"])

    keys_grid_desc=list(grid_with_asc_desc["data_desc"].index)
    keys_grid_desc=[int(x) for x in keys_grid_desc]

    grid_desc=grid_desc.to_dict(orient="index")
    
    for h in keys_grid_desc:
        grid_desc[h]=pd.DataFrame.from_dict(grid_desc[h], orient='index')
        grid_desc[h].index = pd.to_datetime(grid_desc[h].index)

        # Convert start_date and end_date to pandas datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Filter using boolean mask
        mask = (grid_desc[h].index >= start) & (grid_desc[h].index <= end)
        grid_desc[h] = grid_desc[h][mask]

    # Final grid with normalized dates
    final_grid = {}
    for keys in keys_grid_asc:
        # If the key exists in both datasets
        if keys in keys_grid_desc:
            # Does an "outer" join of the same grid. Keeps every date
            final_grid[keys]=pd.merge(grid_asc[keys], grid_desc[keys], left_index=True, right_index=True, how="outer")

            # Sets the first value (displacement) in 0 for every displacement
            final_grid[keys].iloc[0]=0.0
        
    for key in final_grid:
        # Sets first displacement to 0 for each key (Again?)
        final_grid[key].iloc[0]=0.0

        # Renames the label of displacements
        final_grid[key].columns=["Asc_TS", "Desc_TS"]
        final_grid[key]['incidence_angle_asc'] = incidence_asc.loc[key]
        final_grid[key]['azimuth_angle_asc'] = azimuth_asc.loc[key]
        final_grid[key]['incidence_angle_desc'] = incidence_desc.loc[key]
        final_grid[key]['azimuth_angle_desc'] = azimuth_desc.loc[key]
    
    # Gets the first key (index) of the final grid
    first_key_of_final_grid=list(final_grid.keys())[0]

    # Get the indexes of the first key (This indexes are dates)
    oidx = final_grid[first_key_of_final_grid].index

    # Casts this indexes into dates
    oidx=pd.to_datetime(oidx)

    # Latest date - Oldest date / Number of total dates
    # This is calculated to get a deltatime to interpolate
    # Rounds the frecuency to day
    frequency=Timedelta((oidx.max() - oidx.min())/ int(len(oidx))); frequency=Timedelta.round(frequency, "D")

    # Generates a new datelist from min date to max date each "frecuency". Like np.arange but with dates
    nidx = pd.date_range(oidx.min(), oidx.max(), freq=frequency)
    
    resampled_grid = {}
    for key in final_grid:
        """
        Does the next for each key:
        1. Reindexes each grid for index by making a union for old-indexes (Original dates) and new-indexes (Frecuency calculated)
        2. Linear interpolation where the dates had NaN (For new-indexes)
        3. Removes old indexes and keeps just the interpolated ones
        """
        # Separate time-series data from constant metadata
        ts_data = final_grid[key][['Asc_TS', 'Desc_TS']]
        angles = final_grid[key][['incidence_angle_asc', 'azimuth_angle_asc', 
                                   'incidence_angle_desc', 'azimuth_angle_desc']].iloc[0]
        
        # Resample only the time-series data
        resampled_ts = ts_data.reindex(oidx.union(nidx)).interpolate(method="linear").reindex(nidx)
        
        # Add back the constant angles to all rows
        resampled_grid[key] = resampled_ts
        for col in angles.index:
            resampled_grid[key][col] = angles[col]
    
    # Returns new grid and new dates
    return resampled_grid, nidx

# fuction for reprojection of Asc and Desc displacements along Vertical and Horizontal components
# input: resampled grid, Asc LOS angle (radians), Desc LOS angle (radians)
def reprojection(resampled_grid, horz_az_angle=-90):
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
    
    for key, value in resampled_grid.items():
        # Extract angles for this grid cell (they're constant across all dates)
        los_inc_angle_asc = resampled_grid[key]['incidence_angle_asc'].iloc[0]
        los_az_angle_asc = resampled_grid[key]['azimuth_angle_asc'].iloc[0]
        los_inc_angle_desc = resampled_grid[key]['incidence_angle_desc'].iloc[0]
        los_az_angle_desc = resampled_grid[key]['azimuth_angle_desc'].iloc[0]
        
        # Build the design matrix G for this grid cell
        # G has shape (2, 2) for [asc, desc] x [horizontal, vertical]
        los_inc_angle = np.array([los_inc_angle_asc, los_inc_angle_desc])
        los_az_angle = np.array([los_az_angle_asc, los_az_angle_desc])
        
        # Design matrix from MintPy
        # G[i, 0] = horizontal component = sin(inc) * cos(az - horz_az)
        # G[i, 1] = vertical component = cos(inc)
        G = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            G[i, 0] = np.sin(np.deg2rad(los_inc_angle[i])) * np.cos(np.deg2rad(los_az_angle[i] - horz_az_angle))
            G[i, 1] = np.cos(np.deg2rad(los_inc_angle[i]))
        
        # Prepare the LOS displacement vector [Asc, Desc]
        dlos = np.array([
            resampled_grid[key]["Asc_TS"].values,
            resampled_grid[key]["Desc_TS"].values
        ])
        
        # Solve for [horizontal, vertical] using pseudo-inverse
        # dhv = pinv(G) * dlos
        dhv = np.dot(np.linalg.pinv(G), dlos)
        
        print(dhv)
        import time
        time.sleep(10)
        # Assign results
        resampled_grid[key]["Vh_TS"] = dhv[0, :]  # Horizontal component
        resampled_grid[key]["Vv_TS"] = dhv[1, :]  # Vertical component
        print(resampled_grid[key])
        exit()
    
    return print("reprojection successful")
