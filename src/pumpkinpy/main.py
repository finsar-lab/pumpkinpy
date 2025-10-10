#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Original implementation:
https://github.com/maybedave/InSAR-Time-Series-Clustering

Based on:
Festa, D., Novellino, A., Hussain, E., Bateson, L., Casagli, N., Confuorto, P., Del Soldato, M., Raspini, F.: Unsupervised detection of InSAR ground motion patterns based on PCA and K-means analysis, International Journal of Applied Earth Observation and Geoinformation, Volume 118, 2023, 103276, ISSN 1569-8432, https://doi.org/10.1016/j.jag.2023.103276.

TS-InSAR resampling and reprojection of Asc and Desc data into Vertical and Horizontal gridded data
Execute a cluster analysis of TS-InSAR data by using PCA and K-Means
Decompose clustered time series through Linear Regression and Fast Fourier Transform

@author: Davide Festa (davide.festa@unifi.it)
***Version 10/11/2022

"""
import tomllib
from pathlib import Path

from kneed import KneeLocator
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
# import subfunctions 
import reprojection as V_H_reprojection
import clustering as KMeans_clustering
import pca as PCAnalysis
import regression as regression_DFT
from joblib import Parallel, delayed

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# Load initial data:
with open("/home/doki/code/pumpkinpy/config.toml", "rb") as f:
    config = tomllib.load(f)

path_to_InSAR_data = Path(config["paths"]["data_dir"])
data_dir = Path(config["paths"]["data_dir"])
outpath = Path(config["paths"]["output_dir"]) # 3) path to output folder

# Read CSV of the ascending geometry time series Insar dataset containing the "X" "Y" field projected coordinates
#ASC = pd.read_csv(path_to_InSAR_data / 'ascending_InSAR_sample.csv', sep=',')

# Read CSV of the descending geometry time series Insar dataset containing the "X" "Y" field projected coordinates 
#DESC = pd.read_csv(path_to_InSAR_data / 'descending_InSAR_sample.csv', sep=',')

ASC = pd.read_parquet(path_to_InSAR_data / "asc_dataset.parquet")
DESC = pd.read_parquet(path_to_InSAR_data / "desc_dataset.parquet")

datasets_names = ["data_asc", "data_desc"]
# unique dictionary containing both ascending and descending dataframes
Dataset_asc_desc_dict = dict(zip(datasets_names, [ASC, DESC]))

# Setting up dictionaries containing Insar-ts-related info, where each one containing both ascending and descending datasets attributes
data_coordinates = {}
ts = {}
date_ts = {}
date_formatted = {}
df_coords = {}
lats = {}
lons = {}
df_array = {}
angles_data = {}

# Requirements:
# The time series-related columns starts with "D" and no other columns should start with "D"

for key in Dataset_asc_desc_dict:
    data_coordinates[key] = pd.DataFrame(
        Dataset_asc_desc_dict[key], columns=['X', 'Y'])
    ts[key] = Dataset_asc_desc_dict[key].loc[:, Dataset_asc_desc_dict[key].columns.str.startswith(
        'D')]
    
    # Extract the angle columns
    angle_columns = []
    angle_columns.append('incidence_angle')
    angle_columns.append('azimuth_angle')
    
    # Extract angles data
    angles_data[key] = Dataset_asc_desc_dict[key][angle_columns]

    date_ts[key] = list(ts[key].columns.values)
    date_ts[key] = [e[1:] for e in date_ts[key]]
    date_formatted[key] = np.array([dt.datetime.strptime(
        d, '%Y%m%d').date() for d in date_ts[key]])  # line for formatting date names
    date_formatted[key] = list(date_formatted[key])
    ts[key].columns = date_formatted[key]
    # dict with final asc/desc arrays containing X Y coordinates and time series and angles
    df_coords[key] = pd.concat([data_coordinates[key], angles_data[key], ts[key]], axis=1)
    #df_coords[key] = pd.concat([data_coordinates[key], ts[key]], axis=1)

    lats[key] = pd.DataFrame(df_coords[key], columns=['X'])
    lons[key] = pd.DataFrame(df_coords[key], columns=['Y'])
    # dict with final asc/desc arrays containing only time series displacement values
    df_array[key] = df_coords[key].iloc[:, 4:].to_numpy()

# Retrieving lats and lons from both ascending and descending datataset to set up the extension of the fishnet
AscDesc_lats = pd.concat([lats["data_asc"], lats["data_desc"]])
AscDesc_lons = pd.concat([lons["data_asc"], lons["data_desc"]])
Tot_points = pd.concat([AscDesc_lats, AscDesc_lons], axis=1)

# -------------------------- ASSESSMENT OF OPTIMAL GRID SIZE ------------------
# Evaluation of the best trade-off size by finding the smallest grid size able to capture ascending and descending data
# The size is evaluated by plotting the results of totcells/commoncells (totcells is the total number of cells produced at the n size grid and commoncells is the number of cell containing asc and desc data) against the n range of size values
# The knee point of the plotted curve is the best grid size 

# Cellsize input list
print("Creating cellsize")
cellsize=list(np.arange(15, 216, 10))
totcells=[]
totcells_to_commoncells=[]

def process_cellsize(e, Tot_points, df_coords):
    """Process a single cellsize value"""
    cell = V_H_reprojection.gridding(Tot_points, e)
    grid_with_asc_desc = V_H_reprojection.grid_dissolve(
        cell, df_coords["data_asc"], df_coords["data_desc"])
    
    common_indexes = grid_with_asc_desc["data_asc"].index.intersection(
        grid_with_asc_desc["data_desc"].index)
    
    totcell = int(cell.count().iloc[0])
    ratio = totcell / len(common_indexes)
    return totcell, ratio


results = Parallel(n_jobs=16, verbose=10)(
    delayed(process_cellsize)(e, Tot_points, df_coords) 
    for e in cellsize
)

# Unpack results
totcells, totcells_to_commoncells = zip(*results)
totcells = list(totcells)
totcells_to_commoncells = list(totcells_to_commoncells)

print("Creating dataframe")
for_grid_size_asses=pd.DataFrame(
    {'cellsize': cellsize,
     'totcells': totcells,
     'totcells_to_commoncells': totcells_to_commoncells
    })

print("Doing knee")
gridsize=KneeLocator(for_grid_size_asses["cellsize"], for_grid_size_asses["totcells_to_commoncells"],curve="convex", direction="decreasing")
gridsize.plot_knee()
#plt.savefig("knee_plot.png")
best_grid_size=gridsize.knee
print(best_grid_size)

# TO MANUALLY OVERRIDE THE AUTOMATED GRID SIZE SELECTION:
# best_grid_size=

#------------------------------------------------------------------------------
 
# Makes the optimal size Grid
cell=V_H_reprojection.gridding(Tot_points, best_grid_size)

# # Checking gridding results by plotting
#cell.plot(facecolor="none", edgecolor='grey')
#plt.scatter(Tot_points["X"], Tot_points["Y"], s=1)
#plt.show()
#plt.savefig("gridding_results.png")

print("Plotting")

# Returns a dictionary with both dissolved datasets (The grid with the mean displacement values)
grid_with_asc_desc = V_H_reprojection.grid_dissolve(
    cell, df_coords["data_asc"], df_coords["data_desc"])

# Returns a dictionary with merged asc, desc and the interpolated dates as indexes (Also returns interpolated dates)
resampled_grid, new_dates = V_H_reprojection.resampling_ts(
    date_formatted["data_asc"], date_formatted["data_desc"], grid_with_asc_desc)

# checking for Asc and Desc newly interpolated time series
print("Before Plotting")
#resampled_grid[int(list(resampled_grid.keys())[0])].plot(style='.-')

# fuction for reprojection of Asc and Desc displacements along Vertical and Horizontal components
# input: resampled grid, Asc LOS angle (radians), Desc LOS angle (radians), WHICH NEED TO BE ADJUSTED TO THE ACQUISITION GEOMETRY OF THE ANALYSED A-DINSAR DATASET
V_H_reprojection.reprojection(resampled_grid)

# Extraction of Vertical and Horizontal displacement time series for PCA and clustering analysis
print("Extracting horizontal and vertical")
Vv_df_array = []
Vh_df_array = []

for key in resampled_grid:
    Vv_df_array.append(resampled_grid[key]
                       ["Vv_TS"].to_numpy().reshape((1, -1)))
    Vh_df_array.append(resampled_grid[key]
                       ["Vh_TS"].to_numpy().reshape((1, -1)))

Vv_df_array = np.concatenate(Vv_df_array, axis=0)
Vv_df_array[:, 0] = 0.0
Vh_df_array = np.concatenate(Vh_df_array, axis=0)
Vh_df_array[:, 0] = 0.0

# Taking out cells with empty data
index_cells_geometry = [x for x in list(
    cell.index) if x in list(resampled_grid.keys())]
geometry_cell = cell.iloc[index_cells_geometry]

# check the grid after deleting the empty cells
geometry_cell.plot(facecolor="none", edgecolor='grey')


# --------------------------------------PCA analysis---------------------------
# PCA to constrain the number of clusters to use with the unsupervised clustering,
# see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# The analysis is repeated for Horizontal time series and Vertical time series

# df_array_std should be close to 1, stdarray_mean should be close to 0, stdarray_std is the standardised arrray
print("PCA")
df_array_std, stdarray_mean, stdarray_std = PCAnalysis.std_scaler(
    Vh_df_array)  
df_array_std2, stdarray_mean2, stdarray_std2 = PCAnalysis.std_scaler(
    Vv_df_array)

# PCA_tot_variance is the matrix variance value computed, perc_variance is the same expressed in percentage
PCA_tot_variance, perc_variance = PCAnalysis.pca_algorithm(df_array_std)
PCA_tot_variance2, perc_variance2 = PCAnalysis.pca_algorithm(df_array_std2)

# # function asking computing no. of components to retain based on the total percentage of variance that we want to retain
#wanted_percentage = PCAnalysis.required_percentage()
#wanted_percentage2 = PCAnalysis.required_percentage()

# number of components retrieved by means of cut-off value in terms of variance percentage
#number_of_components = PCAnalysis.components_accounting_for_required_perc_variance(
   # perc_variance, wanted_percentage)  # no. of components resulting
#number_of_components2 = PCAnalysis.components_accounting_for_required_perc_variance(
   # perc_variance2, wanted_percentage2)  # no. of components resulting

# automatic method to find the optimal number of components based on the best trade-off value
# evaluated on the curve of cumulative percentage of variance, by using the KneeLocator function
kvh = KneeLocator(np.asarray([(range(len(perc_variance)))]).squeeze(), np.asarray(
    [perc_variance]).squeeze(), curve="convex", direction="decreasing")
kvv = KneeLocator(np.asarray([(range(len(perc_variance2)))]).squeeze(), np.asarray(
    [perc_variance2]).squeeze(), curve="convex", direction="decreasing")

kvh.plot_knee()
plt.savefig('kvh_plot.png', dpi=300, bbox_inches='tight')
kvv.plot_knee()
plt.savefig('kvv_plot.png', dpi=300, bbox_inches='tight')

number_of_components = kvh.knee
number_of_components2 = kvv.knee

# --------------------------------------Cluster analysis-----------------------
# for theory, see https://scikit-learn.org/stable/modules/clustering.html
# for parameters setting, https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html

print("Cluster")

km, cluster_center, cluster_center_shape, time_series_class, labels, count_labels, inertia = KMeans_clustering.Euclidean_KMeans(
    Vh_df_array, number_of_components)
km2, cluster_center2, cluster_center_shape2, time_series_class2, labels2, count_labels2, inertia2 = KMeans_clustering.Euclidean_KMeans(
    Vv_df_array, number_of_components2)

# plot the % of the clusters
KMeans_clustering.cluster_distribution_plotter(
    number_of_components, count_labels, labels)
#plt.savefig('/Clusters_%_distributionVh.png')
KMeans_clustering.cluster_distribution_plotter(
    number_of_components2, count_labels2, labels2)
#plt.savefig('/Clusters_%_distributionVv.png')

# Plotting the clusters centre
# for plotting, https://tslearn.readthedocs.io/en/stable/auto_examples/clustering/plot_kmeans.html#sphx-glr-auto-examples-clustering-plot-kmeans-py
# taking the 10th and 90th percentile to display along with the clusters centre

Vh_clusters_centroids = KMeans_clustering.cluster_center_plotter(
    Vh_df_array, number_of_components, time_series_class, cluster_center)
#plt.savefig('/Clusters_centersVh.png')
Vv_clusters_centroids = KMeans_clustering.cluster_center_plotter(
    Vv_df_array, number_of_components2, time_series_class2, cluster_center2)
#plt.savefig('/Clusters_centersVv.png')

labels = labels+1
labels2 = labels2+1
labels_dataframe = pd.DataFrame(labels, columns=['cluster'])
labels_dataframe2 = pd.DataFrame(labels2, columns=['cluster'])

df_coords_clusters = pd.concat(
    [geometry_cell.reset_index(drop=True), labels_dataframe.reset_index(drop=True), pd.DataFrame(index_cells_geometry, columns=["cell_index"])], axis=1)
df_coords_clusters2 = pd.concat(
    [geometry_cell.reset_index(drop=True), labels_dataframe2.reset_index(drop=True), pd.DataFrame(index_cells_geometry, columns=["cell_index"])], axis=1)

# export cluster location as Shapefile
df_coords_clusters.crs = '3857'
df_coords_clusters2.crs = '3857'

df_coords_clusters.to_file(
    outpath / 'cluster_horizontal_components.shp')
df_coords_clusters2.to_file(
    outpath / 'cluster_vertical_components.shp')

#----------------------------------------- TIME SERIES DECOMPOSE --------------
# for linear regression theory, see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# for Fast fourier Transform theory, see https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html

# Here the decomposition is applied to one of the resulting cluster centroid series
# To decompose the cluster centroid of interest you have to manually assign it in the following line:
Vh_clusters_centroid_1=pd.DataFrame(Vh_clusters_centroids[0,:], index=new_dates)

# function computing Linear regression and DFT for a time series
# input: dates (DatetimeIndex), dataframe containing cluster centroid discrete values
# output:  slope of the best-fit line, intercept of the best-fit line, rmse of the regression, dataframe containing the first 6 peaks of the spectral power with conversion of frequency to relative period
slope, intercept, rmse, powerspectrum = regression_DFT.LinRegression_DFT(new_dates, Vh_clusters_centroid_1)
