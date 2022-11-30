import geoplot
from shapely.ops import unary_union
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import alphashape
import geopandas as gpd
import geopandas
import libpysal
from spopt.region import RegionKMeansHeuristic
import shapely
import pandas as pd

def save_to_geojson(df, fn):
    if os.path.exists(fn):
        # os.unlink(fn)
        pass
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(dict(geometry=df))
    df = df.reset_index()
    schema = {**gpd.io.file.infer_schema(df), 'geometry': 'Unknown'}
    df.to_file(fn, driver='GeoJSON', schema=schema)

def mini_shapes(lat_all, lon_all, coords=False):
    shapes = []
    shape_coords_list = []
    sz_lat = max(lat_all[1:] - lat_all[0:-1])
    sz_lon = max(lon_all[1:] - lon_all[0:-1])
    # print(sz_lat)
    # print(sz_lon)
    alpha = 0.5

    for lat, lon in zip(lat_all, lon_all):
        shape_coords = [(round(lon + incr_lon * sz_lon / 2, 5), round(lat + incr_lat * sz_lat / 2, 5)) for incr_lat in
                        (0, 1, -1) for incr_lon in (0, 1, -1)
                        if not (incr_lon == incr_lat == 0)]
        if not (coords):
            shapes.append(alphashape.alphashape(shape_coords, alpha=alpha))

        shape_coords_list.append(shape_coords)

    if coords:
        return shape_coords_list

    return shapes

def get_rescaled_lats_and_lons(lats,lons,sel_lat,sel_lon,new_sz,attr_wc):

    def check_gran_larger_than_size(lats_,lons_,new_sz):
        sz_lat = round(max(lats_[1:] - lats_[0:-1]), 5)
        sz_lon = round(max(lons_[1:] - lons_[0:-1]), 5)
        assert round(sz_lat, 5) == round(sz_lon, 5) <= new_sz



    if len(attr_wc) ==1:
        lats_ = np.round(lats[attr_wc][sel_lat[attr_wc]], 5)
        lons_ = np.round(lons[attr_wc][sel_lon[attr_wc]], 5)
        check_gran_larger_than_size(lats_,lons_,new_sz)
        new_lats = [l for l in np.arange(min(lats_) + new_sz / 2, max(lats_) - new_sz / 2, new_sz)]
        new_lons = [l for l in np.arange(min(lons_) + new_sz / 2, max(lons_) - new_sz / 2, new_sz)]

    elif len(attr_wc)==2:
        min_lat,max_lat,min_lon,max_lon = -180,180,-180,180
        lats_,lons_ = dict(),dict()
        for at in attr_wc:
            lats_[at] = np.round(lats[at][sel_lat[at]], 5)
            lons_[at] = np.round(lons[at][sel_lon[at]], 5)

            min_lat = max(min_lat,min(lats_[at]))
            max_lat = min(max_lat, max(lats_[at]))

            min_lon = max(min_lon,min(lons_[at]))
            max_lon = min(max_lon, max(lons_[at]))

            check_gran_larger_than_size(lats_[at],lons_[at],new_sz)

        new_lats = [l for l in np.arange(min_lat + new_sz / 2, max_lat - new_sz / 2, new_sz)]
        new_lons = [l for l in np.arange(min_lon + new_sz / 2, max_lon - new_sz / 2, new_sz)]



    return new_lats,new_lons

def change_data_granularity(attr,lats,lons, new_lats, new_lons, new_sz):

    pointsused = 0

    assert attr.shape == (len(lats), len(lons))



    attr_new = np.zeros((len(new_lats), len(new_lons)))

    for new_lat in new_lats:
        # Let's start by getting all the indices of old latitudes that that are now linked to this new lat.
        # Assumption of symmetry around new lat
        lat_range = (new_lat - new_sz / 2, new_lat + new_sz / 2)

        indices_in_a = np.where(np.logical_and(lats >= lat_range[0], lats < lat_range[1]))[0]
        # indices_border_a = np.where(np.logical_and(lats==lat_range[0], lats==lat_range[1]))

        for new_lon in new_lons:
            lon_range = (new_lon - new_sz / 2, new_lon + new_sz / 2)

            indices_in_o = np.where(np.logical_and(lons >= lon_range[0], lons < lon_range[1]))[0]
            # indices_border_o = np.where(np.logical_and(lons==lon_range[0], lons==lon_range[1]))

            list_points_in = [attr[x, y] for x in indices_in_a for y in indices_in_o]
            new_value = np.mean(list_points_in)
            pointsused += len(list_points_in)
            attr_new[np.where(new_lats == new_lat)[0], np.where(new_lons == new_lon)[0]] = new_value

    return attr_new, pointsused

# 
# def change_granularity(attr, lats, lons, new_sz):
#     ##To avoid numerical problems, let's first round the original coordinates
#     lats = np.round(lats, 5)
#     lons = np.round(lons, 5)
# 
#     pointsused = 0
#     sz_lat = round(max(lats[1:] - lats[0:-1]), 5)
#     sz_lon = round(max(lons[1:] - lons[0:-1]), 5)
# 
#     assert round(sz_lat, 5) == round(sz_lon, 5) <= new_sz
# 
#     assert attr.shape == (len(lats), len(lons))
# 
#     new_lats = [l for l in np.arange(min(lats) + new_sz / 2, max(lats) + new_sz / 2, new_sz)]
#     new_lons = [l for l in np.arange(min(lons) + new_sz / 2, max(lons) + new_sz / 2, new_sz)]
# 
#     attr_new = np.zeros((len(new_lats), len(new_lons)))
# 
#     for new_lat in new_lats:
#         # Let's start by getting all the indices of old latitudes that that are now linked to this new lat.
#         # Assumption of symmetry around new lat
#         lat_range = (new_lat - new_sz / 2, new_lat + new_sz / 2)
# 
#         indices_in_a = np.where(np.logical_and(lats >= lat_range[0], lats < lat_range[1]))[0]
#         # indices_border_a = np.where(np.logical_and(lats==lat_range[0], lats==lat_range[1]))
# 
#         for new_lon in new_lons:
#             lon_range = (new_lon - new_sz / 2, new_lon + new_sz / 2)
# 
#             indices_in_o = np.where(np.logical_and(lons >= lon_range[0], lons < lon_range[1]))[0]
#             # indices_border_o = np.where(np.logical_and(lons==lon_range[0], lons==lon_range[1]))
# 
#             list_points_in = [attr[x, y] for x in indices_in_a for y in indices_in_o]
#             new_value = np.mean(list_points_in)
#             pointsused += len(list_points_in)
#             attr_new[np.where(new_lats == new_lat)[0], np.where(new_lons == new_lon)[0]] = new_value
# 
#     return new_lons, new_lats, attr_new, pointsused

def get_lons_lats_from_ds(ds):
    lat = dict()
    lon = dict()
    start = dict()
    end = dict()
    for attr in ds:
        lat[attr] = ds[attr]['lat']
        lon[attr] = ds[attr]['lon']

        start[attr] = 1
        end[attr] = len(lat[attr]) * len(lon[attr])
    return lat, lon

def intersect_lats_lons(lat, lon):
    b_lat = min(max(lat['s'][:]), max(lat['w'][:]),np.float64(65.))
    b_lon = min(max(lon['s'][:]), max(lon['w'][:]),np.float64(35.))

    # sel_lat = lat[attr_1] <= b_lat
    # sel_lon = lon[attr_2] <= b_lon

    sel_lat = dict()
    sel_lon = dict()

    sel_lat["s"] = lat["s"] <= b_lat
    sel_lon["s"] = lon["s"] <= b_lon

    sel_lat["w"] = lat["w"] <= b_lat
    sel_lon["w"] = lon["w"] <= b_lon


    return sel_lat,sel_lon

def get_selection_list(lon_all_gran,lat_all_gran,mainland_only= False):
    if mainland_only:
        europe_land = gpd.read_file("../resources/regions_onshore.geojson")
        eu_land = shapely.ops.unary_union(europe_land["geometry"])
        all_points_gran = [shapely.geometry.Point((lon, lat)) for lon, lat in zip(lon_all_gran, lat_all_gran)]
        selection = [np.any([point.within(geom) for geom in eu_land]) for point in all_points_gran]
    else:
        selection = list(np.repeat(True,len(lat_all_gran)))

    return selection

def generate_regional_kmeans_model(lon_all_gran, lat_all_gran, attributes,nb_regions,mainland_only =False):


    selection = get_selection_list(lon_all_gran,lat_all_gran,mainland_only)

    data = np.zeros(shape=(len(lon_all_gran), len(attributes)))[selection]

    for att_index in range(len(attributes)):
        data[:, att_index] = attributes[att_index].flatten('F')[selection]

    data /= np.max(np.abs(data), axis=0)

    gdf = create_geodataframe_mini_shapes(lat_all_gran,lon_all_gran,data,selection)
    # gdf.plot(column = "values_1")
    # plt.show()
    w = libpysal.weights.Rook.from_dataframe(gdf)
    model = RegionKMeansHeuristic(data, nb_regions, w)
    return model,data,selection

def generate_regional_kmeans_model_locs(lon_all_gran, lat_all_gran, attributes,weight,nb_regions,mainland_only =False):


    selection = get_selection_list(lon_all_gran,lat_all_gran,mainland_only)

    data = np.zeros(shape=(len(lon_all_gran), len(attributes)))[selection]

    for att_index in range(len(attributes)):
        data[:, att_index] = attributes[att_index].flatten('F')[selection]

    data /= np.max(np.abs(data), axis=0)
    if weight == 0:
        data = data[:,:-2]
    else:
        data[:,-2:] *= weight

    gdf = create_geodataframe_mini_shapes(lat_all_gran,lon_all_gran,data,selection)

    w = libpysal.weights.Rook.from_dataframe(gdf)
    model = RegionKMeansHeuristic(data, nb_regions, w)
    return model,data,selection

def create_geodataframe_mini_shapes(lat_all_gran,lon_all_gran,data,selection):
    mini_shapes_gran = np.array(mini_shapes(lat_all_gran, lon_all_gran))
    mini_shapes_gran = mini_shapes_gran[selection]

    columns = [f"values_{i}" for i in range(data.shape[1])]
    df_data = pd.DataFrame(columns=columns,data=data)
    gdf = gpd.GeoDataFrame(geometry=mini_shapes_gran,data=df_data)

    return gdf

def plot_regions_scatter_and_save(lat_all, lon_all, ds,attributes,nb_regions,seed):
    fig, axs = plt.subplots(1, len(attributes) + 1 , figsize=(12, 4))

    colors = [model.centroids_[label, 0] for label in model.labels_]
    axs[0].scatter(lon_all_gran, lat_all_gran, c=colors)
    axs[0].set_title(f'K-means clusters {attributes}')

    for attr_idx in range(len(attributes)):

        axs[attr_idx+1].scatter(lon_all[attributes[attr_idx]], lat_all[attributes[attr_idx]],
                        c=(np.mean(ds[attributes[attr_idx]][name[attributes[attr_idx]]][:, :, :].data, axis=0).flatten('F')))
        axs[attr_idx+1].set_title(f'Input data mean {attributes[attr_idx]}')


    fig.suptitle(f'Clustering {attributes[attr_idx]} {nb_regions}')
    dir_path = os.path.join(snakemake.output.renewable_shapes.partition("resources")[0], "results", "plots")
    path = os.path.join(dir_path, f"renewable_clusters_scatter_{attributes}_{nb_regions}_seed{seed}")
    print(path)

    isExist = os.path.exists(dir_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
    plt.savefig(path)

def plot_regions_scatter_gran_and_save(lat_all_gran, lon_all_gran, model,labels,centroids,seed,snakemake_output,gran,attr_wc):
    data = model.data
    fig, axs = plt.subplots(1,data.shape[1] + 1 , figsize=(12, 4))

    colors = [centroids[label, 0] for label in labels]
    axs[0].scatter(lon_all_gran, lat_all_gran, c=colors)
    axs[0].set_title(f'K-means clusters')

    for attr_idx in range(data.shape[1]):

        axs[attr_idx+1].scatter(lon_all_gran, lat_all_gran,
                        c=(data[:,attr_idx]))
        axs[attr_idx+1].set_title(f'Input data mean {attr_idx} ' )


    #fig.suptitle(f'Clustering {attributes[attr_idx]} {nb_regions}')
    rdir = snakemake_output.renewable_shapes.partition("/")[2]
    dir_path = os.path.join(snakemake_output.renewable_shapes.partition("resources")[0], "results", "plots",rdir)
    path = os.path.join(dir_path, f"renewable_clusters_scatter_{attr_wc}_{gran}.png")
    print(path)

    isExist = os.path.exists(dir_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
    plt.savefig(path)

def plot_selection(data_selection,selection):
    fig, axs = plt.subplots(1, data_selection.shape[1])

    for attr_idx in range(data_selection.shape[1]):
        if data_selection.shape[1] == 1:
            axs.scatter(lon_all_gran[selection], lat_all_gran[selection], c=data_selection[:, attr_idx])
        else:
            axs[attr_idx].scatter(lon_all_gran[selection], lat_all_gran[selection], c=data_selection[:, attr_idx])

    dir_path = os.path.join(snakemake.output.renewable_shapes.partition("resources")[0], "results", "plots","clustering_selection")
    path = os.path.join(dir_path, f"renewable_clusters_scatter_{attr_wc}_{mainland_only}")
    print(path)

    isExist = os.path.exists(dir_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
    plt.savefig(path)

def create_geodataframe_with_shapes_from_k_means_model(lat_all_gran,lon_all_gran,model):
    clusters = geopandas.GeoDataFrame()

    mini_shapes_attr_gran = np.array(mini_shapes(lat_all_gran, lon_all_gran))

    for cluster in np.unique(model.labels_):
        idx = (model.labels_ == cluster)
        polygons = mini_shapes_attr_gran[idx]
        cluster_shape = shapely.ops.unary_union(polygons)


        frame_row = geopandas.GeoDataFrame(
            {"name": [f"Ren_{cluster}"], "geometry": cluster_shape, "Cluster": cluster})
        i=0
        for center in model.centroids_[cluster]:
            i+=1
            frame_row[f"center{i}"]= center

        #clusters = clusters.append(frame_row)
        clusters = pd.concat([clusters,frame_row],axis=0,ignore_index=True)
    return clusters

def load_data_sets(input):
    ds = dict()
    ds["s"] = nc.Dataset(input.cutout_solar)
    ds["w"] = nc.Dataset(input.cutout_wind)
    return ds

def get_number_of_buddies(point, model):
    cluster = model.labels_[point]
    # buddies = []
    nb_buddies = 0
    for neighbor in model.w.neighbors[point]:
        # buddies.append[neighbor]
        if model.labels_[neighbor] == cluster:
            nb_buddies += 1
    return nb_buddies

def get_nb_buddies_list(model):
    nb_buddy_list = []
    for point in range(len(model.data)):
        nb_buddy_list.append(get_number_of_buddies(point,model))
    return np.array(nb_buddy_list)

