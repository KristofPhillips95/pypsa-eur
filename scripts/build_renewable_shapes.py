import geoplot
from shapely.ops import unary_union
import os
from _helpers import configure_logging
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import alphashape
import geopandas as gpd
import libpysal
from spopt.region import RegionKMeansHeuristic
import shapely
import pandas as pd


def save_to_geojson(df, fn):
    if os.path.exists(fn):
        os.unlink(fn)
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
    print(sz_lat)
    print(sz_lon)
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

def change_granularity(attr, lats, lons, new_sz):
    ##To avoid numerical problems, let's first round the original coordinates
    lats = np.round(lats, 5)
    lons = np.round(lons, 5)

    pointsused = 0
    sz_lat = round(max(lats[1:] - lats[0:-1]), 5)
    sz_lon = round(max(lons[1:] - lons[0:-1]), 5)

    assert round(sz_lat, 5) == round(sz_lon, 5) <= new_sz

    assert attr.shape == (len(lats), len(lons))

    new_lats = [l for l in np.arange(min(lats) + new_sz / 2, max(lats) + new_sz / 2, new_sz)]
    new_lons = [l for l in np.arange(min(lons) + new_sz / 2, max(lons) + new_sz / 2, new_sz)]

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

    return new_lons, new_lats, attr_new, pointsused


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


##TODO this implementation leans on the assumption that attr 1 has the highest lat range, and attr 2 has highest lon range
def intersect_lats_lons(lat, lon):
    b_lat = min(max(lat['s'][:]), max(lat['w'][:]),np.float64(65.))
    b_lon = min(max(lon['s'][:]), max(lon['w'][:]),np.float64(35.))

    #b_lat = np.float64(65.)
    #b_lon = np.float64(35.)


    sel_lat = lat[attr_1] <= b_lat
    sel_lon = lon[attr_2] <= b_lon

    sel_lat = dict()
    sel_lon = dict()

    sel_lat["s"] = lat["s"] <= b_lat
    sel_lon["s"] = lon["s"] <= b_lon

    sel_lat["w"] = lat["w"] <= b_lat
    sel_lon["w"] = lon["w"] <= b_lon


    return sel_lat,sel_lon


def generate_regional_kmeans_model(l1, l2, attributes,nb_regions):


    w_gran = libpysal.weights.lat2W(len(l2), len(l1))

    data = np.zeros(shape=(len(lon_all_gran), len(attributes)))

    for att_index in range(len(attributes)):
        data[:, att_index] = attributes[att_index].flatten('F')

    data /= np.max(np.abs(data), axis=0)

    model = RegionKMeansHeuristic(data, nb_regions, w_gran)
    return model


def plot_regions_scatter_and_save(lat_all, lon_all, ds,attributes,nb_regions):
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
    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    path = os.path.join(snakemake.output.renewable_shapes.partition("resources")[0], "results", "plots" , f"renewable_clusters_scatter_{attributes}_{nb_regions}")
    print(path)
    plt.savefig(path)


def create_geodataframe_with_shapes_from_k_means_model(lat_all_gran,lon_all_gran,model):
    clusters = gpd.GeoDataFrame()

    mini_shapes_attr_gran = np.array(mini_shapes(lat_all_gran, lon_all_gran))

    for cluster in np.unique(model.labels_):
        idx = (model.labels_ == cluster)
        polygons = mini_shapes_attr_gran[idx]
        cluster_shape = shapely.ops.unary_union(polygons)

        frame_row = gpd.GeoDataFrame(
            {"name": [f"Ren_{cluster}"], "geometry": cluster_shape})
        i=0
        for center in model.centroids_[cluster]:
            i+=1
            frame_row[f"center{i}"]= center

        #clusters = clusters.append(frame_row)
        clusters = pd.concat([clusters, frame_row], axis=0, ignore_index=True)
    return clusters


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_renewable_shapes',regions = 's-20')
    configure_logging(snakemake)
    save_fig = snakemake.config["renewable_zones"]["save_fig"]

    #Load datasets of era and sara into dict
    ds = dict()
    ds["s"] = nc.Dataset(snakemake.input.cutout_solar)
    ds["w"] = nc.Dataset(snakemake.input.cutout_wind)

    #Extract main parameters of clustering method from config file and wildcard
    gran = snakemake.config["renewable_zones"]["granularity"]
    attr_wc = snakemake.wildcards.regions.partition("-")[0]
    print(attr_wc)
    print(snakemake.wildcards.regions.partition("-")[2])

    if snakemake.wildcards.regions == "c":
        nb_regions = 1
        clusters = gpd.read_file(snakemake.input.country_shapes)
    else:
        nb_regions = int(snakemake.wildcards.regions.partition("-")[2])

        #Name of the relevant data in datasets
        name = {'w': 'wnd100m', 's': 'influx_direct'}
        attr_1 = 'w'
        attr_2 = 's'

        #Get latitudes and longitudes from the dataset
        lat,lon = get_lons_lats_from_ds(ds)
        #And select the ones that occur in both datasets
        sel_lat, sel_lon = intersect_lats_lons(lat,lon)

        l1,l2 = get_rescaled_lats_and_lons(lat,lon,sel_lat,sel_lon,gran,attr_wc)

        #Change the granularity to the desired resolution. This is done for computational feasibility
        # l1, l2, at_1, pu = change_granularity(np.mean(ds[attr_1][name[attr_1]][:, :, :].data[:,sel_lat[attr_1],:][:,:,sel_lon[attr_1]],
        #                                               axis=0), lat[attr_1][sel_lat[attr_1]], lon[attr_1][sel_lon[attr_1]], gran)
        # l1, l2, at_2, pu = change_granularity(np.mean(ds[attr_2][name[attr_2]][:, :, :].data[:, sel_lat[attr_2],:][:,:,sel_lon[attr_2]],
        #                                               axis=0), lat[attr_2][sel_lat[attr_2]], lon[attr_2][sel_lon[attr_2]], gran)
        at_1, pu = change_data_granularity(
            np.mean(ds[attr_1][name[attr_1]][:, :, :].data[:, sel_lat[attr_1], :][:, :, sel_lon[attr_1]],
                    axis=0), lat[attr_1][sel_lat[attr_1]], lon[attr_1][sel_lon[attr_1]], l1, l2, gran)
        at_2, pu = change_data_granularity(
            np.mean(ds[attr_2][name[attr_2]][:, :, :].data[:, sel_lat[attr_2], :][:, :, sel_lon[attr_2]],
                    axis=0), lat[attr_2][sel_lat[attr_2]], lon[attr_2][sel_lon[attr_2]], l1,l2, gran)

        #Vector of repeated lats and lons
        lat_all_gran = np.tile(l1, len(l2))
        lon_all_gran = np.repeat(l2, len(l1))

        #Create the actual regional k-means model to be solved
        if attr_wc == 'ws':
            model = generate_regional_kmeans_model(l1,l2,[at_1, at_2],nb_regions)
        elif attr_wc  == 'w':
            model = generate_regional_kmeans_model(l1,l2,[at_1],nb_regions)
        elif attr_wc == 's':
            model = generate_regional_kmeans_model(l1,l2,[at_2],nb_regions)

        #And solve it. Seed is given for reproducibility
        np.random.seed(snakemake.config["renewable_zones"]["seed"])
        model.solve()

        #Generate shapes from the datapoints' locations and cluster labels from the model solve
        clusters = create_geodataframe_with_shapes_from_k_means_model(lat_all_gran,lon_all_gran,model)

        if save_fig:
            lat_all = dict()
            lon_all = dict()
            for attr in ds:
                lat_all[attr] = np.tile(lat[attr], len(lon[attr]))
                lon_all[attr] = np.repeat(lon[attr], len(lat[attr]))

            plot_regions_scatter_and_save(lat_all, lon_all, ds, attr_wc, nb_regions)

    save_to_geojson(clusters,snakemake.output.renewable_shapes)

