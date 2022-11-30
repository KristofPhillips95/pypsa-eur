import renewable_shapes_methods as rsm
import numpy as np
import geopandas as gpd
from _helpers import configure_logging
import os
from joblib import Parallel,delayed
import time


def solve_model_and_save_to_dict(seed, model, dist, gdfs, bl_dict):
    #print(seed)
    np.random.seed(seed)
    model.solve()
    dist[seed] = dict()
    for label in model.labels_:
        dist[seed][label] = ((model.data[model.labels_ == label] - model.centroids_[label]) ** 2)
    gdfs[seed] = rsm.create_geodataframe_with_shapes_from_k_means_model(lat_all_gran, lon_all_gran, model)
    bl_dict[seed] = rsm.get_nb_buddies_list(model)

def solve_model_and_return(seed, model):
    #print(seed)
    np.random.seed(seed)
    model.solve()
    dist = dict()
    for label in model.labels_:
        dist[label] = ((model.data[model.labels_ == label] - model.centroids_[label]) ** 2)
    gdf = rsm.create_geodataframe_with_shapes_from_k_means_model(lat_all_gran, lon_all_gran, model)
    bl = rsm.get_nb_buddies_list(model)
    return seed,dist,gdf,bl,model.labels_,model.centroids_

def unpack_and_assign_to_dicts(list,dist,gdfs,bl_dict,labels_dict,centroids_dict):
    for tuple in list:
        seed = tuple[0]
        dist[seed] = tuple[1]
        gdfs[seed] = tuple[2]
        bl_dict[seed] = tuple[3]
        labels_dict[seed] = tuple[4]
        centroids_dict[seed] = tuple[5]

def solve_multiple_times(seeds, gdfs, bl_dict, dist, labels_dict, centroids_dict):
    tick = time.time()
    if parallel:
        tuples_list = Parallel(n_jobs=-1)(delayed(solve_model_and_return)(seed, model) for seed in seeds)
        unpack_and_assign_to_dicts(tuples_list, dist, gdfs, bl_dict, labels_dict, centroids_dict)
        tock = time.time()
        print(f"Time spent in parallel solve equals:", tock - tick)
    else:
        for seed in seeds:
            solve_model_and_save_to_dict(seed, model, dist, gdfs, bl_dict)
            tock = time.time()
        print(f"Time spent in sequential solve equals:", tock - tick)
    return tock - tick

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_renewable_shapes',regions = 's-20')
    configure_logging(snakemake)
    save_fig = snakemake.config["renewable_zones"]["save_fig"]
    mainland_only = snakemake.config["renewable_zones"]["mainland_only"]
#    plot_selection_attr = snakemake.config["renewable_zones"]["plot_selection_attr"]
    loc_weight = snakemake.config["renewable_zones"]["loc_weight"]
    nb_iterations = snakemake.config["renewable_zones"]["nb_iterations"]
    parallel = snakemake.config["renewable_zones"]["parallel"]
    save_all_clusterings = snakemake.config["renewable_zones"]["save_all_clusterings"]

    #Load datasets of era and sara into dict
    ds = rsm.load_data_sets(snakemake.input)


    #Extract main parameters of clustering method from config file and wildcard
    gran = snakemake.config["renewable_zones"]["granularity"]
    attr_wc = snakemake.wildcards.regions.partition("-")[0]
    print(attr_wc)
    print(snakemake.wildcards.regions.partition("-")[2])
    if snakemake.wildcards.regions == "c":
        nb_regions = 1
        clusters = gpd.read_file(snakemake.input.country_shapes)
        rsm.save_to_geojson(clusters, snakemake.output.renewable_shapes)
    else:
        nb_regions = int(snakemake.wildcards.regions.partition("-")[2])

        # Name of the relevant data in datasets
        name = {'w': 'wnd100m', 's': 'influx_direct'}
        attr_1 = 'w'
        attr_2 = 's'

        # Get latitudes and longitudes from the dataset
        lat, lon = rsm.get_lons_lats_from_ds(ds)
        # And select the ones that occur in both datasets
        sel_lat, sel_lon = rsm.intersect_lats_lons(lat, lon)
        # Rescale lats and lons to desired granularity
        l1, l2 = rsm.get_rescaled_lats_and_lons(lat, lon, sel_lat, sel_lon, gran, attr_wc)
        # Rescale actual data to desired granularity
        at_1, pu = rsm.change_data_granularity(
            np.mean(ds[attr_1][name[attr_1]][:, :, :].data[:, sel_lat[attr_1], :][:, :, sel_lon[attr_1]],
                    axis=0), lat[attr_1][sel_lat[attr_1]], lon[attr_1][sel_lon[attr_1]], l1, l2, gran)
        at_2, pu = rsm.change_data_granularity(
            np.mean(ds[attr_2][name[attr_2]][:, :, :].data[:, sel_lat[attr_2], :][:, :, sel_lon[attr_2]],
                    axis=0), lat[attr_2][sel_lat[attr_2]], lon[attr_2][sel_lon[attr_2]], l1, l2, gran)

        #Create vector of repeated lats and lons
        lat_all_gran = np.tile(l1, len(l2))
        lon_all_gran = np.repeat(l2, len(l1))

        # Create the actual regional k-means model to be solved
        if attr_wc == 'ws':
            model,data,selection = rsm.generate_regional_kmeans_model_locs(lon_all_gran, lat_all_gran, [at_1, at_2,lat_all_gran,lon_all_gran],loc_weight, nb_regions)
        elif attr_wc == 'w':
            model,data,selection = rsm.generate_regional_kmeans_model_locs(lon_all_gran, lat_all_gran, [at_1,lat_all_gran,lon_all_gran],loc_weight, nb_regions)
        elif attr_wc == 's':
            model,data,selection = rsm.generate_regional_kmeans_model_locs(lon_all_gran, lat_all_gran, [at_2,lat_all_gran,lon_all_gran],loc_weight, nb_regions)


        # Now we start to actually solve the clustering problem. Seed is given for reproducibility
        np.random.seed(snakemake.config["renewable_zones"]["seed"])

        #Initialize some dictionaries to store informative values

        gdfs = dict()
        bl_dict = dict()
        dist = dict()
        labels_dict = dict()
        centroids_dict = dict()
        seeds = np.random.randint(0, 1e9, nb_iterations)

        #And solve the clustering problem model a number of times with different seed values

        timer = solve_multiple_times(seeds,gdfs,bl_dict,dist,labels_dict,centroids_dict)

        #Initialize some dictionaries to store informative values
        dist_per_att = dict()
        dist_per_seed = dict()
        bl_sums = dict()
        circumference = dict()


        for seed in dist:
            bl_sums[seed] = dict()
            d_per_at = (np.array([dist[seed][c_label].sum(axis=0) for c_label in dist[seed]]).sum(axis=0))
            dist_per_att[seed] = d_per_at
            dist_per_seed[seed] = d_per_at.sum()
            circumference[seed] = (4 - bl_dict[seed]).sum()
            for n in range(1, 5):
                bl_sums[seed][n] = np.count_nonzero(bl_dict[seed] == n)

        path_tosave_to = os.path.join(os.path.dirname(os.path.dirname(snakemake.output.renewable_shapes)), "ren_shapes" , f"gdfs_{loc_weight}_{parallel}")
        circ_min = float("inf")
        seed_min = 0
        for seed in gdfs:
            gdfs[seed]["dist_tot"] = dist_per_seed[seed]
            gdfs[seed]["circumference"] = circumference[seed]
            gdfs[seed]["total_time"] = timer

            if circumference[seed] < circ_min:
                circ_min = circumference[seed]
                seed_min = seed

            for at in range(len(dist_per_att[seed])):
                gdfs[seed][f"dist_per_{at}"] = dist_per_att[seed][at]

            #gdfs[seed]["dist_per_att"] = dist_per_att[seed]


            if not(os.path.exists(path_tosave_to)):
                os.makedirs(path_tosave_to)

            if save_all_clusterings:
                rsm.save_to_geojson(gdfs[seed],os.path.join(path_tosave_to,f"ren_clust_{snakemake.wildcards.regions}_{gran}_{seed}"))

        if save_fig:
            rsm.plot_regions_scatter_gran_and_save(lat_all_gran, lon_all_gran, model,labels_dict[seed_min], centroids_dict[seed_min],seed,snakemake.output,gran,snakemake.wildcards.regions)
        rsm.save_to_geojson(gdfs[seed_min],snakemake.output.renewable_shapes)







