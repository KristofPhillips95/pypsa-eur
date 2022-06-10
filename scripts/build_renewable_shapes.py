import geopandas as gpd
import geoplot
from shapely.ops import unary_union
import os
from _helpers import configure_logging



def save_to_geojson(df, fn):
    if os.path.exists(fn):
        os.unlink(fn)
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(dict(geometry=df))
    df = df.reset_index()
    schema = {**gpd.io.file.infer_schema(df), 'geometry': 'Unknown'}
    df.to_file(fn, driver='GeoJSON', schema=schema)


if __name__ == "__main__":

    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('build_renewable_shapes')
    configure_logging(snakemake)
    countries = gpd.read_file(snakemake.input.country_shapes)
    offshore = gpd.read_file(snakemake.input.offshore_shapes)
    #europe = gpd.read_file("resources/europe_shape.geojson")

    latitudes = []
    for country in countries.geometry:
        latitudes.append(country.centroid.y)
    max_lat = max(latitudes)
    min_lat = min(latitudes)
    center = (max_lat + min_lat) / 2
    north_list = []
    south_list = []
    offshore_list = []
    for ind in countries.index:
        if countries.geometry[ind].centroid.y >= center:
            north_list.append(countries.geometry[ind])
        elif countries.geometry[ind].centroid.y <= center:
            south_list.append(countries.geometry[ind])
    for ind in offshore.index:
        offshore_list.append(offshore.geometry[ind])


    un = unary_union(north_list)
    us = unary_union(south_list)
    uo = unary_union(offshore_list)

    d = {"name": ["North", "South", "Offshore"], "geometry": [un, us,uo]}
    ren_zones = gpd.GeoDataFrame(d)

    #save_to_geojson(ren_zones,snakemake.output.ren_shapes)
    save_to_geojson(ren_zones,snakemake.output.renewable_shapes)