from shapely.geometry import Point
import shapely
import pandas as pd
import numpy as np
import geopandas as gpd
from _helpers import configure_logging
import pypsa

def _set_renewable_zone(n, renewable_shapes, offshore_shapes):
    buses = n.buses

    def buses_in_shape(shape):
        shape = shapely.prepared.prep(shape)
        return pd.Series(
            np.fromiter((shape.contains(Point(x, y))
                        for x, y in buses.loc[:,["x", "y"]].values),
                        dtype=bool, count=len(buses)),
            index=buses.index
        )
    ren_zones = gpd.read_file(renewable_shapes).name
    renewable_shapes = gpd.read_file(renewable_shapes).set_index('name')['geometry']
    offshore_shapes = gpd.read_file(offshore_shapes).set_index('name')['geometry']

    for ren_zone in ren_zones:
        ren_shape = renewable_shapes[ren_zone]
        onshore_country_b = buses_in_shape(ren_shape)

        buses.loc[onshore_country_b, 'ren_zone'] = ren_zone

    rz_nan_b = buses.ren_zone.isnull()

    buses.loc[rz_nan_b,"ren_zone"] = buses.loc[rz_nan_b].apply(lambda row: find_renewable_zone(renewable_shapes, row), axis=1)



def find_renewable_zone(zones,bus):
    loc = Point(bus.x,bus.y)
    d = float('inf')
    for zone_name in zones.index:
        if zones[zone_name].contains(loc):
            return zone_name
        else:
            dz = loc.distance(zones[zone_name])
            if dz < d:
                d = dz
                closest = zone_name
    return closest


if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('add_renewable_zones')
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)
    _set_renewable_zone(n,snakemake.input.renewable_shapes,None)

    n.export_to_netcdf(snakemake.output[0])
