import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from numba import jit
import numpy as np
from mpi4py import MPI
import sys

# Load the CSV file
df = pd.read_csv('/home/kaiwen1/30123-Project-Kaiwen/NDVI-sample.csv')

# Load the GeoJSON and shapefiles
local_roads = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-local-roads.geojson')
main_roads = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-main-roads.geojson')
villages = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-villages.geojson')
protected_areas = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-protected/protected_areas.shp')
waterways = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-water/hotosm_cod_waterways_lines_shp.shp')

target_crs = "EPSG:4326"  # Common CRS (WGS 84)
projected_crs = "EPSG:3857"  # Common projected CRS (Pseudo-Mercator)

# Function to set CRS if it's not set
def set_crs_if_needed(gdf, target_crs):
    if gdf.crs is None:
        gdf.set_crs(target_crs, inplace=True)
    return gdf

# Ensure all GeoDataFrames have the same CRS
local_roads = set_crs_if_needed(local_roads, target_crs)
main_roads = set_crs_if_needed(main_roads, target_crs)
villages = set_crs_if_needed(villages, target_crs)
protected_areas = set_crs_if_needed(protected_areas, target_crs)
waterways = set_crs_if_needed(waterways, target_crs)

# Reproject to projected CRS before calculating centroids
villages = villages.to_crs(projected_crs)
local_roads = local_roads.to_crs(projected_crs)
main_roads = main_roads.to_crs(projected_crs)
protected_areas = protected_areas.to_crs(projected_crs)
waterways = waterways.to_crs(projected_crs)

# Convert geometries to centroids
villages['geometry'] = villages['geometry'].centroid
local_roads['geometry'] = local_roads['geometry'].centroid
main_roads['geometry'] = main_roads['geometry'].centroid
protected_areas['geometry'] = protected_areas['geometry'].centroid
waterways['geometry'] = waterways['geometry'].centroid

# Transform all GeoDataFrames back to the target CRS
local_roads = local_roads.to_crs(target_crs)
main_roads = main_roads.to_crs(target_crs)
villages = villages.to_crs(target_crs)
protected_areas = protected_areas.to_crs(target_crs)
waterways = waterways.to_crs(target_crs)

# Convert the CSV data to a GeoDataFrame
df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
mining_locations = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

@jit(nopython=True)
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # 6371 km is the radius of the Earth
    km = 6371 * c
    m = km * 1000
    return m

def calculate_nearest_distance(lon, lat, lons, lats):
    min_dist = np.inf
    for i in range(len(lons)):
        dist = haversine(lon, lat, lons[i], lats[i])
        if dist < min_dist:
            min_dist = dist
    return min_dist

def main():
    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide the mining_locations among the processes
    chunk_size = len(mining_locations) // size
    if rank == size - 1:
        chunk = mining_locations[rank * chunk_size:]
    else:
        chunk = mining_locations[rank * chunk_size:(rank + 1) * chunk_size]

    # Print the number of data points each process is handling and flush the output
    print(f"Rank {rank}: Processing {len(chunk)} data points", flush=True)

    mining_lons, mining_lats = chunk.geometry.x.values, chunk.geometry.y.values
    village_lons, village_lats = villages.geometry.x.values, villages.geometry.y.values
    waterway_lons, waterway_lats = waterways.geometry.x.values, waterways.geometry.y.values
    local_road_lons, local_road_lats = local_roads.geometry.x.values, local_roads.geometry.y.values
    main_road_lons, main_road_lats = main_roads.geometry.x.values, main_roads.geometry.y.values
    protected_area_lons, protected_area_lats = protected_areas.geometry.x.values, protected_areas.geometry.y.values

    distances_to_village = np.empty(len(chunk))
    distances_to_waterway = np.empty(len(chunk))
    distances_to_local_road = np.empty(len(chunk))
    distances_to_main_road = np.empty(len(chunk))
    distances_to_protected_area = np.empty(len(chunk))

    for i in range(len(chunk)):
        lon, lat = mining_lons[i], mining_lats[i]
        distances_to_village[i] = calculate_nearest_distance(lon, lat, village_lons, village_lats)
        distances_to_waterway[i] = calculate_nearest_distance(lon, lat, waterway_lons, waterway_lats)
        distances_to_local_road[i] = calculate_nearest_distance(lon, lat, local_road_lons, local_road_lats)
        distances_to_main_road[i] = calculate_nearest_distance(lon, lat, main_road_lons, main_road_lats)
        distances_to_protected_area[i] = calculate_nearest_distance(lon, lat, protected_area_lons, protected_area_lats)

        # Print statement to track progress and flush the output
        print(f"Rank {rank}: Processed point {i + 1}/{len(chunk)}", flush=True)

    chunk['distance_to_village'] = distances_to_village
    chunk['distance_to_waterway'] = distances_to_waterway
    chunk['distance_to_local_road'] = distances_to_local_road
    chunk['distance_to_main_road'] = distances_to_main_road
    chunk['distance_to_protected_area'] = distances_to_protected_area

    # Gather results from all processes
    gathered_chunks = comm.gather(chunk, root=0)

    if rank == 0:
        # Concatenate all chunks
        result_df = pd.concat(gathered_chunks)
        # Save the updated DataFrame to a new CSV file
        result_df.to_csv('/home/kaiwen1/30123-Project-Kaiwen/NDVI-sample-with-distances_newer.csv', index=False)
        # Display the first few rows of the updated DataFrame
        print(result_df.head(), flush=True)

if __name__ == "__main__":
    main()


# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# from numba import jit
# import numpy as np
# from mpi4py import MPI
# import sys

# # Load the CSV file
# df = pd.read_csv('/home/kaiwen1/30123-Project-Kaiwen/NDVI-sample.csv')

# # Load the GeoJSON and shapefiles
# local_roads = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-local-roads.geojson')
# main_roads = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-main-roads.geojson')
# villages = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-villages.geojson')
# protected_areas = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-protected/protected_areas.shp')
# waterways = gpd.read_file('/home/kaiwen1/30123-Project-Kaiwen/Congo-water/hotosm_cod_waterways_lines_shp.shp')

# target_crs = "EPSG:4326"  # Common CRS (WGS 84)
# projected_crs = "EPSG:3857"  # Common projected CRS (Pseudo-Mercator)

# # Function to set CRS if it's not set
# def set_crs_if_needed(gdf, target_crs):
#     if gdf.crs is None:
#         gdf.set_crs(target_crs, inplace=True)
#     return gdf

# # Ensure all GeoDataFrames have the same CRS
# local_roads = set_crs_if_needed(local_roads, target_crs)
# main_roads = set_crs_if_needed(main_roads, target_crs)
# villages = set_crs_if_needed(villages, target_crs)
# protected_areas = set_crs_if_needed(protected_areas, target_crs)
# waterways = set_crs_if_needed(waterways, target_crs)

# # Reproject to projected CRS before calculating centroids
# villages = villages.to_crs(projected_crs)
# local_roads = local_roads.to_crs(projected_crs)
# main_roads = main_roads.to_crs(projected_crs)
# protected_areas = protected_areas.to_crs(projected_crs)
# waterways = waterways.to_crs(projected_crs)

# # Convert geometries to centroids
# villages['geometry'] = villages['geometry'].centroid
# local_roads['geometry'] = local_roads['geometry'].centroid
# main_roads['geometry'] = main_roads['geometry'].centroid
# protected_areas['geometry'] = protected_areas['geometry'].centroid
# waterways['geometry'] = waterways['geometry'].centroid

# # Transform all GeoDataFrames back to the target CRS
# local_roads = local_roads.to_crs(target_crs)
# main_roads = main_roads.to_crs(target_crs)
# villages = villages.to_crs(target_crs)
# protected_areas = protected_areas.to_crs(target_crs)
# waterways = waterways.to_crs(target_crs)

# # Convert the CSV data to a GeoDataFrame
# df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
# mining_locations = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# @jit(nopython=True)
# def haversine(lon1, lat1, lon2, lat2):
#     # convert decimal degrees to radians
#     lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
#     # haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     c = 2 * np.arcsin(np.sqrt(a))
    
#     # 6371 km is the radius of the Earth
#     km = 6371 * c
#     m = km * 1000
#     return m

# def calculate_nearest_distance(lon, lat, lons, lats):
#     min_dist = np.inf
#     for i in range(len(lons)):
#         dist = haversine(lon, lat, lons[i], lats[i])
#         if dist < min_dist:
#             min_dist = dist
#     return min_dist

# def main():
#     # MPI setup
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     # Divide the mining_locations among the processes
#     chunk_size = len(mining_locations) // size
#     if rank == size - 1:
#         chunk = mining_locations[rank * chunk_size:]
#     else:
#         chunk = mining_locations[rank * chunk_size:(rank + 1) * chunk_size]

#     # Print the number of data points each process is handling and flush the output
#     print(f"Rank {rank}: Processing {len(chunk)} data points", flush=True)

#     mining_lons, mining_lats = chunk.geometry.x.values, chunk.geometry.y.values
#     village_lons, village_lats = villages.geometry.x.values, villages.geometry.y.values
#     waterway_lons, waterway_lats = waterways.geometry.x.values, waterways.geometry.y.values
#     local_road_lons, local_road_lats = local_roads.geometry.x.values, local_roads.geometry.y.values
#     main_road_lons, main_road_lats = main_roads.geometry.x.values, main_roads.geometry.y.values
#     protected_area_lons, protected_area_lats = protected_areas.geometry.x.values, protected_areas.geometry.y.values

#     distances_to_village = np.empty(len(chunk))
#     distances_to_waterway = np.empty(len(chunk))
#     distances_to_local_road = np.empty(len(chunk))
#     distances_to_main_road = np.empty(len(chunk))
#     distances_to_protected_area = np.empty(len(chunk))

#     for i in range(len(chunk)):
#         lon, lat = mining_lons[i], mining_lats[i]
#         distances_to_village[i] = calculate_nearest_distance(lon, lat, village_lons, village_lats)
#         distances_to_waterway[i] = calculate_nearest_distance(lon, lat, waterway_lons, waterway_lats)
#         distances_to_local_road[i] = calculate_nearest_distance(lon, lat, local_road_lons, local_road_lats)
#         distances_to_main_road[i] = calculate_nearest_distance(lon, lat, main_road_lons, main_road_lats)
#         distances_to_protected_area[i] = calculate_nearest_distance(lon, lat, protected_area_lons, protected_area_lats)

#         # Print statement to track progress and flush the output
#         print(f"Rank {rank}: Processed point {i + 1}/{len(chunk)}", flush=True)

#     chunk['distance_to_village'] = distances_to_village
#     chunk['distance_to_waterway'] = distances_to_waterway
#     chunk['distance_to_local_road'] = distances_to_local_road
#     chunk['distance_to_main_road'] = distances_to_main_road
#     chunk['distance_to_protected_area'] = distances_to_protected_area

#     # Gather results from all processes
#     gathered_chunks = comm.gather(chunk, root=0)

#     if rank == 0:
#         # Concatenate all chunks
#         result_df = pd.concat(gathered_chunks)
#         # Save the updated DataFrame to a new CSV file
#         result_df.to_csv('/home/kaiwen1/30123-Project-Kaiwen/NDVI-sample-with-distances.csv', index=False)
#         # Display the first few rows of the updated DataFrame
#         print(result_df.head(), flush=True)

# if __name__ == "__main__":
#     main()


