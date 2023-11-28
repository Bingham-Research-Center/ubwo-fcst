"""Misc. utility functions
"""

import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in kilometers (You can change this to miles by using 3956)
    r = 6371

    # Compute distance
    distance = c * r

    return distance

def normalise_longitude(lon):
    """
    Normalize the longitude to the range [-180, 180].
    """
    return ((lon + 180) % 360) - 180

def find_closest_point(lat_array, lon_array, target_lat, target_lon):
    # Create an empty array to store distances
    lon_array = normalise_longitude(lon_array)
    distances = np.zeros(lat_array.shape)

    # Compute distance from the target to each point in the array
    for i in range(lat_array.shape[0]):
        for j in range(lat_array.shape[1]):
            distances[i, j] = haversine(target_lat, target_lon, lat_array[i, j], lon_array[i, j])

    # Find the index of the minimum distance
    closest_idx = np.unravel_index(np.argmin(distances), distances.shape)

    # Return the closest latitude and longitude
    closest_lat = lat_array[closest_idx]
    closest_lon = lon_array[closest_idx]

    # closest_lon = normalise_longitude(closest_lon)
    return closest_lat, closest_lon, closest_idx

