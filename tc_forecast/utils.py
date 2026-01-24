"""
Utility functions for TC Forecast
"""


import math
import numpy as np
from typing import Tuple




def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
   """
   Calculate great circle distance between two points on Earth
  
   Args:
       lat1, lon1: First point coordinates
       lat2, lon2: Second point coordinates
      
   Returns:
       Distance in kilometers
   """
   ra = 6378.137  # Earth radius in km
  
   # Convert to radians
   lat1_rad = math.radians(lat1)
   lon1_rad = math.radians(lon1)
   lat2_rad = math.radians(lat2)
   lon2_rad = math.radians(lon2)
  
   # Haversine formula
   dlat = lat2_rad - lat1_rad
   dlon = lon2_rad - lon1_rad
  
   a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
   c = 2 * math.asin(math.sqrt(a))
  
   distance = ra * c
   return distance




def offsets_to_coordinates(offsets: np.ndarray, last_lat: float, last_lon: float
                         ) -> Tuple[np.ndarray, np.ndarray]:
   """
   Convert lat/lon offsets to absolute coordinates
  
   Args:
       offsets: Array of shape (n_steps * 2,) with [lat0, lon0, lat1, lon1, ...]
       last_lat: Last known latitude
       last_lon: Last known longitude
      
   Returns:
       lats, lons: Arrays of absolute coordinates
   """
   n_steps = len(offsets) // 2
  
   lats = []
   lons = []
  
   for i in range(n_steps):
       lat = offsets[i * 2] + last_lat
       lon = offsets[i * 2 + 1] + last_lon
       lats.append(lat)
       lons.append(lon)
  
   return np.array(lats), np.array(lons)


