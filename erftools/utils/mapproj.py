def calculate_utm_zone(longitude):
     """
     Calculate the UTM zone for a given longitude.
     """
     return int((longitude + 180) // 6) + 1
