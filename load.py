import numpy as np
import pandas as pd
import os
from datetime import datetime

def load_all_sis_data(folder_path):
    """
    Loads all SIS data files in a folder into a single 3D NumPy array,
    creating a dictionary to map element names to array indices.

    Args:
        folder_path (str): The path to the folder containing the SIS data files.

    Returns:
        numpy.ndarray: A 3D NumPy array (energy, time, element) representing the flux data.
              The 'element' dimension corresponds to the order in which files are loaded.
        numpy.ndarray: A 1D NumPy array of datetime objects representing the time axis.
        dict: A dictionary mapping element names to their corresponding indices in the 
              third dimension of the data array.
    """

    all_flux_data = []
    datetime_values = None
    element_mapping = {}  # Dictionary to store element-index mapping

    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            element_name = filename.split("_")[0].capitalize()  # Extract element name
            element_mapping[element_name] = i  # Map element name to index

            data = np.loadtxt(filepath, skiprows=25)
            fp_year = data[:, 0]
            flux_values = data[:, 1:9]

            if datetime_values is None:
                # Use higher precision for fractional years
                datetime_values = np.array([pd.Timestamp(datetime(year=int(fp), month=1, day=1) +
                                                pd.Timedelta(days=(fp - int(fp)) * 365))
                                            for fp in fp_year])

            all_flux_data.append(flux_values)

    all_flux_data = np.stack(all_flux_data, axis=2).transpose(1, 0, 2)
    return all_flux_data, datetime_values, element_mapping