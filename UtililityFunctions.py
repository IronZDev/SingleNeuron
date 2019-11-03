import random
import numpy as np


# Calculating gaussian distribution
def calculate_gaussian(size=1000, number_of_clusters=3, mean_min=-10.0, mean_max=10.0, std_deviation_min=0.5,
                       std_deviation_max=2.0):
    # Generate random clasters in 2d
    colors_available = ['r', 'g']
    data = [[], [], []]
    for cluster in range(number_of_clusters):
        center = [random.uniform(mean_min, mean_max), random.uniform(mean_min, mean_max)]
        x, y = np.random.multivariate_normal(center,
                                             [[random.uniform(std_deviation_min, std_deviation_max), 0],
                                              [0, random.uniform(std_deviation_min, std_deviation_max)]],
                                             size).T
        data[0].append(x)
        data[1].append(y)
        # Assign a color to each cluster
        data[2].extend([colors_available[cluster % 2] for i in range(size)])
    # Merge points together

    data[0] = np.concatenate((data[0]))
    data[1] = np.concatenate((data[1]))
    return data


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
