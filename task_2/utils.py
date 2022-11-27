import json
import math
import random

import numpy as np


def get_distance(first_point, second_point):
    return math.sqrt(
        (first_point[0] - second_point[0]) ** 2 + (first_point[1] - second_point[1]) ** 2 + (
                first_point[2] - second_point[2]) ** 2)


def get_random_weight(lower_bound, upper_bound):
    return random.uniform(lower_bound, upper_bound)


def matrix2array(matrix):
    n = len(matrix)
    m = len(matrix[0])
    result_array = np.zeros((n * n, 3))

    for i in range(n):
        for j in range(m):
            result_array[i * n + j] = matrix[i, j]

    return result_array


def get_triangles(n):
    result_array = []

    for i in range(n - 1):
        for j in range(n - 1):
            result_array.append(np.array([i * n + j, i * n + j + n, i * n + j + 1]).astype(np.int32))
            result_array.append(np.array([i * n + j + 1, i * n + j + n, i * n + j + n + 1]).astype(np.int32))

    return np.asarray(result_array)


def read_data(path):
    data = json.load(open(path))["surface"]

    points = data["points"]
    indices = data["indices"]
    grid = data["gridSize"]

    return points, indices, grid
