import sys

from task_2.utils import read_data
from task_2.nurbs_surface import NurbsSurface

if __name__ == '__main__':
    path = sys.argv[1]

    points, indices, grid = read_data(path)

    is_weighted = True
    points_count = 50
    lower_weight = 0
    upper_weight = 1
    basis_degree = 3

    surface = NurbsSurface(points, grid, indices, basis_degree)

    if is_weighted:
        surface.generate_weights(lower_weight, upper_weight)

    surface.visualize(points_count, is_weighted)
