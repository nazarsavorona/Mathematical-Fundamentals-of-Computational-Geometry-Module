import numpy as np
import open3d
from numpy import sqrt

from task_2.utils import get_distance, get_random_weight, get_triangles, matrix2array


class NurbsSurface:
    def __init__(self, points, grid, indices, k):
        self.control_points = None
        self.data_points = None
        self.n = None
        self.k = None
        self.s_column = None
        self.s_row = None
        self.knots_column = None
        self.knots_row = None
        self.weights = None

        self.initialize_surface(points, grid, indices, k)

    def basis_function(self, t, i, k, knot_vector):
        if k == 0:
            if t == knot_vector[i] or knot_vector[i] < t < knot_vector[i + 1] or \
                    (t == knot_vector[len(knot_vector) - 1] and i == len(knot_vector) - self.k - 2):
                return 1
            return 0

        first_part = 0.
        first_denominator = (knot_vector[i + k] - knot_vector[i])
        if first_denominator != 0:
            first_part = (t - knot_vector[i]) / first_denominator * self.basis_function(t, i, k - 1, knot_vector)

        second_part = 0.
        second_denominator = (knot_vector[i + k + 1] - knot_vector[i + 1])
        if second_denominator != 0:
            second_part = (knot_vector[i + k + 1] - t) / second_denominator * self.basis_function(t, i + 1, k - 1,
                                                                                                  knot_vector)

        return first_part + second_part

    def initialize_surface(self, points, grid, indices, k):
        self.n = grid[0]
        self.k = k
        self.data_points = np.zeros((self.n, self.n, 3))

        for i, grid_position in enumerate(indices):
            self.data_points[grid_position[0], grid_position[1]] = points[i]

        self.weights = np.ones((self.n, self.n), dtype=np.float32)

        self.find_s()
        self.generate_knots()
        self.generate_control_points()

    def find_params_vector(self, points_vector):
        n = self.n

        result = np.zeros(n, dtype=np.float32)
        result[0] = 0.
        result[n - 1] = 1.

        d = sum(sqrt(get_distance(points_vector[i], points_vector[i - 1])) for i in range(1, n))
        for i in range(1, n - 1):
            result[i] = result[i - 1] + sqrt(get_distance(points_vector[i], points_vector[i - 1])) / d
        return result

    def find_s_vector(self, parameters_matrix):
        n = self.n
        result = np.zeros(n, dtype=np.float32)
        for i in range(n):
            temp_vector = parameters_matrix[:, i]
            result[i] = sum(temp_vector[j] for j in range(n)) / n
        return result

    def find_s(self):
        n = self.n

        column_matrix = np.zeros((n, n), dtype=np.float32)
        row_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            column_matrix[i] = self.find_params_vector(self.data_points[:, i])
            row_matrix[i] = self.find_params_vector(self.data_points[i])

        self.s_column = self.find_s_vector(column_matrix)
        self.s_row = self.find_s_vector(row_matrix)

    def generate_knots(self):
        k = self.k
        n = self.n
        knots_num = n + k + 1

        self.knots_column = np.zeros(knots_num, dtype=np.float32)
        self.knots_row = np.zeros(knots_num, dtype=np.float32)

        for j in range(1, n - k):
            self.knots_column[j + k] = 1. / k * sum(self.s_column[m] for m in range(j, j + k))
            self.knots_row[j + k] = 1. / k * sum(self.s_row[m] for m in range(j, j + k))

        for j in range(k + 1):
            self.knots_column[j] = 0.
            self.knots_row[j] = 0.
            self.knots_column[knots_num - j - 1] = 1.
            self.knots_row[knots_num - j - 1] = 1.

    def generate_control_points(self):
        n = self.n
        k = self.k

        q_matrix = np.zeros((n, n, 3))
        self.control_points = np.zeros((n, n, 3))
        temp_matrix = np.zeros((n, n), dtype=np.float32)

        for d in range(n):
            for i in range(n):
                for j in range(n):
                    temp_matrix[i, j] = self.basis_function(self.s_column[i], j, k, self.knots_column)
            q_matrix[:, d] = np.linalg.solve(temp_matrix, self.data_points[:, d])

        for c in range(n):
            for i in range(n):
                for j in range(n):
                    temp_matrix[i, j] = self.basis_function(self.s_row[i], j, k, self.knots_row)
            self.control_points[c] = np.linalg.solve(temp_matrix, q_matrix[c])

    def get_nurbs_denominator(self, u, v):
        k = self.k
        n = self.n
        result = 0.

        for i in range(n):
            temp = 0
            for j in range(n):
                n_j_k = self.basis_function(v, j, k, self.knots_row)
                temp += n_j_k * self.weights[i, j]
            n_i_k = self.basis_function(u, i, k, self.knots_column)
            result += temp * n_i_k

        return result

    def get_point_on_surface(self, u, v, is_weighted=False):
        k = self.k
        n = self.n
        result = np.zeros(3)

        for i in range(n):
            temp = 0
            for j in range(n):
                n_j_k = self.basis_function(v, j, k, self.knots_row)
                current_term = n_j_k * self.control_points[i, j]
                if is_weighted:
                    current_term *= self.weights[i, j]
                temp += current_term
            n_i_k = self.basis_function(u, i, k, self.knots_column)
            result += temp * n_i_k

        return result

    def get_surface_mesh(self, points_count, is_weighted=False):
        surface_points = []
        u = np.linspace(0, 1, points_count)
        v = np.linspace(0, 1, points_count)

        for i in range(len(u)):
            for j in range(len(v)):
                current_point = self.get_point_on_surface(u[i], v[j], is_weighted)
                if is_weighted:
                    current_point /= self.get_nurbs_denominator(u[i], v[j])
                surface_points.append(current_point)
        return surface_points

    def generate_weights(self, lower_bound, upper_bound):
        n = self.n

        for i in range(n):
            for j in range(n):
                self.weights[i, j] = get_random_weight(lower_bound, upper_bound)

    def get_mesh(self, number_of_points, is_weighted=False):
        points_array = self.get_surface_mesh(number_of_points, is_weighted)
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(points_array)
        triangles = get_triangles(number_of_points)
        mesh.triangles = open3d.utility.Vector3iVector(triangles)
        return mesh

    def visualize(self, number_of_points, is_weighted=False):
        data_points = open3d.geometry.PointCloud()
        data_points.points = open3d.utility.Vector3dVector(matrix2array(self.data_points))

        surface_mesh = self.get_mesh(number_of_points, is_weighted)
        open3d.visualization.draw_geometries([data_points, surface_mesh], mesh_show_back_face=True,
                                             mesh_show_wireframe=True)
