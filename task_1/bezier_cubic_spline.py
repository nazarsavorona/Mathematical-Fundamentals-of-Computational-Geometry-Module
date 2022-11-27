import cv2
import numpy as np

from task_1.utils import draw_points


class BezierCubicSpline:
    def __init__(self, points):
        self.points = None
        self.a_control_points = None
        self.b_control_points = None

        self.find_control_points(points)

    def find_control_points(self, points):
        self.points = np.array(points, dtype=np.float32)

        n = len(self.points) - 1

        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2

        v = [2 * (2 * self.points[i] + self.points[i + 1]) for i in range(n)]
        v[0] = self.points[0] + 2 * self.points[1]
        v[n - 1] = 8 * self.points[n - 1] + self.points[n]

        self.a_control_points = np.linalg.solve(C, v)
        self.b_control_points = [0] * n

        for i in range(n - 1):
            self.b_control_points[i] = 2 * self.points[i + 1] - self.a_control_points[i + 1]

        self.b_control_points[n - 1] = (self.a_control_points[n - 1] + self.points[n]) / 2

    def draw(self, img):
        t = np.linspace(0, 1, 75)

        for i in range(0, len(self.points) - 1):
            start_point = self.points[i]
            end_point = self.points[i + 1]
            first_control_point = self.a_control_points[i]
            second_control_point = self.b_control_points[i]

            for k in range(0, len(t) - 1):
                interval_start = BezierCubicSpline.get_point(start_point, first_control_point, second_control_point,
                                                             end_point, t[k])
                interval_end = BezierCubicSpline.get_point(start_point, first_control_point, second_control_point,
                                                           end_point, t[k + 1])

                cv2.line(img, (int(interval_start[0]), int(interval_start[1])),
                         (int(interval_end[0]), int(interval_end[1])), (0, 255, 255), 3)

        draw_points(img, self.points, (255, 0, 0), 5)

    @staticmethod
    def get_point(p1, a, b, p2, t):
        return pow(1 - t, 3) * p1 + 3 * pow(1 - t, 2) * t * a + 3 * (1 - t) * t * t * b + t * t * t * p2
