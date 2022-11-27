import sys

import cv2
import numpy as np

from task_1.utils import read_points, draw_points
from task_1.bezier_cubic_spline import BezierCubicSpline

if __name__ == '__main__':
    path = sys.argv[1]

    points = read_points(path)

    cubicSpline = BezierCubicSpline(points)

    img = np.ones((700, 800, 3), dtype=np.uint8) * 255

    draw_points(img, points, (255, 0, 0), 5)
    cv2.imshow("Given Points", img)

    cubicSpline.draw(img)
    cv2.imshow("Bezier Cubic Spline", img)

    cv2.waitKey(0)
