import cv2
import json


def read_points(path):
    return json.load(open(path))['curve']


def draw_points(img, points, color, radius):
    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)
