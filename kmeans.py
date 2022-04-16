from matplotlib import pyplot
from math import sqrt
import numpy
from scipy.spatial import ConvexHull
from scipy import interpolate
import random


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    x = 0.0
    y = 0.0
    cluster = 0


def data_to_points(data):
    values = data.values
    points = []
    for _, coordinates in enumerate(values):
        points.append(Point(coordinates[1], coordinates[0]))
    return points


def group_points_by_clusters(points, k):
    grouped_points = [[] for _ in range(k)]
    for point in points:
        grouped_points[point.cluster].append(point)
    return grouped_points


def calculate_new_centroids(points, k):
    grouped_points = group_points_by_clusters(points, k)
    new_centroids = [Point(0.0, 0.0) for _ in range(k)]
    for i in range(k):
        sum_x = 0
        sum_y = 0
        for point in grouped_points[i]:
            sum_x += point.x
            sum_y += point.y
        new_centroids[i] = Point(sum_x / len(grouped_points[i]), sum_y / len(grouped_points[i]))
    return new_centroids


def kmeans(points, k):
    centroids = random.sample(points, k)
    diff = True
    while diff:
        for i in range(len(points)):
            min_dist = float('inf')
            for j in range(len(centroids)):
                dist = sqrt((centroids[j].x - points[i].x) ** 2 + (centroids[j].y - points[i].y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    points[i].cluster = j
        new_centroids = calculate_new_centroids(points, k)

        # check if centroids and new_centroids are equal, can be refactored
        zeroes = 0
        for i in range(k):
            if centroids[i].x - new_centroids[i].x + centroids[i].y - new_centroids[i].y:
                centroids = new_centroids
                break
            elif zeroes == k - 1:
                diff = False
                break
            else:
                zeroes += 1

    return points, centroids


def plot_points(points, color_map):
    for point in points:
        pyplot.scatter(point.x, point.y, s=25, marker='o', color=color_map(point.cluster))


def plot_centroids(k, centroids, color_map):
    for i in range(k):
        pyplot.scatter(centroids[i].x, centroids[i].y, s=125, marker='^', color=color_map(i))


def plot_interpolated_areas(k, points, color_map):
    # prepare rows x 2 sized array for ConvexHull, can be refactored
    points_list = [[] for _ in range(k)]
    for point in points:
        points_list[point.cluster].append(point.y)
        points_list[point.cluster].append(point.x)
    points_by_cluster = []
    for i in range(k):
        points_by_cluster.append(numpy.reshape(numpy.array(points_list[i]), (-1, 2)))

    for i in range(k):
        hull = ConvexHull(points_by_cluster[i])
        x_hull = numpy.append(points_by_cluster[i][hull.vertices, 0], points_by_cluster[i][hull.vertices, 0][0])
        y_hull = numpy.append(points_by_cluster[i][hull.vertices, 1], points_by_cluster[i][hull.vertices, 1][0])

        dist = numpy.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        dist_along = numpy.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0)
        interp_d = numpy.linspace(dist_along[0], dist_along[-1], 50)
        interp_y, interp_x = interpolate.splev(interp_d, spline)

        pyplot.fill(interp_x, interp_y, '--', c=color_map(i), alpha=0.2)


def plot_kmeans(data, country, k):
    points = data_to_points(data)
    points, centroids = kmeans(points, k)
    color_map = pyplot.cm.get_cmap("hsv", k + 1)

    plot_points(points, color_map)
    plot_centroids(k, centroids, color_map)
    plot_interpolated_areas(k, points, color_map)

    pyplot.title("Concentration of cities in " + country + " as clasterized by kmeans", fontsize=14)
    pyplot.xlabel('Longitude', fontsize=14)
    pyplot.ylabel('Latitude', fontsize=14)
    pyplot.show()
