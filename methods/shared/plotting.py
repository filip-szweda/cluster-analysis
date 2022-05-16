import numpy
from matplotlib import pyplot
from scipy.spatial import ConvexHull
from scipy import interpolate


def scatter_values(values, clusters, color_map):
    for value_number, value in enumerate(values):
        pyplot.scatter(value[1], value[0], s=25, marker='o', color=color_map(clusters[value_number]))


def scatter_centroids(centroids, color_map):
    for cluster_number, centroid in enumerate(centroids):
        pyplot.scatter(centroid[1], centroid[0], s=125, marker='^', color=color_map(cluster_number))


def fill_interpolated_areas(k, clusters, values, color_map):
    indexes_by_cluster = [[] for _ in range(k)]
    for value_number in range(len(values)):
        indexes_by_cluster[clusters[value_number]].append(value_number)

    for cluster_number in range(k):
        cluster_values = values[indexes_by_cluster[cluster_number]]
        hull = ConvexHull(cluster_values)
        x_hull = numpy.append(cluster_values[hull.vertices, 0], cluster_values[hull.vertices, 0][0])
        y_hull = numpy.append(cluster_values[hull.vertices, 1], cluster_values[hull.vertices, 1][0])

        dist = numpy.sqrt((x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2)
        dist_along = numpy.concatenate(([0], dist.cumsum()))
        spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0)
        interp_d = numpy.linspace(dist_along[0], dist_along[-1], 50)
        interp_y, interp_x = interpolate.splev(interp_d, spline)

        pyplot.fill(interp_x, interp_y, '--', c=color_map(cluster_number), alpha=0.2)
