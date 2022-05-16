import numpy
from matplotlib import pyplot
from math import sqrt
from scipy.spatial import ConvexHull
from scipy import interpolate

from kmeans import kmeans


def sum_distances_from_centroid(centroid, values):
    dist_sum = 0
    for value in values:
        dist_sum += sqrt((centroid[0] - value[0]) ** 2 + (centroid[1] - value[1]) ** 2)
    return dist_sum


def bisecting_kmeans(values, k):
    centroids, clusters = kmeans(values, 2)
    while len(centroids) < k:
        max_dist = float('-inf')
        values_to_divide = []
        indices_to_divide = []
        for cluster in range(len(clusters)):
            indices = []
            for index, assigned_cluster in enumerate(clusters):
                if assigned_cluster == cluster:
                    indices.append(index)
            cluster_values = [values[i] for i in indices]
            dist = sum_distances_from_centroid(centroids[cluster], cluster_values)
            if dist > max_dist:
                max_dist = dist
                values_to_divide = cluster_values
                indices_to_divide = indices
                centroids_to_del = cluster

        new_centroids, new_clusters = kmeans(values_to_divide, 2)
        for i in range(len(new_clusters)):
            new_clusters[i] += len(centroids) # this is wrong
        for i, indice in enumerate(indices_to_divide):
            clusters[indice] = new_clusters[i]
        del centroids[centroids_to_del]
        for cen in new_centroids:
            centroids.append(cen)
    return centroids, clusters


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


def plot_bisecting_kmeans(data, country, k):
    values = data.values
    centroids, clusters = bisecting_kmeans(values, k)
    color_map = pyplot.cm.get_cmap("hsv", k + 1)

    scatter_values(values, clusters, color_map)
    scatter_centroids(centroids, color_map)
    fill_interpolated_areas(k, clusters, values, color_map)

    pyplot.title("Concentration of cities in " + country + " as clasterized by kmeans")
    pyplot.xlabel('Longitude')
    pyplot.ylabel('Latitude')
    pyplot.show()
