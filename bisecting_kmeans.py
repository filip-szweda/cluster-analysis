import numpy
from matplotlib import pyplot
from math import sqrt
from scipy.spatial import ConvexHull
from scipy import interpolate

from kmeans import kmeans, scatter_values, scatter_centroids, fill_interpolated_areas


def sum_distances_from_centroid(centroid, values):
    dist_sum = 0
    for value in values:
        dist_sum += sqrt((centroid[0] - value[0]) ** 2 + (centroid[1] - value[1]) ** 2)
    return dist_sum


def bisecting_kmeans(values, k):
    centroids, clusters = kmeans(values, 2)

    while len(centroids) < k:
        max_dist = float('-inf')
        for i, centroid in enumerate(centroids):
            cluster_values_indexes = [index for index, cluster in enumerate(clusters) if cluster == i]
            cluster_values = [values[index] for index in cluster_values_indexes]
            dist = sum_distances_from_centroid(centroid, cluster_values)
            if dist > max_dist:
                max_dist = dist
                values_to_divide = cluster_values
                values_indexes_to_divide = cluster_values_indexes
                centroid_index_to_del = i

        new_centroids, new_clusters = kmeans(values_to_divide, 2)

        max_cluster = max(clusters)
        for i, _ in enumerate(new_clusters):
            new_clusters[i] += max_cluster + 1

        for i, index in enumerate(values_indexes_to_divide):
            clusters[index] = new_clusters[i]

        del centroids[centroid_index_to_del]
        for new_centroid in new_centroids:
            centroids.append(new_centroid)

        for i, _ in enumerate(clusters):
            clusters[i] += k
        for i in range(len(centroids) - 1, -1, -1):
            max_cluster = max(clusters)
            clusters[:] = [x if x != max_cluster else i for x in clusters]

    return centroids, clusters


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
