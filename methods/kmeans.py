from pandas import DataFrame
from numpy import count_nonzero
from random import sample
from matplotlib import pyplot
from math import sqrt

import methods.shared.plotting


def kmeans(values, k):
    diff = True
    clusters = [0 for _ in range(len(values))]
    centroids = sample(list(values), k)
    while diff:
        for value_number, value in enumerate(values):
            min_dist = float('inf')
            for cluster_number, centroid in enumerate(centroids):
                dist = sqrt((centroid[0] - value[0]) ** 2 + (centroid[1] - value[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    clusters[value_number] = cluster_number
        new_centroids = DataFrame(values).groupby(by=clusters).mean().values
        if k == 2 or not count_nonzero(centroids - new_centroids):
            diff = False
        else:
            centroids = new_centroids
    return centroids, clusters


def plot_kmeans(data, country, k):
    values = data.values
    centroids, clusters = kmeans(values, k)
    color_map = pyplot.cm.get_cmap("hsv", k + 1)

    methods.shared.plotting.scatter_values(values, clusters, color_map)
    methods.shared.plotting.scatter_centroids(centroids, color_map)
    methods.shared.plotting.fill_interpolated_areas(k, clusters, values, color_map)

    pyplot.title("Concentration of cities in " + country + " as clasterized by kmeans")
    pyplot.xlabel('Longitude')
    pyplot.ylabel('Latitude')
    pyplot.show()
