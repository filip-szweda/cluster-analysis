import pandas
from matplotlib import pyplot
from math import sqrt
from numpy import count_nonzero


def kmeans(data, k):
    values = data.values
    diff = True
    cluster = [0] * values.shape[0]
    centroids = data.sample(n=k).values
    while diff:
        for i, row in enumerate(values):
            min_dist = float('inf')
            for idx, centroid in enumerate(centroids):
                dist = sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    cluster[i] = idx
        new_centroids = pandas.DataFrame(values).groupby(by=cluster).mean().values
        if not count_nonzero(centroids - new_centroids):
            diff = False
        else:
            centroids = new_centroids
    return centroids, cluster


def clusters_to_colors(cluster):
    colors = []
    for i in range(len(cluster)):
        if cluster[i] == 0:
            colors.append('b')
        elif cluster[i] == 1:
            colors.append('g')
        elif cluster[i] == 2:
            colors.append('r')
        elif cluster[i] == 3:
            colors.append('c')
        elif cluster[i] == 4:
            colors.append('m')
        elif cluster[i] == 5:
            colors.append('y')
    return colors


def plot_kmeans(data, country, k):
    centroids, cluster = kmeans(data, k)
    colors = clusters_to_colors(cluster)

    pyplot.scatter(data.lng, data.lat, color=colors)

    for i in range(k):
        pyplot.scatter(centroids[i][1], centroids[i][0], s=125, color='k')

    pyplot.title("Concentration of cities in " + country + " as clasterized by kmeans")
    pyplot.xlabel('Longitude')
    pyplot.ylabel('Latitude')
    pyplot.show()
