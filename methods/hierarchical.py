import math
from math import sqrt
from matplotlib import pyplot
import methods.shared.plotting
import scipy.cluster.hierarchy as shc


def hierarchical_clustering(data, n):
    number_of_clusters = data.shape[0]
    clusters = []
    points = list(data.index)

    idx = 0
    # convert data to a list of clusters
    for i in points:
        tmp = list()
        clusters.append(tmp)
        clusters[idx].append(i)
        idx += 1

    # do while number of clusters won't be equal to the amount
    # estimated by a dendrogram
    while number_of_clusters != n:
        tmp_max = -math.inf
        tmp_min = math.inf

        # find two nearest clusters
        # complete linkage method
        for i, clust in enumerate(clusters):
            for j, sec_clust in enumerate(clusters):
                if i != j:
                    # calculate distance between each two points of the two clusters
                    for point in clust:
                        point_x = data.loc[point, 'lng']
                        point_y = data.loc[point, 'lat']
                        for second_point in sec_clust:
                            second_point_x = data.loc[second_point, 'lng']
                            second_point_y = data.loc[second_point, 'lat']
                            tmp_dist = sqrt((point_x - second_point_x) ** 2 + (point_y - second_point_y) ** 2)

                            # choose the longest distance out of all of them
                            if tmp_max < tmp_dist:
                                tmp_max = tmp_dist

                    if tmp_min > tmp_dist:
                        tmp_min = tmp_dist
                        first_clust = i
                        second_clust = j

        # if first cluster has smaller index than second cluster
        # move everything from second cluster to first cluster
        if first_clust < second_clust:
            for point in clusters[second_clust]:
                clusters[first_clust].append(point)
            clusters[second_clust].clear()
            clusters.pop(second_clust)
        # otherwise
        else:
            for point in clusters[first_clust]:
                clusters[second_clust].append(point)
            clusters[first_clust].clear()
            clusters.pop(first_clust)

        number_of_clusters -= 1

    return clusters


def plot_hierarchical(data, country, n):
    clustered = hierarchical_clustering(data, n)
    values = data.values
    color_map = pyplot.cm.get_cmap("hsv", n + 1)

    tupled = []
    for i, cluster in enumerate(clustered):
        for point in cluster:
            tupled.append((point, i))

    tupled.sort()
    index, clusters = list(zip(*tupled))
    methods.shared.plotting.scatter_values(values, clusters, color_map)
    methods.shared.plotting.fill_interpolated_areas(n, clusters, values, color_map)

    pyplot.title("Concentration of cities in " + country + " as clasterized by hierarchical")
    pyplot.xlabel('Longitude')
    pyplot.ylabel('Latitude')
    pyplot.show()
