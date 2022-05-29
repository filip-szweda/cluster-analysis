import math
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    # dendrogram to determine the number of clusters and epsilon of the hierarchical clustering
    # simply uncomment this fragment of code to determine the value of n needed for the particular data
    # plt.figure(figsize=(10, 7))
    # temp = shc.linkage(data, 'single')
    # dend = shc.dendrogram(temp)
    # plt.show()

    clustered = hierarchical_clustering(data, n)

    index = []
    clusters = []
    for i, cluster in enumerate(clustered):
        for point in cluster:
            index.append(point)
            clusters.append(i)

    # make a tuple out of clustered because it's the solution I know how to do
    merged_list = [(index[i], clusters[i]) for i in range(0, len(index))]

    clusters_df = pd.DataFrame(merged_list, columns=["index", "clusters"])

    plt.figure(figsize=(7, 5))

    for clust in np.unique(clusters):
        plt.scatter(data.loc[clusters_df["index"][clusters_df["clusters"] == clust], 'lng'],
                    data.loc[clusters_df["index"][clusters_df["clusters"] == clust], 'lat'], s=17)

    plt.title("Concentration of cities in " + country + " as clasterized by hierarchical clustering")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
