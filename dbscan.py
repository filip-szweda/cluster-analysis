from random import choice
from numpy import sqrt
from worldcities import parse_worldcities
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def db_scan(data, epsilon, min_points):
    # cluster of the index "0" will contain points labeled as noise
    number_of_clusters = 1
    clusters = []
    type_of_point = 0

    # takes all indexes of given points
    un_pts = list(data.index)

    # temporary new cluster
    curr_clust = set()

    while len(un_pts) != 0:
        # start finding points for a new cluster
        first_point = 1

        # choose random index from the list of unvisited points
        curr_clust.add(choice(un_pts))
        while len(curr_clust) != 0:
            index = curr_clust.pop()

            x = data.loc[index, 'lng']
            y = data.loc[index, 'lat']

            # check if there are any points in the radius
            tmp = data[(sqrt((x - data['lng'].astype(float)) ** 2 + (y - data['lat'].astype(float)) ** 2) <= epsilon) & ~data.index.isin([index])]

            n_indexes = tmp.index
            # values of points:
            # 0 - core point
            # 1 - border point
            # 2 - noise point
            # if the number of points in the radius exceeds min_points then the point is a core point

            if len(tmp) >= min_points:
                type_of_point = 0
            # if the number of points in the radius doesn't exceed min_points, but isn't 0, then it's a border point
            elif min_points > len(tmp) > 0:
                type_of_point = 1
            # if the number of points in the radius equals zero, it is a noise point
            elif len(tmp) == 0:
                type_of_point = 2

            # if the first point is a border point, it's labeled as noise along with its neighbours
            if type_of_point == 1 & first_point == 1:
                clusters.append((index, 0))
                n_indexes = set(un_pts) & set(n_indexes)
                for neigh in n_indexes:
                    clusters.append((neigh, 0))
                un_pts.remove(index)
                un_pts = [e for e in un_pts if e not in n_indexes]
                n_indexes.clear()
                continue

            # remove index from unvisited points
            un_pts.remove(index)

            # select neighbouring indexes that only are unvisited
            n_indexes = set(un_pts) & set(n_indexes)

            # if the element is core
            if type_of_point == 0:
                first_point = 0
                clusters.append((index, number_of_clusters))
                # add neighbours of the core point to be examined next
                curr_clust.update(n_indexes)
                n_indexes.clear()
            # if the element is border
            elif type_of_point == 1:
                clusters.append((index, number_of_clusters))
                n_indexes.clear()
                continue
            # if the element is noise
            elif type_of_point == 2:
                clusters.append((index, 0))
                n_indexes.clear()
                continue

        if not first_point:
            number_of_clusters += 1

    return clusters


def plot_dbscan(data, country, eps, min_pts):
    clustered = db_scan(data, eps, min_pts)
    index, clusters = list(zip(*clustered))
    clusters_df = pd.DataFrame(clustered, columns=["index", "clusters"])

    plt.figure(figsize=(8, 6))

    for clust in np.unique(clusters):
        plt.scatter(data.loc[clusters_df["index"][clusters_df["clusters"] == clust], 'lng'],
                    data.loc[clusters_df["index"][clusters_df["clusters"] == clust], 'lat'], s=17)

    plt.title("Concentration of cities in " + country + " as clasterized by dbscan")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

