from data.worldcities import parse_worldcities
from methods.kmeans import plot_kmeans
from methods.bisecting_kmeans import plot_bisecting_kmeans
from methods.dbscan import plot_dbscan
from methods.hierarchical import plot_hierarchical


def main():
    # set country
    country = 'Belgium'

    # return lng and lat of cities
    data = parse_worldcities(country)

    # plot data as clasterized by kmeans
    plot_kmeans(data, country, 6)

    # plot data as clasterized by bisecting kmeans
    plot_bisecting_kmeans(data, country, 6)

    # plot data as clasterized by dbscan
    plot_dbscan(data, country, 0.15, 8)

    # plot data as clasterized by hierarchical clustering
    plot_hierarchical(data, country, 5)


if __name__ == '__main__':
    main()
