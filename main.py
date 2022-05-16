from data.worldcities import parse_worldcities
from methods.kmeans import plot_kmeans
from methods.bisecting_kmeans import plot_bisecting_kmeans
from methods.dbscan import plot_dbscan


def main():
    # set country
    country = 'Poland'

    # return lng and lat of cities
    data = parse_worldcities(country)

    # plot data as clasterized by kmeans
    plot_kmeans(data, country, 6)

    # plot data as clasterized by bisecting kmeans
    plot_bisecting_kmeans(data, country, 6)

    # plot data as clasterized by dbscan
    plot_dbscan(data, country, 0.4, 4)


if __name__ == '__main__':
    main()
