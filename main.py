from worldcities import parse_worldcities
from kmeans import plot_kmeans
from bisecting_kmeans import plot_bisecting_kmeans
from dbscan import plot_dbscan


def main():
    # set country
    country = 'Poland'

    # return lng and lat of cities
    data = parse_worldcities(country)

    # plot data as clasterized by kmeans
    plot_kmeans(data, country, 2)

    # plot data as clasterized by bisecting kmeans
    # plot_bisecting_kmeans(data, country, 6)

    # plot data as clasterized by dbscan
    plot_dbscan(data, country, 0.4, 4)


if __name__ == '__main__':
    main()
