from worldcities import parse_worldcities
from kmeans import plot_kmeans


def main():
    # set country
    country = 'Poland'

    # return lng and lat of cities
    data = parse_worldcities(country)

    # plot data as clasterized by kmeans
    plot_kmeans(data, country, 6)


if __name__ == '__main__':
    main()
