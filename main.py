import pandas
from matplotlib import pyplot
import numpy


def kmeans(data, k):
    x = data.values
    diff = 1
    cluster = numpy.zeros(x.shape[0])
    centroids = data.sample(n=k).values
    while diff:
        for i, row in enumerate(x):
            mn_dist = float('inf')
            for idx, centroid in enumerate(centroids):
                d = numpy.sqrt((centroid[0] - row[0]) ** 2 + (centroid[1] - row[1]) ** 2)
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
        new_centroids = pandas.DataFrame(x).groupby(by=cluster).mean().values
        if numpy.count_nonzero(centroids - new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster


def test_kmeans(data, country, k):
    # right now only centroids are shown
    centroids, cluster = kmeans(data, k)

    pyplot.scatter(data.lng, data.lat, color='k')

    for i in range(k):
        pyplot.scatter(centroids[i][1], centroids[i][0], s=200, color='g')

    pyplot.title("Concentration of cities in " + country + " as clasterized by kmeans")
    pyplot.xlabel('Latitude')
    pyplot.ylabel('Longitude')
    pyplot.show()


def main():
    country = 'United States'

    data = pandas.read_csv('worldcities.csv', usecols=['lat', 'lng', 'country'])
    data = data[data['country'] == country]
    data.drop('country', inplace=True, axis=1)

    test_kmeans(data, country, 6)


if __name__ == '__main__':
    main()
