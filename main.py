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


def test_kmeans(data, country, k):
    centroids, cluster = kmeans(data, k)
    colors = clusters_to_colors(cluster)

    pyplot.scatter(data.lng, data.lat, color=colors)

    for i in range(k):
        pyplot.scatter(centroids[i][1], centroids[i][0], s=200, color='k')

    pyplot.title("Concentration of cities in " + country + " as clasterized by kmeans")
    pyplot.xlabel('Longitude')
    pyplot.ylabel('Latitude')
    pyplot.show()


def main():
    country = 'Poland'

    data = pandas.read_csv('worldcities.csv', usecols=['lat', 'lng', 'country'])
    data = data[data['country'] == country]
    data.drop('country', inplace=True, axis=1)

    test_kmeans(data, country, 6)


if __name__ == '__main__':
    main()
