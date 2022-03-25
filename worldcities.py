import pandas


def parse_worldcities(country):
    data = pandas.read_csv('worldcities.csv', usecols=['lat', 'lng', 'country'])
    data = data[data['country'] == country]
    data.drop('country', inplace=True, axis=1)
    return data
