import csv

with open('sample.csv', 'rb') as csv_file:
    data = list(csv.DictReader(csv_file))

keys = ['Name', 'Location']
new_data = [dict((k, d[k]) for k in keys) for d in data]

print new_data
