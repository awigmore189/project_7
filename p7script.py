%matplotlib inline

import pandas as pd
import matplotlib as plt
import numpy as np
import sklearn as sk
from scipy.stats import pearsonr, normaltest
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import metrics
import warnings
# warnings.simplefilter(action = "ignore", category = FutureWarning)

import psycopg2
from sqlalchemy import create_engine

# reading in csv files, pushing them to a local db created on my command line
# cancellations = pd.read_csv('airport_cancellations.csv')
# operations = pd.read_csv('Airport_operations.csv')
# airports = pd.read_csv('airports.csv')

# upload csv files to a local postgres as tables
# engine = create_engine('postgresql://canadasfinest189@localhost:5432/project_7')
# delays.to_sql('cancellations', engine)
# operations.to_sql('operations', engine)
# airports.to_sql('airports', engine)

# using sql query to merge and import data
# data = pd.read_sql_query('''select * FROM operations JOIN cancellations ON
# (operations."airport" = cancellations."Airport" AND operations."year"=cancellations."Year")
# JOIN airports ON operations.airport=airports."LocID";''', engine)

# writing to csv to work offline just in case
# data.to_csv("airports_merged.csv", encoding = 'utf-8', index=False)

#reading in data to save time.
airports = pd.read_csv('airports_merged.csv')
airports


names =['index', 'airport', 'year', 'departures',
       u'arrivals', u'percent on-time gate departures',
       u'percent on-time airport departures', u'percent on-time gate arrivals',
       u'average_gate_departure_delay', u'average_taxi_out_time',
       u'average taxi out delay', u'average airport departure delay',
       u'average airborne delay', u'average taxi in delay',
       u'average block delay', u'average gate arrival delay', u'index.1',
       u'Airport', u'Year', u'Departure Cancellations',
       u'Arrival Cancellations', u'Departure Diversions',
       u'Arrival Diversions', u'index.2', u'Key', u'LocID', u'AP_NAME',
       u'ALIAS', u'facility type', u'faa region', u'county', u'city', u'state',
       u'AP Type', u'latitude', u'longitude', u'boundary data available']

airports.columns = names
drop_list = ['Year','index.2', 'Key', 'LocID', 'boundary data available']
drop_list2 = ['index.1']
airports.drop(drop_list, axis=1, inplace=True)
airports.drop(drop_list2, axis=1, inplace=True)
airports.drop('Airport', axis=1, inplace=True)
airports.drop('index', axis=1, inplace=True)
# eda & some small graphing shit

avg = pd.groupby('airport')
names2 = airports.columns

# encoding stuff, refining stuff, part 4 stuff
features_mask = [u'year',u'departures', u'arrivals',
    u'percent on-time gate departures',u'percent on-time airport departures',u'percent on-time gate arrivals',
    u'average_gate_departure_delay',u'average_taxi_out_time',u'average taxi out delay',u'average airport departure delay',
    u'average airborne delay',u'average taxi in delay', u'average block delay', u'average gate arrival delay',
    u'Departure Cancellations',u'Arrival Cancellations', u'Departure Diversions',u'Arrival Diversions',u'AP_NAME',
    u'ALIAS',u'faa region',u'city',u'state',u'AP Type']

categorical_mask = [u'AP_NAME',u'ALIAS',u'faa region',u'city',u'state',u'AP Type']
numerical_mask = [u'year',u'departures', u'arrivals', u'percent on-time gate departures',
    u'percent on-time airport departures',u'percent on-time gate arrivals',
    u'average_gate_departure_delay',u'average_taxi_out_time',u'average taxi out delay',u'average airport departure delay',
    u'average airborne delay',u'average taxi in delay', u'average block delay', u'average gate arrival delay',
    u'Departure Cancellations',u'Arrival Cancellations', u'Departure Diversions',u'Arrival Diversions']

# encoding categorical variables

from sklearn.preprocessing import LabelEncoder, StandardScaler

for i in categorical_mask:
    le = LabelEncoder()
    x = le.fit_transform(airports[i])
    airports[i] = pd.Series(x)

for i in features_mask:
    scal = StandardScaler()
    x = scal.fit_transform(airports[i])
    airports[i] = pd.Series(x)


# part 5 - defining variables
x = airports[features_mask].values
y = airports['airport'].values

# covarince matrix & Eigen & Eigen Pairs
covMat = np.cov(x.T)
eigenValues, eigenVectors = np.linalg.eig(covMat)


eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
eigenPairs.sort()
eigenPairs.reverse()
for i in eigenPairs:
    print(i[0])

# explained variance
totalEigen = sum(eigenValues)
varExpl = [(i / totalEigen)*100 for i in sorted(eigenValues, reverse=True)]

cvarex = np.cumsum(varExpl)
print(cvarex)


# PCA Transformation
pcask = PCA(n_components=2)
PCA = pcask.fit_transform(x)

Ydf = pd.DataFrame(PCA)
airports3 = airports[['airport', 'year']]
airport_pca = airports3.join(Ydf, on=None, how='left')
airport_pca

graph = airport_pca.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))
for i, airports in enumerate(airports3['airport']):
    graph.annotate(airport, (airport_pca.iloc[i].PC2, airport_pca.iloc[i].PC1))
