from pandas import read_csv
from datetime import datetime
# load data

def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv('merge.csv',  parse_dates = [['Year', 'Month', 'Day', 'Hour']], index_col=0, date_parser=parse)

del dataset['Dew Point']
del dataset['Wind Direction']
del dataset['Solar Radiation (GHI)']

dataset.columns = [ 'Temperature', 'Pressure', 'Relative Humidity','Wind Speed','Temp']
dataset['Temp']= dataset['Wind Speed']
dataset['Wind Speed']= dataset['Temperature']
dataset['Temperature']= dataset['Temp']
dataset.columns = ['Wind Speed',  'Pressure','Relative Humidity', 'Temperature','Temp']
del dataset['Temp']
dataset.index.name = 'date'

print(dataset.head(5))
# save to file
dataset.to_csv('train.csv')
