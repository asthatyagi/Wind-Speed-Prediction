from pandas import read_csv
from datetime import datetime
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv('2011.csv',  parse_dates = [['Year', 'Month', 'Day', 'Hour']], index_col=0, date_parser=parse)
# manually specify column names
del dataset['Dew Point']
del dataset['Wind Direction']

dataset.columns = [ 'GHI','Temperature', 'Pressure', 'Relative Humidity','Wind Speed','Temp']
dataset['Temp']= dataset['Wind Speed']
dataset['Wind Speed']= dataset['GHI']
dataset['Temperature']= dataset['Temp']
dataset.columns = ['Wind Speed', 'Temperature' , 'Pressure','Relative Humidity', 'GHI','Temp']
del dataset['Temp']
dataset.index.name = 'date'
# mark all NA values with 0
#dataset['train'].fillna(0, inplace=True)
# drop the first 24 hours
#dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('train.csv')
