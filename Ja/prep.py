from pandas import read_csv
from datetime import datetime
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('merge.csv',  parse_dates = [['Year', 'Month', 'Day', 'Hour']], index_col=0, date_parser=parse)
# manually specify column names
#del dataset['Dew Point']
#del dataset['Wind Direction']

dataset.columns = [ 'GHI','Dew Point','Temperature', 'Pressure', 'Relative Humidity','Wind Direction','Wind Speed','Temp']
dataset['Temp']= dataset['Wind Speed']
dataset['Wind Speed']= dataset['GHI']
dataset['GHI']= dataset['Temp']
dataset.columns = ['Wind Speed','Dew Point','Temperature',  'Pressure','Relative Humidity','Wind Direction', 'GHI','Temp']
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
