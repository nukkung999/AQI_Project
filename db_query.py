import datacube
from datacube.utils.masking import mask_invalid_data
import numpy as np
import pandas as pd
import joblib

# For this program we use Open Datacube (ODC) as store procedure and query API

query = {
    'time': ('2020-02-03', '2020-02-03'), # Format (<From (YYYY-MM-DD)>, <To (YYYY-MM-DD)>)
	'lat': (13.39822, 15.50416), # Lat (Bottom, Top)
	'lon': (99.70185, 101.819404), # Long (Left, Right)
}

def stringDOYTran(dateString):
	year = int(dateString.split('-')[0])
	month = int(dateString.split('-')[1])
	day = int(dateString.split('-')[2])
	DOY = 0

	for i in range(1, month):
		if i == 2:
			if isLeapYear(year):
				DOY += 29
			else:
				DOY += 28
		elif i <= 7:
			if i%2 == 0:
				DOY += 30
			else:
				DOY += 31
		else:
			if i%2 == 0:
				DOY += 31
			else:
				DOY += 30
	
	DOY += day
	return DOY

def isLeapYear(year):
	if year % 400 == 0:
		return True
	elif year % 100 == 0:
		return False
	elif year % 4 == 0:
		return True
	else:
		return False

dc = datacube.Datacube()
data = dc.load(product='ls8_lasrc_new', measurements=['coastal_aerosol', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa'], **query) 
# Product name relate your ingested product name
data = mask_invalid_data(data) #mask null value

date = str(data.time[0].data)
date = date.split('T')[0]
doy = stringDOYTran(date)

data = data.to_array(dim='color')
data = data.transpose(*(data.dims[1:]+data.dims[:1]))
data = data.to_masked_array()
data /= 10000
row = data.shape[1]
col = data.shape[2]

dat = data[0].copy()
dat = dat.reshape(row*col, data.shape[3])

col_name = ['coastal_aerosol', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']
df = pd.DataFrame(dat, columns = col_name)
li_doy = np.zeros(df.shape[0]).astype(int)
li_doy += doy
df["DOY"] = li_doy
joblib.dump(df, ".\\12905020200203_data.df") # File name up to you
print("Dump Data Complete!!!")