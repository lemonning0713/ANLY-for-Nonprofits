#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:58:33 2019

@author: mashagubenko
"""

# Import libraries
import pandas as pd
import numpy as np

# Load raw data sets
op_raw = pd.read_excel(open('../data/Raw Data/op data anal - Morona BMAP.xlsx', 'rb'), usecols = "A:H", header = 1, sheet_name = 'subset summary')
op_raw2 = pd.read_excel(open('../data/Raw Data/DATOS OPERACIONALES PAD A 2016 FINAL.xlsx', 'rb'), usecols = "A:J", header = 2, sheet_name = '2016')
bio_raw = pd.read_excel(open('../data/Raw Data/raw data for Andres - Morona BMAP.xlsx', 'rb'), header = 0, sheet_name = 'Composition')
bio_raw2 = pd.read_excel(open('../data/Raw Data/raw data for Andres - Morona BMAP.xlsx', 'rb'),usecols = "A:G", header = 0, sheet_name = 'Environmental')
latlong = pd.read_excel(open('../data/Raw Data/amphibian plot coordinates.xlsx', 'rb'), usecols = "A:E", header = 0, sheet_name = 'Sheet1')

# Import variable translation csv
variable_data = pd.read_csv('../data/Raw Data/Variable_translate.csv', header = None, encoding = 'unicode_escape')    

# Tranfrom csv to dictionary
trans_dict = variable_data.set_index(0).T.to_dict('index')[1]

# Translate column names to English
bio_raw = bio_raw.rename(columns = trans_dict)

#Reforma the date column
bio_raw["Date"] = pd.to_datetime(bio_raw["Date"]).dt.strftime('%Y-%m-%d')

#Fix the excel issue in the date column by making sure all of the dates are within 
#the following phase timelines
# June-July 2014 = EC
# Sept Oct 2014 = LC
# Jan to Feb 2015 = DR
# 13 - 27 Jul 2016 = RI
# 9 - 21 Jul 2016 = AB
temp1 = []
temp2  = []
temp3 = []
for i in range(0,bio_raw.shape[0]):
    temp1.append(int(bio_raw.Date[i][0:4]))
    if int(bio_raw.Date[i][5:7]) in [6,7,9,10,1,2]:
        temp2.append(int(bio_raw.Date[i][5:7]))
        temp3.append(int(bio_raw.Date[i][8:10]))
    else:
        temp2.append(int(bio_raw.Date[i][8:10]))
        temp3.append(int(bio_raw.Date[i][5:7]))
bio_raw['Year'] = temp1
bio_raw['Month'] = temp2
bio_raw['Date2'] = temp3

# Add the columns back into one date column
for i in range(0,bio_raw.shape[0]):
    bio_raw['Date'][i] = str(bio_raw.Year[i]) + '-' + str(bio_raw.Month[i]) +'-' + str(bio_raw.Date2[i])
bio_raw["Date"] = pd.to_datetime(bio_raw["Date"]).dt.strftime('%Y-%m-%d')

# Make all non existent value np.nan
for i in ['Desconocido','desconocido', '-', 'NaN', 'nan', 'desconocida', 'na', 'No tiene', 'no tiene', 'No colectado', 'no', 'no tiene fotos','NO', 'falta el codigo']:
    bio_raw = bio_raw.replace(i,np.nan) #replace inconmplete values with npnan

# Make sure the numeric values have . instead of ,
for col in ['Tail Length']:
    bio_raw[col] = bio_raw[col].str.replace(',','.')

# Clean a few specific irregularities in the dataset
bio_raw = bio_raw.replace("<0.05",0.04)
bio_raw = bio_raw.replace('< 0.05',0.04)
bio_raw = bio_raw.replace('< 0.5',0.4)
bio_raw = bio_raw.replace('<0.5',0.4)
bio_raw = bio_raw.replace('< 0.02',0.01)
bio_raw = bio_raw.replace('<0.02',0.01)
bio_raw = bio_raw.replace('>0.02',0.01) #assume that this is a typo and is supposed to be <
bio_raw = bio_raw.replace('19 L', '19L')

# Make sure the values in the following columns are numeric
cols = ['Phase #', 'Plot', 'Distance', '# Evaluators', 'Long E', 'Lat (N)',
       'Height', 'Location (m)', 'Mass', 'Body Length', 'Tail Length']
for col in cols:
    bio_raw[col] = pd.to_numeric(bio_raw[col])

# Clean the location type column
bio_raw['Location Type'] = bio_raw['Location Type'].str.lower()
bio_raw['Location Type'] = bio_raw['Location Type'].replace('sobre hoja ', 'sobre hoja')
bio_raw['Location Type'] = bio_raw['Location Type'].replace('en hoja ', 'en hoja')

# Clean the behavior column
bio_raw['Behavior'] = bio_raw['Behavior'].str.lower()
bio_raw['Behavior'] = bio_raw['Behavior'].replace('esperando ', 'esperando')

###### Operations Data

# Fix the date column
op_raw["Date"] = pd.to_datetime(op_raw["Date"]).dt.strftime('%Y-%m-%d')

# Check how many dates there are in common between bio and op data
len(set(op_raw['Date']).intersection(set(bio_raw['Date'])))

# Make sure the following columns are numeric
cols2 = ['S_peeps', 'S_hour', 'OPE', 'Rain', 'S_fuel', 'flights']
for col in cols2:
    op_raw[col] = pd.to_numeric(op_raw[col])

# Create a new column of hours per person and cap at 24
op_raw['S_hr/pp'] = op_raw['S_hour']/op_raw['S_peeps']
op_raw['S_hr/pp'][op_raw['S_hr/pp'] > 24] = 24

# Rename columns of the 2016 op data 
cols3 = ['Date','W_Tomorrow', 'W_Late', 'S_peeps', 'OPE','Rain', 'Fuel_D', 'Fuel_S', 'flights','S_hour']
op_raw2.columns = cols3
op_raw2 = op_raw2.drop(op_raw2.index[[297,298]])
op_raw2["Date"] = pd.to_datetime(op_raw2["Date"]).dt.strftime('%Y-%m-%d')

# Replace non existing values with npnan
op_raw2 = op_raw2.replace('NR',np.nan) #replace incomplete values with npnan

# Make the following columns numeric
cols4 = ['S_peeps', 'OPE', 'Rain', 'Fuel_D',
       'Fuel_S', 'flights', 'S_hour']
for col in cols4:
    op_raw2[col] = pd.to_numeric(op_raw2[col])

# Create S_fuel column to match original op dataset
op_raw2['S_fuel'] = op_raw2['Fuel_D'] + op_raw2['Fuel_S']

# Drop redundant columns
op_raw2 = op_raw2.drop(['W_Tomorrow','W_Late','Fuel_D','Fuel_S'], axis = 1)

# Create a new column of hours per person and cap at 24
op_raw2['S_hr/pp'] = op_raw2['S_hour']/op_raw2['S_peeps']
op_raw2['S_hr/pp'][op_raw2['S_hr/pp'] > 24] = 24

# Combine the op data together
opdat = pd.concat([op_raw,op_raw2],ignore_index=True,sort=True)
# Drop the phase column
opdat = opdat.drop(['Phase'],axis=1)
# Merge with the bio data set based on the date column
df = pd.merge(bio_raw, opdat, how='left', on=['Date'])

# Make sure all of the environmental data is numeric
cols5 = ['temp', 'hum', 'luna', 'elev']
for col in cols5:
    bio_raw2[col] = pd.to_numeric(bio_raw2[col])

# Add environmental data to the master dataset where possible by
# distance, plot id and phase 
temp4 = []
temp5 = []
temp6 = []
temp7 = []
for i in range(0,df.shape[0]):
    for j in range(0,bio_raw2.shape[0]):
        if df.Phase[i] == bio_raw2.phase[j]:
            if df.Distance[i] == bio_raw2.dist[j]:
                if df.Plot[i] == bio_raw2['plot_ID'][j]:
                    temp4.append(bio_raw2.temp[j])
                    temp5.append(bio_raw2.hum[j])
                    temp6.append(bio_raw2.luna[j])
                    temp7.append(bio_raw2.elev[j])
df['Temp'] = temp4
df['Hum'] = temp5
df['Luna'] = temp6
df['Elev'] = temp7

temp8 = []
temp9 = []
for i in range(0,df.shape[0]):
    for j in range(0,latlong.shape[0]):
        if df['Plot'][i] == latlong['Plot ID'][j]:
            temp8.append(latlong['UTM North'][j])
            temp9.append(latlong['UTM East'][j])
            
df['Lat'] = temp8
df['Long'] = temp9
df = df.drop(['Long E', 'Lat (N)'],axis=1)

df.to_csv('../data/cleaned data/CleanDatav2.csv', index=False)
#Df is the final clean dataset

