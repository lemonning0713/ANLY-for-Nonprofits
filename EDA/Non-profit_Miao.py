import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

#Data cleaning
raw = pd.read_excel('raw data for Andres - Morona BMAP.xlsx', sheet_name = 'Composition')
raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
raw = raw.sort_values(['Phase #','Date']) #Sort by Phase and Date
raw["SVL (mm)"] = raw["SVL (mm)"].replace(['na'],[0])
#raw.dtypes
raw = raw.drop(['Zona','Cola (mm)', 'Codigo foto','Codigo hisopo','Codigo voucher','Obs adicionales'], axis=1)
#raw

#Choose representative species for visualization
##Find the frequency of animal species in each phase
groups = raw.groupby(['Phase #','Especie']).size().reset_index(name='count') 
groups = groups.sort_values(by = ['Phase #','count'], ascending = [True,False])
#groups

#Choose 5 most popular animal species in each phase
groups_selected = pd.DataFrame()
for i in range(1,6):
    groups_selected = groups_selected.append(groups[groups['Phase #']== i][0:5])
#groups_selected
    
print('Representative animal species are:', groups_selected.Especie.unique())

#Visualization 
## The quantity of each representative species in each phase
canvas = dict(alpha=1)
labels = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
for n in groups_selected.Especie.unique():
    animal = raw[raw["Especie"] == n]
    plt.figure()
    data0 = pd.DataFrame()
    for i in range(1,6):
        data0 = data0.append(animal[animal['Phase #']==i]["SVL (mm)"])
    plt.hist(data0, **canvas,  label= labels)
    plt.gca().set(title='Histogram of SVL(mm) - '+n, ylabel='Quantity')
    plt.xlim(0,50)
    plt.legend()
    plt.show()




