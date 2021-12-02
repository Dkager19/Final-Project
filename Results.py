#!/usr/bin/env python
# coding: utf-8

# In[ ]:


''' Important Questions and bugs to fix include 
1. Try to combine all of the regulated and unregulated df together without the program crashing 
2. append the various storms together by return period and make sure that they are in 1 plot
3. Trim data and create singular function loading in each part of the data 
4. 

final results should have 6 NH storms in 1 figure and 6 SC storms in the other figure (2 figures)
Use monthy median discharge to compare between regulation types over the entire study period instead of 1st figure
Plot both flood frequncy analysis with a function that does plotting and calculations for NH and SC


'''


# In[2]:


'''
Question:How do similar sized regulated and unregulated rivers flood frequency compare between various climates
and water resource development types?




Section 2  Hydrographs for each storm  and mean areal discharge per watershed year 
Read data 
Read available 15min discharge for all four gauges USGS 
Read available 1 hourly  precipitation for both States NOAA
Read longest available  daily discharge values for each site 
Import constants and defined variables
Watershed area in km^2 for all sites
Trim data 
Trim data for each site based on longest period of 15min discharge and 1 hourly precipitation
Eventually trim both dfSC and dfNH by the specific identified storms 
Trim daily discharge data to longest available time period between sites

Creating Dataframes.
Create dataframe named dfSCQ and  dfSCP
Create dataframe labeled dfNHQ and dfNHP
Create dataframe dfstorms_NH and dfstorms_SC 
Create dfNH_median and dfSC_median

Data wrangling 
Convert string values into numerics for the precipitation dataframe 

Linearly interpolate for dfSCQ and dfSCP as well as dfNHQ and dfNHP for short gaps in the data 
Resample and determine frequency of dfmedian to be based on median discharge value on particular day of year throughout the sampling period

Use the watershed area of a site to convert volumetric discharge into a an areal depth for all data frames
Data identification 
Identify the 3 biggest storms for both States within the trimmed dataframe that occur during the snow free season (NA for SC)



Hydrograph Separation
Create a new series named totalq that consists of the discharge data (i.e., hydrograph) for the first storm only.
Determine the antecedent discharge from time of hydrograph rise 
Determine time of peak discharge 
Calculate the event duration N=.82A^.2
Create new series baseq which is copy of totalq to ensure that baseflow equals total flow
Find the best fit straight line for hydrograph recession 
Find the best fit line between the etroplated base flow discharge at peak and the measured total discharge at the end of the event 
Move into plotting function that should work with no base flow data 


Calculations
For each storm event calculate the 
Total water input in cm 
Total discharge in cm
Total event flow in cm 
Antecedent discharge in cm/hr
Maximum precipitation intensity cm/hr
Peak event discharge cm/hr
Duration of water input in days 
Precipitation centroid lag to peak in days 

Calculate average months discharge depth between the regulated and unregulated rivers


Plotting 
Plot dfmedian for all sites with the x label being time and the y label being discharge in areal depth 
Then plot the hydrograph of each storm in a plotting function that has discharge, precipitation and baseflow 

Final Generated plots 
Median discharge depth cm/day for all stream gauges timeseries plot
Hydrograph with precipitation, discharge and baseflow for each storm (lab9 figure2)
Table showing precipitation centroid lag to peak discharge as well as other calculations

'''













# In[1]:





#%% Import libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt

#%% Inputs
#regulated
watershed_a= 657.86e6
#regulated NH 
watershed_km= 657.86
#unregulated NH 
watershed_km2= 396.27
watershed_a2= 396.27e6



#regulated
filenamesR=['NH_regulated15min2005.txt',
            'NH_regulated15min2011.txt',
            'NH_regulated15min2017.txt',
            '/Users/desmond/Downloads/UNH final hydrology data/NH_regulated15min2019.txt']

#regulated
filenamesU=['NH_unregulated15min2005.txt',
            'NH_unregulated15min2011.txt',
            'NH_unregulated15min2017.txt',
            '/Users/desmond/Downloads/UNH final hydrology data/NH_unregulated15min2019.txt']


#loads in discharge data for 2019 regulated water year 
dfQ= pd.read_csv(filenamesR[3], delimiter= '\t', comment= '#',header=1,
                 parse_dates=['20d'], na_values=( -9999, 'T'))

#loads in discharge data for 2019 unregulated water year 
dfQ2= pd.read_csv(filenamesU[3], delimiter= '\t', comment= '#',header=1,
                 parse_dates=['20d'], na_values=( -9999, 'T'))

             

dfQ=dfQ[['20d','14n']]

#renames columns 20d and 14n date and discharge 

dfQ= dfQ.rename (columns={'20d':'DATE'})
dfQ=dfQ.rename(columns={'14n':'discharge'})
dfQ=dfQ.set_index('DATE')

dfQ2=dfQ2[['20d','14n']]

#renames columns 20d and 14n date and discharge 

dfQ2= dfQ2.rename (columns={'20d':'DATE'})
dfQ2=dfQ2.rename(columns={'14n':'discharge'})
dfQ2=dfQ2.set_index('DATE')







for col in dfQ:
    dfQ[col] = pd.to_numeric(dfQ[col], errors='coerce')
    
for col in dfQ2:
    dfQ[col] = pd.to_numeric(dfQ[col], errors='coerce')    


#%%

# Remove NaN's from data frame
dfQ.interpolate(method='linear',inplace = True)
dfQ.fillna(method = 'ffill',inplace=True)
dfQ.fillna(method = 'bfill',inplace=True)

#remove NAN form dataframe
# Remove NaN's from data frame
dfQ2.interpolate(method='linear',inplace = True)
dfQ2.fillna(method = 'ffill',inplace=True)
dfQ2.fillna(method = 'bfill',inplace=True)



#%% areal depth of discharge 


#to cubic feet per second to cubic centimeters per second
dfQ['discharge_cm3']=dfQ['discharge']*28316.85*3600
#mcm^3/ cm^2
dfQ['discharge_cm_hr']= dfQ['discharge_cm3']/(watershed_a*10000)

#to cubic feet per second to cubic centimeters per second
dfQ2['discharge_cm3']=dfQ2['discharge']*28316.85*3600
#mcm^3/ cm^2
dfQ2['discharge_cm_hr']= dfQ2['discharge_cm3']/(watershed_a2*10000)


totalQ_cm= sum(dfQ['discharge_cm_hr']*.25)


print("Total discharge = "+str(round(totalQ_cm,3))+" cm")



totalQ_cm2= sum(dfQ2['discharge_cm_hr']*.25)



print("Total discharge = "+str(round(totalQ_cm2,3))+" cm")



# Determine sampling frequency
freq = dfQ.index.to_series().diff().median()   # timedelta object

# Resample datetime index to ensure regularly spaced data with no missing dates
dfQ= dfQ.resample(freq).median()

# Determine sampling frequency
freq2 = dfQ2.index.to_series().diff().median()   # timedelta object

# Resample datetime index to ensure regularly spaced data with no missing dates
dfQ2= dfQ2.resample(freq2).median()


#plot daily discharge vs peak flow 
fig,(ax)= plt.subplots()
line1= ax.plot(dfQ['discharge_cm_hr'], color= 'k', linestyle= '-',label='Median discharge Regulated')
#add peak flow data to time series 
line2= ax.plot(dfQ2['discharge_cm_hr'], color= 'r', linestyle='-', label= 'Median discharge Unregulated')
#ax.set_xlim(startdate1,enddate1)
#sets horizontal line for average yearly data 
ax.set_ylabel(' cfs', color='k') 
ax.set_title('NH Median discharge ')
lines = line1 + line2
labs = [l.get_label() for l in lines]


ax.legend(lines, labs, loc='center right')

plt.show()

'''
Part 2 of analysis 
'''

filenames= ['/Users/desmond/Downloads/UNH final hydrology data/2006-2012NH.csv','/Users/desmond/Downloads/UNH final hydrology data/2012-2021NH.csv']
#loads in precip data for 2021 water year 
dfP2= pd.read_csv(filenames[0], delimiter= ',',comment= '#',header=0,
                 parse_dates=['DATE'],na_values =( -9999,'T' ))

#loads in precip data for 2021 water year 
dfP= pd.read_csv(filenames[1], delimiter= ',',comment= '#',header=0,
                 parse_dates=['DATE'],na_values =( -9999,'T' ))
#
#loads in discharge data for 2019 regulated water year 
dfQ= pd.read_csv(filenamesR[3], delimiter= '\t', comment= '#',header=1,
                 parse_dates=['20d'], na_values=( -9999, 'T'))

#loads in discharge data for 2019 unregulated water year 
dfQ2= pd.read_csv(filenamesU[3], delimiter= '\t', comment= '#',header=1,
                 parse_dates=['20d'], na_values=( -9999, 'T'))

             
#retains desired columns 
dfP=dfP[['HourlyPrecipitation','DATE']]
dfP2=dfP2[['HourlyPrecipitation','DATE']]


dfQ=dfQ[['20d','14n']]

#renames columns 20d and 14n date and discharge 

dfQ= dfQ.rename (columns={'20d':'DATE'})
dfQ=dfQ.rename(columns={'14n':'discharge'})
dfQ=dfQ.set_index('DATE')

dfQ2=dfQ2[['20d','14n']]

#renames columns 20d and 14n date and discharge 

dfQ2= dfQ2.rename (columns={'20d':'DATE'})
dfQ2=dfQ2.rename(columns={'14n':'discharge'})
dfQ2=dfQ2.set_index('DATE')






#sets index as date 
dfP= dfP.set_index('DATE')
#sets index as date 
dfP2= dfP2.set_index('DATE')




# for loop to convert data types to numerics 
for col in dfP:
    dfP[col] = pd.to_numeric(dfP[col], errors='coerce')
    
for col in dfP2:
    dfP2[col] = pd.to_numeric(dfP2[col], errors='coerce')   

for col in dfQ:
    dfQ[col] = pd.to_numeric(dfQ[col], errors='coerce')
    
for col in dfQ2:
    dfQ[col] = pd.to_numeric(dfQ[col], errors='coerce')    


#%%

# Remove NaN's from data frame
dfQ.interpolate(method='linear',inplace = True)
dfQ.fillna(method = 'ffill',inplace=True)
dfQ.fillna(method = 'bfill',inplace=True)

#remove NAN form dataframe
# Remove NaN's from data frame
dfQ2.interpolate(method='linear',inplace = True)
dfQ2.fillna(method = 'ffill',inplace=True)
dfQ2.fillna(method = 'bfill',inplace=True)

#remove NAN form dataframe
dfP.interpolate(method='linear',inplace = True)
dfP.fillna(method = 'ffill',inplace=True)
dfP.fillna(method = 'bfill',inplace=True)

#remove NAN form dataframe

dfP2.interpolate(method='linear',inplace = True)
dfP2.fillna(method = 'ffill',inplace=True)
dfP2.fillna(method = 'bfill',inplace=True)




'''
df=pd.merge(dfP['HourlyPrecipitation'],dfP2['HourlyPrecipitation'],
            how='right', left_index=True, right_index=True)
'''

#%% areal depth of discharge 
#conver inches to cm
dfP['precip_cm_hr']=dfP['HourlyPrecipitation']*2.54

#to cubic feet per second to cubic centimeters per second
dfQ['discharge_cm3']=dfQ['discharge']*28316.85*3600
#mcm^3/ cm^2
dfQ['discharge_cm_hr']= dfQ['discharge_cm3']/(watershed_a*10000)

#to cubic feet per second to cubic centimeters per second
dfQ2['discharge_cm3']=dfQ2['discharge']*28316.85*3600
#mcm^3/ cm^2
dfQ2['discharge_cm_hr']= dfQ2['discharge_cm3']/(watershed_a2*10000)

#calculates the total precip and discharge 
totalP_cm= sum(dfP['precip_cm_hr'])
totalQ_cm= sum(dfQ['discharge_cm_hr']*.25)


print("Total precipitation = "+str(round(totalP_cm,3))+" cm")
print("Total discharge = "+str(round(totalQ_cm,3))+" cm")


#calculates the total precip and discharge 
totalP_cm= sum(dfP['precip_cm_hr'])
totalQ_cm2= sum(dfQ2['discharge_cm_hr']*.25)


print("Total precipitation = "+str(round(totalP_cm,3))+" cm")
print("Total discharge = "+str(round(totalQ_cm2,3))+" cm")


#%% Plotting function 

#sets start date and end date with timeseriesplot function
startdate= (dt.datetime(2019,4,18))
enddate= (dt.datetime(2019,4,25))



#creates plotting function that does plotting for all timeseries 
def timeseriesplot(dfP,dfQ,dfQ2,startdate,enddate, df3=None):
    
    fig, ax1= plt.subplots()
    #plot1= fig,(ax1)= plot.subplots(1,sharex=True, figsize=(10,10))
    #plot1= fig,(ax1)= plt.subplots(1,sharex=True, figsize=(10,5))

    ax1.plot(dfP['precip_cm_hr'],color= 'b', label= 'Precip',linestyle='-')
  
    ax1.set_ylim(bottom= 0, top=2.5)
    
    ax1.set_ylabel('Precipitation cm/hr', color='b')
    
    ax1.set_ylim(bottom = 0)
    
    ax1.set_xlim(startdate,enddate)

    ax2= ax1.twinx()
    
    ax2.set_ylabel('Discharge (mm/day)', color='r')               # create blue y-axis label
    
    ax2.plot(dfQ['discharge_cm_hr'], color='r', label='dischargeR')
    fig.autofmt_xdate()
    
    ax2.plot(dfQ2['discharge_cm_hr'], color='g', label='dischargeU')
    fig.autofmt_xdate()
    
    ax1.invert_yaxis()
    ax1.set_title('Regulated and unregulated NH ')
    #ax1.legend(loc='best')
    ax2.legend(loc='best')
    # plots timeseries even if baseflow isnt present 
    if df3 is not None:
        ax2.plot(df3, color='orange')



#%% section 2

#sets start and end date for storm 1
storm_start= (dt.datetime(2019,4,18))
storm_end= (dt.datetime(2019,4,25))
#timeseriesplot(dfP,dfQ, startdate= storm_start, enddate=storm_end)

#sets strart date and end date to storm 2
storm_start= (dt.datetime(2019,11,21))
storm_end= (dt.datetime(2019,11,25))
#timeseriesplot(dfP,dfQ,startdate= storm_start, enddate=storm_end)

#sets start date and end date for storm 3
storm_start= (dt.datetime(2019,5,30))
storm_end= (dt.datetime(2019,6,6))
#timeseriesplot(dfP,dfQ, startdate= storm_start, enddate=storm_end)

#creates df storm dataframe from the specfic storm dates 
date= {'storm_start': [dt.datetime(2019,4,18),dt.datetime(2019,11,21),dt.datetime(2019,5,30)],
       'storm_end':[dt.datetime(2019,4,25),dt.datetime(2019,11,25),dt.datetime(2019,6,6)]}
dfstorm= pd.DataFrame(data=date)


for i, v in dfstorm.iterrows():
    storm_start= dfstorm.loc[i,'storm_start']
    storm_end= dfstorm.loc[i,'storm_end']
    timeseriesplot(dfP,dfQ,dfQ2,storm_start,storm_end)

# PLot unregulated 


#%%
#Specify inputs

# List containin names of  daily discharge and annual peak flow
filenames = ['/Users/desmond/Downloads/UNH final hydrology data/peak_unregulatedNH.txt','/Users/desmond/Downloads/UNH final hydrology data/peak_regulatedNH.txt']  
dfpeak2= pd.read_csv(filenames[0], delimiter= '\t',comment= '#',header=1,
                 parse_dates=['10d'])
#impor file
dfpeak= pd.read_csv(filenames[1], delimiter= '\t',comment= '#',header=1,
                 parse_dates=['10d'])

#rename columns dfpeak 
dfpeak=dfpeak.rename(columns={'10d': 'DATE'})
dfpeak=dfpeak.rename(columns={'8s': 'peak_flow'})

#rename columns dfpeak 2
dfpeak2=dfpeak2.rename(columns={'10d': 'DATE'})
dfpeak2=dfpeak2.rename(columns={'8s': 'peak_flow'})

#set index as date 
dfpeak= dfpeak.set_index('DATE')
#drop other columns in peak
dfpeak= dfpeak[['peak_flow']]

#%% accounting for watershed area 
#dfpeak2 unregulated, dfpeak regukated 
'''
#optional trying to account for watershed area 
watershed1= 254
dfpeak['peak_flow']=dfpeak['peak_flow']/watershed1

watershed2= 153
dfpeak2['peak_flow']=dfpeak2['peak_flow']/watershed2
'''
#%%




startdate1= dt.datetime(1957,1,1)
enddate1= dt.datetime(2020,1,1)

dfpeak= dfpeak[startdate1:enddate1]

#set index as date 
dfpeak2= dfpeak2.set_index('DATE')
#drop other columns in peak
dfpeak2= dfpeak2[['peak_flow']]



dfpeak2= dfpeak2[startdate1:enddate1]


#%%

#calculate the mean peak data 
mean= dfpeak.mean()
print('The Mean of annual peak is'+str(mean))
#calculate the std for peak data 

standard_deviation= dfpeak.std()
print(' The standard deviation of the annual peak is '+ str(standard_deviation ))


#sort values by the peak flow
sorted_daily_peak= dfpeak.sort_values(by= 'peak_flow', ascending= False)


#add the 3 highest peak annual discharge values 

print('the largest daily peak flow ' + str(sorted_daily_peak.iloc[0])+ ' on '
      +str((sorted_daily_peak.index[0].strftime('%m/%d/%y'))))
      
print(' the 2nd largest daily peak flow ' + str(sorted_daily_peak.iloc[1])+ ' on '
      +str((sorted_daily_peak.index[1].strftime('%m/%d/%y'))))
      
print(' the 3rd largest daily peak flow ' + str(sorted_daily_peak.iloc[2])+ ' on '
      +str((sorted_daily_peak.index[2].strftime('%m/%d/%y'))))   

#%% Calculating exceedence probablity and return intervals
#sort values by the peak flow
sorted_daily_peak2= dfpeak2.sort_values(by= 'peak_flow', ascending= False)


#create new column rank to determine the highest flows 
sorted_daily_peak2['rank']= sorted_daily_peak2.rank( ascending= False, method= 'first')
#number of years is equal to n
n= len(dfpeak2)
#calculate the exceedence probablity 
sorted_daily_peak2['EP']= sorted_daily_peak2['rank']/(n+1) 
#caluclates the return interval
sorted_daily_peak2['TR']= (1/ sorted_daily_peak2['EP'])
#%%
#create new column rank to determine the highest flows 
sorted_daily_peak['rank']= sorted_daily_peak.rank( ascending= False, method= 'first')
#number of years is equal to n
n= len(dfpeak)
#calculate the exceedence probablity 
sorted_daily_peak['EP']= sorted_daily_peak['rank']/(n+1) 
#caluclates the return interval
sorted_daily_peak['TR']= (1/ sorted_daily_peak['EP'])


#%% find linear trendline for plot2
sorted_daily_peak= sorted_daily_peak.sort_values(by= 'rank', ascending= False)

#create an array to use for dfinterp dataframe                                             
interp= np.array([2,5,10,25,50,100])
#create dfinterp
dfinterp= pd.DataFrame(interp, columns= ['Return Period'])


# sort values by TR 
sorted_daily_peak= sorted_daily_peak.sort_values(by='TR', ascending= True)


#%% find linear trendline for plot2
sorted_daily_peak2= sorted_daily_peak2.sort_values(by= 'rank', ascending= False)

#create an array to use for dfinterp dataframe                                             
interp= np.array([2,5,10,25,50,100])
#create dfinterp
dfinterp2= pd.DataFrame(interp, columns= ['Return Period'])


# sort values by TR 
sorted_daily_peak2= sorted_daily_peak2.sort_values(by='TR', ascending= True)




#%%
# adds interp_values column to df interp

dfinterp['interp_values']= np.interp(dfinterp['Return Period'], sorted_daily_peak['TR'],
                                   sorted_daily_peak['peak_flow'])


#log transformed data  3.1
dfpeak['log10']= np.log10(dfpeak['peak_flow'])
# 3.3 log means 
mean_log= dfpeak['log10'].mean()
print('mean log is' +str(mean_log))
#calculates standard deviation of peak flow log
standard_deviation_log= dfpeak['log10'].std()
print('standard deviation log is' +str(standard_deviation_log))



#%%
# adds interp_values column to df interp

dfinterp2['interp_values']= np.interp(dfinterp2['Return Period'], sorted_daily_peak2['TR'],
                                   sorted_daily_peak2['peak_flow'])


#log transformed data  3.1
dfpeak2['log10']= np.log10(dfpeak2['peak_flow'])
# 3.3 log means 
mean_log2= dfpeak2['log10'].mean()
print('mean log is' +str(mean_log2))
#calculates standard deviation of peak flow log
standard_deviation_log2= dfpeak2['log10'].std()
print('standard deviation log is' +str(standard_deviation_log2))



#%%3.5

#adds the EP to the dfinterp column 
dfinterp['EP']= 1/dfinterp['Return Period']
#calculates the log normal frequncy factor and add kep to interp
dfinterp['Kep']= ((1-dfinterp['EP'])**.135-(dfinterp['EP'])**.135)/.1975

#equation 4 log discharge and add to dfinterp
dfinterp['logQp']= dfpeak['log10'].mean() +dfpeak['log10'].std()*dfinterp['Kep']


#calculated for the dfinterp dataframe
dfinterp['est discharge logQp']=10**(dfinterp['logQp'])

#%%3.5

#adds the EP to the dfinterp column 
dfinterp2['EP']= 1/dfinterp2['Return Period']
#calculates the log normal frequncy factor and add kep to interp
dfinterp2['Kep']= ((1-dfinterp2['EP'])**.135-(dfinterp2['EP'])**.135)/.1975

#equation 4 log discharge and add to dfinterp
dfinterp2['logQp']= dfpeak2['log10'].mean() +dfpeak2['log10'].std()*dfinterp2['Kep']


#calculated for the dfinterp dataframe
dfinterp2['est discharge logQp']=10**(dfinterp2['logQp'])






#%% section 4

# section 4.2 gsx equation 
N=len(dfinterp)
gsx= (N*sum((dfpeak['log10']
                        -dfpeak['log10'].mean())**3)/((N-1)*(N-2)*dfpeak['log10'].mean()**3))
print('gsx='+ str(gsx))

#calculate the  mean square error of station skew 

print('regional skew is .3')
#based on USGS regional skew map 
grx= .3

#equations based on gsx values for mean square error 
a= -.33 +.08+gsx
b= .94 -.26 +gsx

MSEgsx= (10**(a+b)/N**b)
print('MSEgsx='+ str(MSEgsx))
#given mean square error regional skew 
MSEgrx= .302
#the minimum mean square error weighted skew estimate
gx= ((gsx/MSEgsx)+(grx/MSEgrx))/((1/MSEgsx)+(1/MSEgrx))
print('gx=' +str(gx))
#frequency factor kg exceedence probablity
dfinterp['Kg']= (2/gx)*(1+gx*((((1-dfinterp['EP'])**.135)-
                  dfinterp['EP']**.135)/1.185) - ((gx/36)**2))**3 - (2/gx)
#equation 5 log discharge and add to dfinterp
dfinterp['lognew']= dfpeak['log10'].mean() +dfpeak['log10'].std()*dfinterp['Kg']


#calculated for the dfinterp dataframe
dfinterp['lognew']=10**(dfinterp['lognew'])

#%% section 42

# section 4.2 gsx equation 
N2=len(dfinterp2)
gsx2= (N2*sum((dfpeak2['log10']
                        -dfpeak['log10'].mean())**3)/((N2-1)*(N2-2)*dfpeak2['log10'].mean()**3))
print('gsx='+ str(gsx2))

#calculate the  mean square error of station skew 

print('regional skew is .3')
#based on USGS regional skew map 
grx= .3

#equations based on gsx values for mean square error 
a= -.33 +.08+gsx2
b= .94 -.26 +gsx2

MSEgsx2= (10**(a+b)/N2**b)
print('MSEgsx='+ str(MSEgsx2))
#given mean square error regional skew 
MSEgrx2= .302
#the minimum mean square error weighted skew estimate
gx2= ((gsx2/MSEgsx2)+(grx/MSEgrx2))/((1/MSEgsx2)+(1/MSEgrx2))
print('gx=' +str(gx2))
#frequency factor kg exceedence probablity
dfinterp2['Kg']= (2/gx2)*(1+gx2*((((1-dfinterp2['EP'])**.135)-
                  dfinterp2['EP']**.135)/1.185) - ((gx2/36)**2))**3 - (2/gx2)
#equation 5 log discharge and add to dfinterp
dfinterp2['lognew']= dfpeak2['log10'].mean() +dfpeak2['log10'].std()*dfinterp2['Kg']


#calculated for the dfinterp dataframe
dfinterp2['lognew']=10**(dfinterp2['lognew'])

#%%% Plotting and calculations for plot 2

#plot lines 3 and 4 with Weilbull


fig,(ax4)= plt.subplots(1,1)
ax4.set_title('NH Regulated and unregulated Log Pearson III  ')
#plot 2b 
#plots the measured peak discharge values 
ax4.plot(sorted_daily_peak['TR'],sorted_daily_peak['peak_flow'],'o', color= 'k')
                    
ax4.set_xlabel('Return Period years')

# sort values by TR 
sorted_daily_peak= sorted_daily_peak.sort_values(by='TR', ascending= True)
 #find linear trendline of return interval
fit_lineTR= np.polyfit(sorted_daily_peak['TR'],sorted_daily_peak['peak_flow'],1)
ang_coeff_TR= fit_lineTR[0]
intercept_TR= fit_lineTR[1]
# return interval equation 
fit_eq_TR= ang_coeff_TR*sorted_daily_peak['TR']+intercept_TR
#plot linear return period 

# sort values by TR 
sorted_daily_peak= sorted_daily_peak.sort_values(by='TR', ascending= True)

#construct power law trendline 
#power log formula for EP
fit_linep= np.polyfit(np.log10(sorted_daily_peak['EP']),np.log10(sorted_daily_peak['peak_flow']),1)
ang_coeffp= fit_linep[0]
interceptp= fit_linep[1]
#equation for power law EP

dfinterp['logQlog']= ang_coeffp*np.log10(dfinterp['EP'])+interceptp
dfinterp['Qlog']=  10**dfinterp['logQlog']


#plot log pearson 3

ax4.plot(dfinterp['Return Period'], dfinterp['lognew'], color= 'k', label= 'Regulated')

#show labels 

ax4.legend(loc='lower center')


#plots the measured peak discharge values 
ax4.plot(sorted_daily_peak2['TR'],sorted_daily_peak2['peak_flow'],'o', color= 'r')
                    
ax4.set_xlabel('Return Period years')


# sort values by TR 
sorted_daily_peak2= sorted_daily_peak2.sort_values(by='TR', ascending= True)
 #find linear trendline of return interval
fit_lineTR2= np.polyfit(sorted_daily_peak2['TR'],sorted_daily_peak2['peak_flow'],1)
ang_coeff_TR2= fit_lineTR2[0]
intercept_TR2= fit_lineTR2[1]
# return interval equation 
fit_eq_TR2= ang_coeff_TR2*sorted_daily_peak2['TR']+intercept_TR2
#plot linear return period 

# sort values by TR 
sorted_daily_peak2= sorted_daily_peak2.sort_values(by='TR', ascending= True)

#construct power law trendline 
#power log formula for EP
fit_linep2= np.polyfit(np.log10(sorted_daily_peak2['EP']),np.log10(sorted_daily_peak2['peak_flow']),1)
ang_coeffp2= fit_linep2[0]
interceptp2= fit_linep2[1]
#equation for power law EP

dfinterp2['logQlog']= ang_coeffp2*np.log10(dfinterp2['EP'])+interceptp2
dfinterp2['Qlog']=  10**dfinterp2['logQlog']

#plot log pearson 3

ax4.plot(dfinterp2['Return Period'], dfinterp2['lognew'], color= 'r', label= 'Unregulated')

#show labels 

ax4.legend(loc='lower center')

plt.show()




