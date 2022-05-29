#!/usr/bin/env python
# coding: utf-8

# # Part II - Communicate Data Findings: Ford GoBike System Data
# by (TIMILEYIN AKINTILO)

# ## Investigation Overview
#  
# For the presentation, I will focus on just the relationship between ride distance and ride duration. Since both of them are quantitative variables, we will visualize their relationship on a scattered plot. As a result of the presence of outliers and also to avoid overplotting, an axis limit beteen 0 and 10km was set for the x axis. Jitters and transparancy settings will also be employed to yield the best output.
# 
# Afterwards, I will look into the relationship between member birth year, ride duration and member gender. The relationship between the three variables will be visualized on a scattered plot using the regplot method. Jitters and transparancy settings will also be employed to yield the best output.
# 
# Lastly, I will examine the relationship between member birth year, bikeshare for all trip and ride duration. The relationship between the three variables also will be visualized on a scattered plot using the regplot method. Jitters and transparancy settings will also be employed to yield the best output.
# 
# ## Dataset Overview
# 
# This data set includes information about individual rides made in a bike-sharing system covering the greater San Francisco Bay area. The dataset used for the analysis is available here https://video.udacity-data.com/topher/2020/October/5f91cf38_201902-fordgobike-tripdata/201902-fordgobike-tripdata.csv.

# In[1]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


# In[2]:


# load in the dataset into a pandas dataframe
df = diamonds = pd.read_csv('201902-fordgobike-tripdata1.csv')
df


# In[3]:


# dropping the nulls in the start_station_id, start_station_name, end_station_id and end_station_name columns
df = df[df['start_station_id'].notna()]


# In[4]:


#datatype conversion
df['start_date']=pd.to_datetime(df['start_date'])
df['start_time']=pd.to_datetime(df['start_time'])
df['end_date']=pd.to_datetime(df['end_date'])
df['end_time']=pd.to_datetime(df['end_time'])
df['start_station_id'] = df['start_station_id'].astype(int)
df['end_station_id'] = df['end_station_id'].astype(int)


# In[5]:


#converting the longitude and latitude columns from degree to radian
import math
def deg_to_rad(dr):
    return (dr*math.pi)/180
df['start_station_latitude'] = deg_to_rad(df['start_station_latitude'])
df['end_station_latitude'] = deg_to_rad(df['end_station_latitude'])
df['start_station_longitude'] = deg_to_rad(df['start_station_longitude'])
df['end_station_longitude'] = deg_to_rad(df['end_station_longitude'])


# In[6]:


#calculating diferences
df['diff_lon'] =df['end_station_longitude'] - df['start_station_longitude']
df['diff_lat'] = df['end_station_latitude'] - df['start_station_latitude']


# In[7]:


# Computing the ride distance using Haversine formula
df['a'] = np.sin(df['diff_lat'] / 2)**2 + np.cos(df['start_station_latitude']) * np.cos(df['end_station_latitude']) * np.sin(df['diff_lon'] / 2)**2
df['b']= np.sqrt((df['a']))
df['c'] = 2 * (np.arcsin((df['b'])))
r = 6371
df['distance'] = r * df['c']
df.head(1)


# In[8]:


#drop unwanted columns
df.drop(['start_station_longitude', 'end_station_longitude', 'start_station_latitude', 'end_station_latitude', 'diff_lon', 'diff_lat', 'a', 'b', 'c'], axis = 1, inplace=True)


# In[9]:


#make a copy of the dataset
df1 = df.copy()


# In[10]:


# Removing values <= 4000 as outliers
df2=df1[df1.duration_sec <4000]


# ## (Visualization 1)
# 
# Firstly, I will focus on the relationship between ride distance and ride duration. Since both of them are quantitative variables, we will visualize their relationship on a scattered plot. As a result of the presence of outliers and also to avoid overplotting, an axis limit beteen 0 and 10km was set for the x axis. Jitters and transparancy settings will also be employed to yield the best output.
# 
# 

# In[11]:


sb.regplot(data =df2, x='distance', y = 'duration_sec', x_jitter = 0.1, scatter_kws= {'alpha':1/25});
plt.xlabel('Distance (km)')
plt.title('Investigating the relationship between diatance and ride durartion')
plt.xlim((0,10))
plt.ylabel('Duration_sec(seconds)')


# As expected, there is a positive correlation between distance and duration. The farther the distance, the more the ride duration. Most distance for most rides are below 4km and most rides takes below 2000 seconds

# ## (Visualization 2)
# 
# Now, I will look into the relationship between member birth year, ride duration and member gender. The relationship between the three variables will be visualized on a scattered plot using the regplot method. Jitters and transparancy settings will also be employed to yield the best output.

# In[12]:


gender_markers = [['Male', 'o'],
               ['Female', 's'],
                 ['Others', '^']]

for gender, marker in gender_markers:
    base_color= sb.color_palette()[0]
    plot_data = df1[df1['member_gender'] == gender]
    sb.regplot(data = plot_data, x = 'member_birth_year', y = 'duration_sec', color= base_color, x_jitter = 0.04, fit_reg =False, scatter_kws= {'alpha':1/5}, marker = marker)
plt.legend(['Male','Female', 'Others'])
plt.xlabel('Member birth year')
plt.title('Investigating the relationship between member birth year, member gender and ride durartion')
plt.ylabel('Duration_sec(seconds)')


# As seen on the plot, most of the members are born 1940 and 2000, only few were born before 1940. Also, most individuals under the 'other' category have little ride duration mostly below 20000 seconds. Only males and females are seen to have speed ride duration greater than 20000. Futhermore, those males and females having ride duration greater than 20000 were born between 1960 and 2000. Above 20000 seconds there are more of males than females.

# ## (Visualization 3)
# 
# Lastly, I will examine the relationship between member birth year, bikeshare for all trip and ride duration. The relationship between the three variables also will be visualized on a scattered plot using the regplot method. Jitters and transparancy settings will also be employed to yield the best output.
# 

# In[13]:


trip_markers = [['Yes', 'o'],
               ['No', '^']]

for trip, marker in trip_markers:
    base_color= sb.color_palette()[0]
    plot_data = df1[df1['bike_share_for_all_trip'] == trip]
    sb.regplot(data = plot_data, x = 'member_birth_year', y = 'duration_sec',x_jitter = 0.1, color=base_color , scatter_kws= {'alpha':1/2}, fit_reg =False,  marker = marker)
plt.legend(['Yes','No'])
plt.xlabel('Member birth year')
plt.title('Investigating the relationship between member birth year, member gender and ride durartion')
plt.ylabel('Duration_sec(seconds)')


# As seen on the plot, most of the members are born 1940 and 2000, only few were born before 1940. Also, majority of those who use bikeshare for the whole trip had a ride duration below 20000 seconds, same as those that didn't. A very high ride duration is associated with those who used bike share for the whole journey. Only one individual use other means for the trip and had a ride duration above 80000 seconds 

# In[ ]:


# Use this command if you are running this file in local
get_ipython().system('jupyter nbconvert Part_II_slide_deck_template.ipynb --to slides --post serve --no-input --no-prompt')


# In[ ]:





# In[ ]:




