#!/usr/bin/env python
# coding: utf-8

# # Part I - Communicate Data Findings: Ford GoBike System Data
# ## by (TIMILEYIN AKINTILO)
# 
# ## Introduction
# > This dataset includes information about individual rides made in a bike-sharing system covering the greater San Francisco Bay area. The dataset contains 183412 rows and 18 columns.
# 
# 
# 
# ## Preliminary Wrangling
# 

# In[1]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# Now, we load our dataset into the workspace. Note: A little data wrangling was done on the dataset before uplaoding, so as the make it suitable for the desired purpose.

# In[2]:


# load in the dataset into a pandas dataframe, print statistics
df = diamonds = pd.read_csv('201902-fordgobike-tripdata1.csv')
df


# ## Data Accessing
# Now, we will assess our dataset to observe the shape and structure of the dataset

# In[3]:


# high-level overview of data shape and composition
print(df.shape)
print(df.dtypes)
print(df.head(10))


# ## Data cleaning
# Now we clean the dataset to make it fit for analysis
# 1. Dealing with nulls

# Let's examine the dataset for nulls:

# In[4]:


df.isnull().sum()


# Apparently, the start_station_id, start_station_name, end_station_id and end_station_name columns have 197 nulls while the member_birth_year and member_gender have 8265 nulls.
# Because the number of nulls in the former(197) is negligible compared to the whole dataset, we will drop the null. The latter however, has 8265 nulls, which is too much to be termed negligible at this point: we won't drop the nulls

# In[5]:


# dropping the nulls in the former
df = df[df['start_station_id'].notna()]


# In[6]:


df.isnull().sum()


# Now, those nulls have been dropped!!!

# 2. Changing data types
# 
# We will convert the start_date, start_time, end_date and end_time columns to datetime datatype. We will also convert the start_station_id and end_station_id to int datatype

# In[7]:


df['start_date']=pd.to_datetime(df['start_date'])
df['start_time']=pd.to_datetime(df['start_time'])
df['end_date']=pd.to_datetime(df['end_date'])
df['end_time']=pd.to_datetime(df['end_time'])
df['start_station_id'] = df['start_station_id'].astype(int)
df['end_station_id'] = df['end_station_id'].astype(int)


# In[8]:


df.dtypes


# Task completed. Great!!!

# In[9]:


df.head(1)


# 3. Calculate distance from longitude and latitude columns
# 
# From the start_station_latitude, start_station_longitude, end_station_latitude and end_station_longitude column, we will attempt to compute the relative ride distance for each entry. The computation process will be categorised into the folloing steps:
# 
# 1. Converting the 4 columns from degrees to radain
# 2. Calculating the difference in longitudes and latitudes
# 3. Computing the distance
# 

# Step 1: Converting the 4 columns from degrees to radain

# In[10]:


import math
def deg_to_rad(dr):
    return (dr*math.pi)/180
df['start_station_latitude'] = deg_to_rad(df['start_station_latitude'])
df['end_station_latitude'] = deg_to_rad(df['end_station_latitude'])
df['start_station_longitude'] = deg_to_rad(df['start_station_longitude'])
df['end_station_longitude'] = deg_to_rad(df['end_station_longitude'])
    


# In[11]:


df.head(1)


# The 4 columns are now in radian

# Step 2: Calculating the difference in longitudes and latitudes

# In[12]:


df['diff_lon'] =df['end_station_longitude'] - df['start_station_longitude']
df['diff_lat'] = df['end_station_latitude'] - df['start_station_latitude']


# In[13]:


df.head(1)


# Great!!!

# Step 3: Computing the relative distance
# 
# To compute the distance, we use the Haversine formula as given below:
#     diff_lon = lon2 - lon1
#     
#     diff_lat = lat2 - lat1
#     
#     a = sin(diff_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(diff_lon / 2)**2
#     
#     c = 2 * asin(sqrt(a))
#     
#     distance = r* c
#     
#     where r= Radius of earth in kilometers = 6371
#     

# In[14]:


# Computing the ralative ride distance using Haversine formula above
df['a'] = np.sin(df['diff_lat'] / 2)**2 + np.cos(df['start_station_latitude']) * np.cos(df['end_station_latitude']) * np.sin(df['diff_lon'] / 2)**2
df['b']= np.sqrt((df['a']))
df['c'] = 2 * (np.arcsin((df['b'])))
r = 6371
df['distance'] = r * df['c']
df.head(1)


# Fantastic. Now we have our ride distance column

# 4. Drop unwanted columns
# 
# Now we will drop columns that are not crucial to our analysis

# In[15]:


df.drop(['start_station_longitude', 'end_station_longitude', 'start_station_latitude', 'end_station_latitude', 'diff_lon', 'diff_lat', 'a', 'b', 'c'], axis = 1, inplace=True)
df.head()


# In[16]:


df.shape


# Now, we have our desired dataset!!!

# ### What is the structure of your dataset?
# 
# > The dataset has 183215 rows and 15 columns
# 
# ### What is/are the main feature(s) of interest in your dataset?
# 
# > For this anlysis, the main feature of interst in the dataset is the duration of rides (duration_sec)
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# > I expect that the feature that should have the strongest influence on duration of rides is the distance. I aslo expect that other factors such as start and end stations as well as user_type will also influence the duration of rides.

# ## Univariate Exploration
# 
# 

# For this univariate exploration, we are going to investigate the distribution of some individual variables.
# 
# To start with, let us create a copy of our dataset and save it as df1

# In[17]:


df1 = df.copy()


# Dataset copied!

# ## 1. The duration_sec variable
# 
# Being our main feature, we are going to investigate the ride duration variable(duration_sec). We will plot the distribution on an histogram

# QUESTION:What is the average ride duration for bikeshare trips

# In[18]:


df1.describe()


# As observed in the table above, the minimum ride duration is 61 seconds while the maximum ride duration is 85444 seconds. The average ride duration is 514 seconds. Suprisingly, 75% of the enties have ride duration greater or equals to 796 seconds. This suggests the presence of outliers in this column. Let's plot it on an histogram to affirm this hypothesis.

# VISUALIZATION:

# In[19]:


# start with a standard-scaled plot
binsize = 200
bins = np.arange(61, df1['duration_sec'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df1, x = 'duration_sec', bins = bins)
plt.title('Investigating the duration variable')
plt.xlabel('Price ($)')
plt.show()


# Yes. The structure or shape of the histogram affirms the presence of outliers.
# 
# 

# To zoom in into the desired part of the plot, we can make use of axis limit. 

# In[20]:


# Setting the axis limit between 61 and 4000:
binsize = 100
bins = np.arange(61, df1['duration_sec'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df1, x = 'duration_sec', bins = bins)
plt.title('Investigating the duration variable')
plt.xlim((61,4000))
plt.xlabel('Price ($)')
plt.show()


# Now, we have a more relatable plot

# Alternatively, we can remove outliers from our dataset before ploting the distribution on an histogram

# In[21]:


# Removing values <= 4000 as outliers
df2=df1[df1.duration_sec <4000]


# In[22]:


df2.describe()


# Now we plot df2 on an histogram

# In[23]:


# start with a standard-scaled plot
binsize = 100
bins = np.arange(61, df2['duration_sec'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df2, x = 'duration_sec', bins = bins)
plt.xlabel('Duration (sec)')
plt.title('Investigating the duration variable')
plt.show()


# OBSERVATION:

# Great. The plot obtained is similar to the one we got using axis limits

# The histogram plot of duration is Right_skewed. Most of the trips have thier ride duration less than 1000 seconds

# ## 2. Predictor Variable: distance

# QUESTION: What's the average ride distance for bikeshare trips?

# Now, let us examine the distance variable
# 

# In[24]:


df1.distance.describe()


# Now, let us plot the distance vairable on an histogram

# VISUALIZATION:

# In[25]:


binsize = 0.1
bins = np.arange(0, df1['distance'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df1, x = 'distance', bins = bins)
plt.title('Investigating the ride distance')
plt.xlabel('Distance (km)')
plt.show()


# The plot indicates the presence of outliers
# Now, let us make use of axis limit between 0 and 3

# In[26]:


binsize = 0.1
bins = np.arange(0, df1['distance'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df1, x = 'distance', bins = bins)
plt.title('Investigating the ride distance')
plt.xlim((0,3))
plt.xlabel('Distance (km)')
plt.show()


# OBSERVATION:

# This is a much better output. The chart is seemly right-skewed, indicating that more rides have a distance greater than the midpoint (i.e 1.5km). Also it is observe from the chart that some of the distances are zero.

# In[27]:


# Investigating cases where distance is zero
df1[df1.distance==0].head()


# As observed above, the start_station and end_station for these cases are the same. So the total relative distance is truely zero

# ## 3.Predictor Variable: user_type

# QUESTION: Which of the user type is associated with more trips with bikeshare?

# Let us start by examining the user_type column

# In[28]:


df1.user_type.describe()


# In[29]:


df1.groupby(['user_type']).count()


# These infer that we have two user types: Customer and Subscriber

# VISUALIZATION:

# Let us plot the user_type variable on a bar chart

# In[30]:


base_color= sb.color_palette()[0]
user_order = df1['user_type'].value_counts().index
sb.countplot(data=df1, x='user_type', color=base_color, order = user_order);
plt.title('Investigating the user type variable')
plt.xlabel('User_type')
plt.ylabel('Frequency')


# Now, lets see how it looks on a pie chart

# In[31]:


sorted_counts = df1['user_type'].value_counts()

plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90, counterclock = False);
plt.title('Investigating the user type variable')
plt.axis('square')


# ONSERVATION:

# From both plots, it is evident that the subscribers outnumbers customers in this dataset

# ## 4. Predictor variable: member_birth_year

# QUESTION: Which age group is associated with more trips with bikeshare?

# Let's looks at the member_birth_year column

# In[32]:


df1.member_birth_year.describe()


# The highest and the lowest member birth year are 1878 and 2001 respectively

# To be able to effectively plot this variable on an histogram, we need to drop all the nulls in that column

# In[33]:


df3=df1[df1['member_birth_year'].notna()]


# In[34]:


df3.member_birth_year.describe()


# VISUALIZATION:

# We will set the axis limit of the x axis between 1930 and 2005

# In[35]:


binsize = 2
bins = np.arange(1878, df3['member_birth_year'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])
plt.hist(data = df3, x = 'member_birth_year', bins = bins)
plt.title('Investigating the member birth year')
plt.xlabel('Member birth year')
plt.xlim((1930,2005))
plt.show()


# OBSERVATION:

# Cool!!! The distribution is slightly left_skewed. Most of the birth years are are below 1990. Individuals born between 1960 and 2000 use bikeshare more.

# ## 5. Predictor variable: member_gender

# QUESTION: Which gender uses bikeshare the most?

# Let's take a look at the member_gender column

# In[36]:


df1.member_gender.describe()


# In[37]:


df1.groupby(['member_gender']).count()


# VISUALIZATION:

# Let us plot this variable on a bar chart

# In[38]:


base_color= sb.color_palette()[0]
user_order = df1['member_gender'].value_counts().index
sb.countplot(data=df1, x='member_gender', color=base_color, order = user_order);
plt.title('Investigating the member gender variable')
plt.xlabel('Member gender')
plt.ylabel('Frequency')


# Let's examine this same variable on a doughnut plot

# In[39]:


sorted_counts = df1['member_gender'].value_counts()

plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,
        counterclock = False, wedgeprops = {'width' : 0.4});
plt.title('Investigating the member gender variable')
plt.axis('square')


# OBSERVATION:

# Evidently, we have more member who are males(130500) compared to females(40805). We also have another class of gender names others(3647)

# ## Predictor  variable: bike_share_for_all_trip

# QUESTION: Do people use bikeshare for all trip often?

# Let's examine the bike_share_for_all_trip variable

# In[40]:


df1.bike_share_for_all_trip.describe()


# In[41]:


df1.groupby(['bike_share_for_all_trip']).count()


# VISUALIZATION:

# Now, let us visualize the bike_share_for_all_trip on a pie chart

# In[42]:


sorted_counts = df1['bike_share_for_all_trip'].value_counts()

plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90, counterclock = False);
plt.title('Investigating the bike_share_for_all_trip variable')
plt.axis('square')


# OBSERVATION:

# Apparently, most of riders did not utilize bikeshare for the whole trip. 17346 of the riders used bikeshare for the whole trip while 165869 of them used other means alongside bikeshare

# ### Discuss the distribution(s) of your variable(s) of interest. Were there any unusual points? Did you need to perform any transformations?
# 
# > The duration variable took on a large range of values, so I look at the data using an axis limit between 61 and 4000 in other to get the desired outcome. Also, I removed outliers from the dataset before ploting the distribution on an histogram, the result obtained was similar to that obtained using axis limit. The plot obtained is Right_skewed. Most of the trips have thier ride duration less than 1000 seconds
# 
# ### Of the features you investigated, were there any unusual distributions? Did you perform any operations on the data to tidy, adjust, or change the form of the data? If so, why did you do this?
# 
# > When investigating the distance variables, a number of outlier points were identified. Hence, to obtained the required visualization, I made use of axis limit between 0 and 3. This gave a more relatable plot

# ## Bivariate Exploration
# 
# 

# ## 1. Distance vs Duration_sec

# QUESTION: What's the relationship between distance and ride duration

# To start off with, we want to look at the correlations present between distance and duration_sec

# Since both of them are quantitative variables, we will visualize their relationship on a scattered plot.
# 
# As a result of the presence of outliers and also to avoid overplotting, an axis limit beteen 0 and 10km was set for the x axis.
# 
# Jitters and transparancy settings were also employed to yield the best output

# VISUALIZATION:

# In[43]:


sb.regplot(data =df2, x='distance', y = 'duration_sec', x_jitter = 0.1, scatter_kws= {'alpha':1/25});
plt.xlabel('Distance (km)')
plt.title('Investigating the relationship between diatance and ride durartion')
plt.xlim((0,10))
plt.ylabel('Duration_sec(seconds)')


# OBSERVATION:

# As expected, there is a positive correlation between distance and duration. The farther the distance, the more the ride duration. Most distance for most rides are below 4km and most rides takes below 2000 seconds

# ## 2. User_type vs Duration_sec

# QUESTION: What's the relationship between user type and ride duration

# To visualize the relatioship between these quantitative and qualitative variables, we will make use of a violin plot.

# VISUALIZATION:

# In[44]:


base_color = sb.color_palette()[0]
sb.violinplot(data=df2, x='user_type', y='duration_sec', color=base_color, inner=None)
plt.title('Investigating the relationship between user type and ride durartion')


# OBSERVATION:

# As observed from the plot, there are more subscribers than customers. Also, more subscribers have a ride duration of about 500km. At higher durations, there are more customers than subscribers.

# ## 3. Member_birth_year vs Duration

# QUESTION: What's the relationship between member birth year and ride duration?

# Let's take a look the member_birth_year column

# In[45]:


df1.member_birth_year.describe()


# VISUALIZATION:

# Now, we will plot this variable on a scattered plot with jitter and transperency settings

# In[46]:


sb.regplot(data =df1, x='member_birth_year', y = 'duration_sec', x_jitter = 0.1, fit_reg=False);
plt.xlabel('Member birth year')
plt.title('Investigating the relationship between member birth year and ride durartion')
plt.ylabel('Duration_sec(seconds)')


# OBSERVATION:

# The relationship between member's bith year and ride duration is slightly strong and positive. Meaning that older members are associated with lower ride duration. Most members birth year is between 1940 and 2000, and most ride duration is below 20000 seconds

# ## 4. Member_gender vs Duration

# QUESTION: What's the relationship between member gender and ride duration

# Let's examine the member_gender column

# In[47]:


df1.member_gender.describe()


# VISUALIZATION:

# Now, we will visualize the relationship between member's gender and ride duration using a violin plot and a box plot together by the means of subplots 

# In[48]:


plt.figure(figsize = [16, 5])
base_color = sb.color_palette()[0]

plt.subplot(1, 2, 1)
sb.violinplot(data=df2, x='member_gender', y='duration_sec', color=base_color, inner=None)
plt.title('Investigating the relationship between members gender and ride durartion')

plt.subplot(1, 2, 2)
sb.boxplot(data=df2, x='member_gender', y='duration_sec', color=base_color)
plt.title('Investigating the relationship between members gender and ride durartion')


# OBSERVATION:

# The long tail shape observed for all variable depicts the presence of outliers. The shape for the male variable has the longest width, depicting the highest number of entries followed by female ad then others. The box plot also depicts similar observation.

# ## 5. bike_share_for_all_trip vs duration

# QUESTION: What's the relationship between bike_share_for_all_trip and ride duration

# In[49]:


df1.bike_share_for_all_trip.describe()


# VISUALIZATION:

# In[50]:


sb.boxplot(data=df2, x='bike_share_for_all_trip', y='duration_sec', color=base_color)
plt.title('Investigating the relationship between bike_share_for_all variable and ride durartion')


# OBSERVATION:

# From the plot it is evident that those who do not use bike share for all the trip spend more time on the trip compare to those who used bikeshare.

# ## 6. Member_birth_year vs distance

# QUESTION: What's the relationship between distance and member birth year

# Lastly, let us consider the relationship between user type and ride distance

# VISUALIZATION:

# We will plot the relationship on a violin plot

# In[51]:


base_color = sb.color_palette()[0]
sb.violinplot(data=df2, x='user_type', y='distance', color=base_color, inner=None)
plt.title('Investigating the relationship between user type and distance')


# OBSERVATION:
# 

# As seen on the plot, more subscribers have a ride distance below 2km. Neglecting the outliers, more customers engage in long diatance rides compared to subscribers

# ### Talk about some of the relationships you observed in this part of the investigation. How did the feature(s) of interest vary with other features in the dataset?
# 
# As seen in the visualizations above, the relationship between ride duration (duration_sec) and the observed features differs.
# 
# The distance variable has a positive relationship with ride duration as the ride duration increases with increasing ride diatance.
# 
# With respect to the user type, more subscribers uses bikeshare compared customers. Also, more subscribers have a ride duration of about 500km. At higher durations, there are more customers than subscribers.
# 
# With respect to the member's birth year, the relationship between member's bith year and ride duration is slightly strong and positive. Meaning that older members are associated with lower ride duration. Most members birth year is between 1940 and 2000, and most ride duration is below 20000 seconds.
# 
# With respect to the member's gender, more males uses bikeshare, followed by female, then others. The ride duration for the three categories is slightly similar, though, other then to spend a few more seconds on the trip compared to the two other groups
# 
# For the bike_share_for all trips, those who do not use bike share for all the trip spend more time on the trip compare to those who used bikeshare.
# 
# ### Did you observe any interesting relationships between the other features (not the main feature(s) of interest)?
# 
# 
# The relationship between user type and ride distance was looked into. And as seen on the plot, more subscribers have a ride distance below 2km. Negleting the outliers, more customers engage in long distance rides compared to subscribers

# ## Multivariate Exploration
# 
# 

# ## 1. member_birth_year vs member_gender vs durarion sec

# QUESTION: What's the relationship between member birth year, member gender and ride duration?

# Now, let us look at the relationship between member birth year, ride duration and member gender

# VISUALIZATION:

# In[52]:


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


# OBSERVATION:

# As seen on the plot, most of the members were born 1940 and 2000, only few were born before 1940. Also, most individuals under the 'other' category have little ride duration mostly below 20000 seconds. Only males and females are seen to have speed ride duration greater than 20000. Futhermore, those males and females having ride duration greater than 20000 were born between 1960 and 2000. Above 20000 seconds there are more of males than females.

# ## 2. member_birth_year vs bike_share_for_all_trip vs durarion sec

# QUESTION: What's the relationship betweenmember birth year, bike_share_for_all_trip and ride duration?

# Let us now examine the relationship between member birth year, bikeshare for all trip and ride duration

# VISUALIZATION:

# In[53]:


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


# OBSERVATION:

# As seen on the plot, most of the members are born 1940 and 2000, only few were born before 1940. Also, majority of those who use bikeshare for the whole trip had a ride duration below 20000 seconds, same as those that didn't. A very high ride duration is associated with those who used bike share for the whole journey. Only one individual use other means for the trip and had a ride duration above 80000 seconds 

# ### Talk about some of the relationships you observed in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?
# 
# I extended my investigation of ride duration against member birth year in this section by looking at the impact of the two categorical quality features: member's gender and bikeshare for all trips. 
# 
# a. The relationship between ride duration and member birth year was influenced by member's gender as more males born between 1940 and 2000 has high ride duration compared to females and others. 
# 
# b. The relationship between ride duration and member birth year was influenced by whether or not an individual uses bikeshare for the whole trip. It was observed that for members born between 1940 and 2000, those who didn't use bikeshare for the whole trip have higher trip duration than those who used bikeshare. 
# 
# 
# ### Were there any interesting or surprising interactions between features?
# 
# The  result from (a) above implies that irrespective of the distance travelled, the others gender tend to reach the end station faster than females and females faster than males.
# 
# The result from (b) above implies that all things being equal, using bikeshare for once trip helps to save more time

# ## Conclusions
# 
# The following conclusion can be drawn from this analysis
# 
# 1. More subscibers use bikeshare than customers
# 
# 2. According to the data, most individuals that use bikesare are between the age of 12 and 72 today
# 
# 3. More males use bikeshare compare to other gender
# 
# 4. Most individual that use bikeshare do not use it for the whole trip
# 
# 5. The more this ditance travelled using bikeshare, the more the ride duration
# 
# 6. More subscribers have a ride duration of about 500km while that of customers is a little higher than 500km
# 
# 7. Old members that use bikeshare are associated with lower ride duration. Whike younger members spend more time on thier trip with bikeshare
# 
# 8. Those who do not use bike share for all the trip spend more time on the trip compare to those who used bikeshare.
# 
# 9. More customers engage in long diatance rides compared to subscribers
# 
# 10. Most males and females having ride duration greater than 20000 were born between 1960 and 2000. Overall, of all three gender category,males then to spend more time on trips with bikeshare.
# 
# 11. A very high ride duration is associated with those who used bike share for the whole journey.
# 
# 
# 

# In[ ]:




