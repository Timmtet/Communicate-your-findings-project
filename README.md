# (Communicate Data Findings: Ford GoBike System Data)
## by (TIMILEYIN AKINTILO)


## Dataset

This data set includes information about individual rides made in a bike-sharing system covering the greater San Francisco Bay area. The dataset used for the analysis is available here https://video.udacity-data.com/topher/2020/October/5f91cf38_201902-fordgobike-tripdata/201902-fordgobike-tripdata.csv.

The data was cleaned to make is suit for the intended analysis. The following data cleaning operation were carried out:
1. Dealing with nulls:
The start_station_id and start_station_name contained 197 nulls while the member_birth_year and member_gender have 8265 nulls. Because the number of nulls in the former(197) is negligible compared to the whole dataset, we will drop the null. The latter however, has 8265 nulls, which is too much to be termed negligible: we won't drop the nulls.

2. Changing data types
The datatype of start_date, start_time, end_date and end_time columns was converted to datetime datatype. The start_station_id and end_station_id were also converted to int datatype.

3. Calculate diatance from longitude and latitude columns
From the start_station_latitude, start_station_longitude, end_station_latitude and end_station_longitude column,the total ride distance for each entry was computed. The computation process wwas categorised into the folloing steps:

a.Converting the 4 columns from degrees to radain

To compute the diatance, we used the Haversine formula as given below: 
diff_lon = lon2 - lon1

diff_lat = lat2 - lat1

a = sin(diff_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(diff_lon / 2)**2

c = 2 * asin(sqrt(a))

distance = r* c

where r= Radius of earth in kilometers = 6371

b.Calculating the difference in longitudes and latitudes
c.Computing the distance





## Summary of Findings

The following conclusion can be drawn from this analysis

1. More subscibers use bikeshare than customers

2. According to the data, most individuals that use bikesare are between the age of 12 and 72 today

3. More males use bikeshare compare to other gender

4. Most individual that use bikeshare do not use it for the whole trip

5. The more this ditance travelled using bikeshare, the more the ride duration

6. More subscribers have a ride duration of about 500km while that of customers is a little higher than 500km

7. Old members that use bikeshare are associated with lower ride duration. Whike younger members spend more time on thier trip with bikeshare

8. Those who do not use bike share for all the trip spend more time on the trip compare to those who used bikeshare.

9. More customers engage in long diatance rides compared to subscribers

10. Most males and females having ride duration greater than 20000 were born between 1960 and 2000. Overall, of all three gender category,males then to spend more time on trips with bikeshare.

11. A very high ride duration is associated with those who used bike share for the whole journey.



## Key Insights for Presentation

For the presentation, I focus on just the relationship between ride distance and ride duration. Since both of them are quantitative variables, we visualized their relationship on a scattered plot. As a result of the presence of outliers and also to avoid overplotting, an axis limit beteen 0 and 10km was set for the x axis. Jitters and transparancy settings were also employed to yield the best output.
As expected, there is a positive correlation between distance and duration. The farther the distance, the more the ride duration. Most distance for most rides are below 4km and most rides takes below 2000 seconds

Afterwards, I looked into the relationship between member birth year, ride duration and member gender. The relationship between the three variables was visualized on a scattered plot using the regplot method. Jitters and transparancy settings were also employed to yield the best output.
As seen on the plot, most of the members are born 1940 and 2000, only few were born before 1940. Also, most individuals under the 'other' category have little ride duration mostly below 20000 seconds. Only males and females are seen to have speed ride duration greater than 20000. Futhermore, those males and females having ride duration greater than 20000 were born between 1960 and 2000. Above 20000 seconds there are more of males than females.

Lastly, I examined the relationship between member birth year, bikeshare for all trip and ride duration. The relationship between the three variables was also visualized on a scattered plot using the regplot method. Jitters and transparancy settings were also employed to yield the best output.
As seen on the plot, most of the members are born 1940 and 2000, only few were born before 1940. Also, majority of those who use bikeshare for the whole trip had a ride duration below 20000 seconds, same as those that didn't. A very high ride duration is associated with those who used bike share for the whole journey. Only one individual use other means for the trip and had a ride duration above 80000 seconds 








