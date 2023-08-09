#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import csv
import geopandas as gpd
import fiona
from shapely.geometry import Polygon, LineString, Point
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from shapely.ops import transform


# In[53]:


def extraction_of_data(trajectorycsv) :
#This fonction converts the information that are in a csv into a dataframe. The input : trajectorycsv is the path to the csv of a video : the trajectories from DataFromSkyViewer. There are 2 outputs : 
#len_video : the length of the video in frames
#output_df : a dataframe which gathers each entity at each frames with information like : TrackID, Time, X, Y, Speed

    output = []
    with open(trajectorycsv) as f:
        reader = csv.reader(f, delimiter=',')
        if 'Trajectory(x' in list(reader)[0][10]:
            indx = 10
        else:
            indx = 8
            
    with open(trajectorycsv) as f:
        reader = csv.reader(f, delimiter=',')
        if indx == 10:
            for row in reader:
                part1 = row[:indx]
                TrackID,Type,Colour,LicencePlate,EntryGate,EntryTime,ExitGate,ExitTime,TraveledDist,AvgSpeed = part1
                part2 = row[indx:]
                big_list = part2
                chunk_size = 6
                while big_list:
                    chunk, big_list = big_list[:chunk_size], big_list[chunk_size:]
                    if len(chunk) == chunk_size:
                        X,Y,Speed,TanAcc,LatAcc,Time = chunk
                        output.append((Time,TrackID,Type,X,Y,Speed))
                        
        if indx == 8:
            for row in reader:
                part1 = row[:indx]
                TrackID,Type,EntryGate,EntryTime,ExitGate,ExitTime,TraveledDist,AvgSpeed = part1
                part2 = row[indx:]
                big_list = part2
                chunk_size = 6
                while big_list:
                    chunk, big_list = big_list[:chunk_size], big_list[chunk_size:]
                    if len(chunk) == chunk_size:
                        X,Y,Speed,TanAcc,LatAcc,Time = chunk
                        output.append((Time,TrackID,Type,X,Y,Speed))
    col_names = ["Time","TrackID","Type","X","Y","Speed"]
    output_df = pd.DataFrame.from_records(output,columns=col_names)
    output_df.drop([0],inplace=True)
    output_df.to_csv('DFS_reformat.csv',index=False)
    output_df = output_df[output_df['Time'] != ' ']
    #There are sometimes blank space : ' ' in the Time column that creates error. We delete those to be sure to have no problems.
    output_df['Time'] = output_df['Time'].astype(float)
    len_video = output_df.Time.max()/0.03333
    #the video is in 30fps so we convert seconds into frames by dividing by (1/30)
    len_video = int(len_video)
    return output_df,len_video                    


# In[54]:


def creation_of_gdf2(output_df) :
    #This fonction filters output_df. It keeps only the information concerning pedestrians, because this is what we care about here.
    #The output gdf2 is a geodataframe with every information of pedestrians in the current video.
    output_df2 = output_df.loc[output_df['Type'] == ' Pedestrian'].copy()
    output_df2['spatially']=0
    output_df2['temporally']=0
    output_df2["geometry"] = output_df2.apply(lambda row: Point(row["Y"], row["X"]), axis=1)
    # The coordinates are (Y,X) here, it comes from the initial csv.
    gdf2 = gpd.GeoDataFrame(output_df2, geometry="geometry")
    return gdf2


# In[55]:


def creation_of_gdf_trajectory(gdf2) :
    #This fonction creates a geodataframe with all the trajectories : 1 line is 1 trajectory and the length of the gdf is equal to the number of pedestrian
    #each trajectory has information about the id of the pedestrian, the length and the linestring : Linestring(Point1,Point2,....,Pointn)
    #The input is gdf2 : a geodataframe which has only information about pedestrian, here 1 line is 1 position
    list_trajectory=[]
    list_length=[]
    list_points =[]
    list_id = []
    for i, row in gdf2.groupby('TrackID'):
        points = [Point(x, y) for x, y in zip(row['Y'], row['X'])]
        trajectory = LineString(points)
        length=trajectory.length
        list_trajectory.append(trajectory)
        list_length.append(length)
        list_points.append(points)
        list_id.append(i)
    
    df_trajectory = pd.DataFrame({'TrackID':list_id, 'length':list_length,'points':list_points})
    gdf_trajectory = gpd.GeoDataFrame(df_trajectory, geometry=list_trajectory)
    gdf_trajectory = gdf_trajectory.sort_values(['TrackID'], ascending=True)
    return gdf_trajectory


# In[56]:


#geomdata = gpd.read_file(intersectionshapefile)


# In[57]:


def check_location(gdf2,geomdata) :
    #This fonction checks every position and attributes to it a location on the map
    #The input are gdf2 (gdf of every pedestrian) and geomdata the shapefile of the intersection, each part of the intersection is a polygon. We check for each point in gdf2 whether the point is in a
    #polygon or not. If the point is outside every polygon, it is in "footpath"
    #The ouput is gdf2 with two new columns : "loc", which is the polygon where the point is inside of and 'loc_type' which is the type of the polygon : "road" ,"cw" or "footpath".
    gdf2['loc'] = 'footpath'
    gdf2['loc_type'] = 'footpath'
    for index, polygon in geomdata.iterrows():
        mask = gdf2.within(polygon['geometry'])
        gdf2.loc[mask, 'loc'] = polygon['name']
        gdf2.loc[mask,'loc_type']= polygon['type']
    return gdf2


# In[76]:


def light_gate_column(geomdata) :
    #This fonction delete the light_gate column in the gdf geomdata. The light gate column indicates for each polygons which gates they are related to in DFS it is something like : "'Gate 1','Gate 2'"
    #The goal here is to have list_list_gate which is a list of each gates related to a polygon : [['Gate 1','Gate 2'],['Gate 4'],['Gate 3','Gate 4']]
    #If a polygon isn't related to a specific gate it has 0 in this column and ['0'] in the list_list_gate
    #The ouput is geomdata with a new column list_gate = list_list_gate
    list_list_gate = []
    for listgate in geomdata['light_gate'].tolist() :
        lst = listgate.split("','")
        list_gate = [element.strip("'") for element in lst]
        list_list_gate.append(list_gate)
        #list_list_gate is like : [[list_gate1],[list_gate2],[list_gate3]...]
    geomdata['list_gate'] = list_list_gate
    geomdata.drop('light_gate', axis=1, inplace=True)
    return geomdata,list_list_gate


# In[59]:


def importation_gate_event(gateeventcsv) :
    #We want here to create a df : gate_df
    #It is composed of every gate event (cars that cross a gate in DFS) on the gates that are concerned here.
    #We filter the gate event to just have the gates that we care for
    #The input is gateeventcsv, the path to the gate event csv from DFS
    #The outputs are gate_df : a filtered gate event and list_list_gate (no changes on it)
    Event_df = pd.read_csv(gateeventcsv)
    Event_df = Event_df.loc[Event_df[' Type'] != ' Motorcycle'].copy()
    Event_df = Event_df.loc[Event_df[' Type'] != ' Pedestrian'].copy()
    Event_df = Event_df.loc[Event_df[' Type'] != ' Bicycle'].copy()
    #We only want cars here (Bicyle are sometimes on the footpath and the road, Motorcycle are often stopped after a gate)
    Event_df['pass'] = 1
    
    gates_list = []
    for listgate in list_list_gate :
        if listgate != ['0'] :
            for i in range(len(listgate)) :
                if listgate[i] not in gates_list :
                    gates_list.append(listgate[i])
                
    gate_df = Event_df[Event_df.Gate.isin(gates_list)]
    return gate_df, list_list_gate


# In[60]:


def list_01_gates(list_list_gate,gate_df,len_video):
    #This function creates a list : list_gate_01
    #This list is a list of lists : each list has the same length as the video and is composed of 0 and 1. Each list is related to a list_gate and when a vehicle crosses the gate we put a 1 in the list.
    #If nothing is passing through the gate we put a 0.
    #When there are several gates in list_gate we add the information of crossing in the same list
    #The inputs are list_list_gate, gate_df to have the information on crossing, len_video to fix the length of each list
    #the outut is list_gate_01
    list_gate_01 = []
    
    for lst in list_list_gate :
        if lst == ['0'] :
            listtoappend = ['0']
        else :
            listgate01 = [0] * len_video
            for gate in lst :
                data = gate_df[gate_df.Gate == gate ][[' Image ID', 'pass']]
                list_x_1 = data[' Image ID'].tolist()
                for i in range(len_video) :
                    for j in range(len(list_x_1)) :
                        if list_x_1[j] == i :
                            listgate01[i] = 1
            listgate01.pop(0)
            listgate01.pop(1)
            listgate01.pop(-1)
            listgate01.pop(-2)
            listtoappend = [0,0] + listgate01 + [0,0]
        list_gate_01.append(listtoappend)
    return list_gate_01


# In[61]:


def len_colour(lst, element):
    #This fonction give the length of the successive elements : we use it to know how long are the phases
    #It takes two inputs : the list we are looking in and element the element we are looking for
    #The output is consecutive_occurences a list of each length of successive elements
    counter = 0
    consecutive_occurrences = []

    for item in lst:
        if item == element:
            counter += 1
        else:
            if counter > 0:
                consecutive_occurrences.append(counter)
            counter = 0

    if counter > 0:
        consecutive_occurrences.append(counter)

    return consecutive_occurrences


# In[62]:


def smooth_data_1(list_gate_01):
    #smooth_data_1 if function that smooth the previous data
    #The input is list_gate_01. For each list in list_gate_01, this code check if there are 2 small signals next to each other. If this is the case, it merges the two small signal into a bigger one.
    #The output is list_gate_01 with merged signal : the lengths of the successive 1 are bigger
    cleanlist01 = []
    for index in range(len(list_gate_01)) :
        lst = list_gate_01[index]
        if lst == ['0'] :
            cleanlist01 = ['0']
        else :
            for a in range(20) :
                list_index_var = []
                listlengreen = []
                listlenred = []
                n  = 0
                m = 3
                for i in range(len(lst)-1) :
                    if lst[i] != lst[i+1] :
                        list_index_var.append(i+1)
                listlengreen = len_colour(lst,1)
                avglengreen = sum(listlengreen) / len(listlengreen)

                for i in range(len(listlengreen)-1) :
                    if listlengreen[i] < 600 :
                        if listlengreen[i+1] < 600 :
                            index1 = list_index_var[n]
                            index2 =list_index_var[m]
                            if list_index_var[m-1]-list_index_var[n+1] < 1000:
                                new_data = [1] * (index2-index1)
                                for k in range(index1,index2 + 1):
                                    lst[k] = new_data[k - index1-1]

                    n+=2
                    m+=2
            cleanlist01 = lst
            list_gate_01[index] = cleanlist01
        
            listlengreen = len_colour(lst,1)
            listlenred = len_colour(lst,0)
    return list_gate_01


# In[63]:


def smooth_data_2(list_gate_01):
    #smooth_data_2 is the 2nd smoothing function. The input is list_gate_01.
    #This function works with the average length of green phases. If a signal is way smaller than the average, we extend it to have a signal that has the same length as the average of the length of green signals
    #The output is list_gate_01, still a list of lists with 0 and 1.
    cleanlist01 = []
    for index in range(len(list_gate_01)) :
        lst = list_gate_01[index]
        if lst == ['0'] :
            cleanlist01 = ['0']
        else :
            listlengreen = len_colour(lst,1)
            listlenred = len_colour(lst,0)
            avglengreen = sum(listlengreen) / len(listlengreen)
            avglenred = sum(listlenred) / len(listlenred)
            list_index_var = []
            for i in range(len(lst)-1) :
                if lst[i] != lst[i+1] :
                    list_index_var.append(i+1)
            n=0
            for i in range(len(listlengreen)) :
                #let's check every length of green phase
                if listlengreen[i] < avglengreen-200 :
                    #the signal is way smaller than the average, we have to make it bigger
                    #there are 3 possible situations : 
                    #1) we have to extend it on the right
                    if listlenred[i] <= avglenred and listlenred[i+1] >= avglenred :
                        redtoreplace = int(listlenred[i+1] - avglenred)
                        index1 = list_index_var[min(n+1,len(list_index_var)-1)]
                        index2 = index1 + redtoreplace
                        if index2 != index1 :
                            new_data = [1] * (index2-index1)
                            for k in range(index1,index2+1):
                                lst[k] = new_data[k - index1-1]
                    if listlenred[i] >= avglenred and listlenred[i+1] <= avglenred :
                        #2) we have to extend it on the left
                        redtoreplace = int(listlenred[i] - avglenred)
                        index2 = list_index_var[min(n,len(list_index_var)-1)]
                        index1 = index2 - redtoreplace
                        if index2 != index1 : 
                            new_data = [1] * (index2-index1)
                            for k in range(index1,index2+1):
                                lst[k] = new_data[k - index1-1]
                    if listlenred[i] >= avglenred and listlenred[i+1] >= avglenred :
                        #3) we have to extend it on both side
                        redtoreplaceleft = int(listlenred[i] - avglenred)
                        redtoreplaceright = int(listlenred[i+1] - avglenred)
                        greentoreplace = int(avglengreen - listlengreen[i])
                        index1r = list_index_var[min(n+1,len(list_index_var)-1)]
                        index2r = index1r + int(greentoreplace/2)
                        index2l = list_index_var[min(n,len(list_index_var)-1)]
                        index1l = index2l - int(greentoreplace/2)
                        if greentoreplace/2 > redtoreplaceleft :
                            index1l = list_index_var[min(n,len(list_index_var)-1)] - redtoreplaceleft

                        if greentoreplace/2 > redtoreplaceright :
                            index2r = list_index_var[min(n,len(list_index_var)-1)] + redtoreplaceright
                        if index1r != index2r :
                            new_data = [1] * (index2r-index1r)
                            for k in range(index1r,index2r+1):
                                lst[k] = new_data[k - index1r-1]
                        if index1l != index2l : 
                            new_data = [1] * (index2l-index1l)
                            for k in range(index1l,index2l+1):
                                lst[k] = new_data[k - index1l-1]
                n+=2
        cleanlist01 = lst
        list_gate_01[index] = cleanlist01
    return list_gate_01


# In[64]:


def avg_pedestrian_RG(list_gate_01,geomdata):
    #This function calculates the average green and red for pedestrians
    #The inputs are list_gate_01 which gathers the information of the cycles and geomdata
    #The output is geomdata with two new columns : avg_ped_G and avg_ped_R
    #if a polygon is not related to any traffic light, we put a 0 in the line of the column.
    list_avg_ped_green = []
    list_avg_ped_red = []
    for lst in list_gate_01 :
        if lst == ['0'] :
            list_avg_ped_green.append(0)
            list_avg_ped_red.append(0)
        else :
            listlengreen = len_colour(lst,0)
            listlenred = len_colour(lst,1)
            avglengreen = sum(listlengreen) / len(listlengreen)
            avglenred = sum(listlenred) / len(listlenred)
            list_avg_ped_green.append(avglengreen/30)
            list_avg_ped_red.append(avglenred/30)
    geomdata['avg_ped_G'] = list_avg_ped_green
    geomdata['avg_ped_R'] = list_avg_ped_red
    return geomdata


# In[65]:


def conversion_01_in_GR(list_gate_01,geomdata):
    #This function converts lists of 0 and 1 into lists of R and G to have a better vision of the cycle when we are looking at the list
    #We will had these information on geomdata that is why we need it in the intput and the output
    #list_gate_01 have the information about the cycles, we need it in the input.
    #geomdata has tw new columns : light which is the list of pedestrian traffic light [R,R,R,R,R,G,G,G,G.....] and index variation, the list of index of each variation of colour
    
    list_gate_GR = []
    listgateGR = []
    for lst in list_gate_01 :
        if lst == ['0'] :
            listgateGR = 0
        else :
            listgateGR = ['R' if i == 1 else 'G' for i in lst]
            #Here R is 1 and G is 0 because we are dealing with pedestrian traffic light it is the opposite of the car traffic light
            listgateGR += 20*listgateGR[-1]
            #We add 20 frames a the end to be sure that the list is long enought and to no have any error (2 or 3 frames should be enought but we are sure to have no errors with 20)
        list_gate_GR.append(listgateGR)
    list_i_phaselight = []
    index_variation = []

    for lst in list_gate_GR :
        if lst != 0 :
            for j in range(len(lst)-1):
                if lst[j] != lst[j+1] :
                    index_variation.append(j+1)
        list_i_phaselight.append(index_variation)
        index_variation = []

    geomdata['light'] = list_gate_GR
    geomdata['index variation'] = list_i_phaselight
    return geomdata


# In[79]:


def determination_of_phase(gdf2,geomdata):
    #determination_of_phase check thanks of the Time information in gdf2 (information of each position) and the phase in geomdata (list of R and G) the exact phase of the polygon the pedestrian is in.
    #There are two inputs gdf2 and geomdata
    #The ouput is gdf2 with two new columns : phase, the current phase of the pedestrian traffic light (R,G or 0 is there is no traffic light) and %_phase, the percentage of the current phase the pedestrian is in
    #gdf2 gets also information in temporally and spatially columns. We have now phase information so we can know about the temporal information.
    list_phase = []
    list_percent_phase = []

    for i in range(len(gdf2)) :
        loc_point = gdf2.iloc[i]['loc']
        if loc_point != 'footpath':
            light_value = geomdata.loc[geomdata['name'] == loc_point, 'light'].values[0]
            if light_value == 0 :
                phase = 0
                percentage = 0
            else :
                #we are in an area which has a traffic light
                time_point = float(gdf2.iloc[i]['Time'])
                index_to_check = int(time_point/0.0333)
                index_variation = geomdata.loc[geomdata['name'] == loc_point, 'index variation'].values[0]
                traffic_light = geomdata.loc[geomdata['name'] == loc_point, 'light'].values[0]
                phase = traffic_light[min(index_to_check,len(traffic_light)-1)]
            
                if index_to_check < index_variation[0]:
                    percentage = index_to_check / index_variation[0] * 100
                else:
                    for j in range(len(index_variation) - 1):
                        if index_to_check > index_variation[j] and index_to_check < index_variation[j + 1]:
                            percentage = ((index_to_check - index_variation[j]) / (index_variation[j + 1] - index_variation[j])) * 100
                            break
            
        else :
            phase = 0
            percentage = 0
        
        list_phase.append(phase)
        list_percent_phase.append(percentage)
    gdf2['%_phase'] = list_percent_phase
    gdf2['phase'] = list_phase
                                       
    gdf2.loc[gdf2['phase'] == 'R','temporally'] = 1
    gdf2.loc[gdf2['loc_type'] == 'road','spatially'] = 1
    return gdf2


# In[67]:


def percentage_of_violation(gdf2,gdf_trajectory):
    #This function calculates the percentage of violation on the whole trajectory, if is superior than 20% we claim that the trajectory can be really studied as a violation
    #The inputs are gdf2 (information on each position) and gdf_trajectory (information on each trajectory)
    #The ouput is gdf_trajectory with four new colums the % of violation temporally and spatially and the Unintended temp/spat behaviour : the trajectory is composed of a big violation
    grouped = gdf2.groupby('TrackID')
    gdf_trajectory['percentage_spatially'] = 0
    gdf_trajectory['percentage_temporally'] = 0

    for track_id, group in grouped:
        len_group = len(group)
        sum_spatially = group['spatially'].sum()
        sum_temporally = group['temporally'].sum()
    
        percent_spatially = sum_spatially / len_group * 100
        #we have the percentage of point on the road
        percent_temporally = sum_temporally / len_group *100
    
        gdf_trajectory.loc[gdf_trajectory['TrackID'] == track_id, 'percentage_spatially'] = percent_spatially
        gdf_trajectory.loc[gdf_trajectory['TrackID'] == track_id, 'percentage_temporally'] = percent_temporally
                                       
    gdf_trajectory['Unintended spat behaviour'] = 0
    gdf_trajectory['Unintended temp behaviour'] = 0

    for index, row in gdf_trajectory.iterrows():
        if row['percentage_spatially'] > 20:
            # Let's set the threshold to 20% for the moment
            gdf_trajectory.at[index, 'Unintended spat behaviour'] = 1
    
        if row['percentage_temporally'] > 20:
            # Let's set the threshold to 20% for the moment
            gdf_trajectory.at[index, 'Unintended temp behaviour'] = 1
    return gdf_trajectory


# In[68]:


def categories(gdf2) :
    #This function creates categories depending on several condition : 1=footpath, 2=crosswalk+Green phase, 3=crosswalk +Red phase, 4=road, 5=crosswalk without traffic light (zebra)
    #This function is not really usefull here, it will be for the analysis of results
    #The input and ouput are gdf2. We had a columns 'category' on the gdf2 with the information inside.
    gdf2['category'] = 0

    gdf2['category'] = np.where(gdf2['loc'] == 'footpath', 1, gdf2['category'])
    gdf2['category'] = np.where((gdf2['loc_type'] == 'cw') & (gdf2['phase'] == 'G'), 2, gdf2['category'])
    gdf2['category'] = np.where((gdf2['loc_type'] == 'cw') & (gdf2['phase'] == 'R'), 3, gdf2['category'])
    gdf2['category'] = np.where(gdf2['loc_type'] == 'road', 4, gdf2['category'])
    gdf2['category'] = np.where((gdf2['loc_type'] == 'cw') & (gdf2['phase'] == 0), 5, gdf2['category'])
    gdf2['Speed'] = gdf2['Speed'].astype(float)
    return gdf2


# In[69]:


def speed_data_polygons(geomdata,gdf2):
    #This function is giving information of speed. Each polygon of geomdata received two nex information the average speed and the stp speed.
    #These results are put in two new columns of geomdata, which is the ouput
    #The inputs are geomdata (we need the list of polygons) and gdf2 (we need the information of speed)
    list_of_polygon = pd.unique(geomdata['name'])
    avgspeed_polygon = [0] * len(list_of_polygon)
    stdspeed_polygon = [0] * len(list_of_polygon)
    grouped = gdf2.groupby('loc')
    
    for i in range(len(list_of_polygon)) :
    
        new_df = gdf2.loc[gdf2['loc'] == list_of_polygon[i]]
        new_df['Speed'] = pd.to_numeric(new_df['Speed'], errors='coerce')
    
        avg_speed = new_df['Speed'].sum() / len(new_df)
        std_speed = new_df['Speed'].std()
        avgspeed_polygon[i] = avg_speed
        stdspeed_polygon[i] = std_speed

    geomdata['avg_speed'] = avgspeed_polygon
    geomdata['std_speed'] = stdspeed_polygon
    return geomdata


# In[70]:


def delete_successive_loc(list_loc):
    #This function is designed for the next cell, it will delete successive element. Thanks to it we can know what is the path of a pedestrian in the intersection
    arr = np.array(list_loc)  # list_loc is converted in numpy
    mask = arr[1:] != arr[:-1]  # mask with the position of loc different from the former one
    new_list = np.concatenate(([arr[0]], arr[1:][mask]))  # get a new list from the mask
    
    return new_list.tolist() 


# In[71]:


def successive_location_per_traj(gdf_trajectory,gdf2):
    #This function give the path of the pedestrian : if we look every successive position of a pedestrian we will have the same information any time because he is moobing frame by frame.
    #This is why we delete the successive location to have something like [footpath, crosswalk 2, footpath]
    #This path is added in gdf_trajectory the output in a new colum 'list_loc'
    #The inputs are gdf_trajectory and gdf2 (has the location of every pedestrian for every frame)
    gdf_trajectory['list_loc'] = 0
    grouped = gdf2.groupby('TrackID')
    list_loc_gdf = []

    for track_id, group in grouped:
        list_loc = delete_successive_loc(group['loc'].tolist())
        list_loc_gdf.append(list_loc)
    
    gdf_trajectory['list_loc'] = list_loc_gdf
    return gdf_trajectory


# In[72]:


def violations_per_polygon(gdf2,geomdata):
    #This function calculates the total of "small violation" per polygon (a "small violation" is a violation of a position : a 1 in the column of spatially or temporally in gdf2)
    #The inputs are gdf2 (to have the small violations) and geomdata (to have the list of polygons)
    #geomdata has two new colums : sum_spatially and sum_temporally, the sum of each small violations, geomdata is the output
    grouped = gdf2.groupby('loc')
    geomdata['sum_spatially'] = 0
    geomdata['sum_temporally'] = 0

    for loc, group in grouped:
        len_group = len(group)
        sum_spatially = group['spatially'].sum()
        sum_temporally = group['temporally'].sum()
    
        geomdata.loc[geomdata['name'] == loc, 'sum_spatially'] = sum_spatially
        geomdata.loc[geomdata['name'] == loc, 'sum_temporally'] = sum_temporally
    
    return geomdata


# In[73]:


def total_trajectory_violation(gdf_trajectory,geomdata,gdf2):
    #This function gives the number of big violations (a big part of the trajectory is a violation) per polygon
    #This information will be in geomdata (list of polygons) in two new columns : total violation spat and total violation temp
    #The inputs are gdf_trajectory (has the information of big violation), geomdata (has the information of polygons) and gdf2 (has the information of localisation and trackID)
    #for each trajectory, the code checks all violation per localisation in gdf2. The one that has the most violation is defined as the localisation where the biggest violation is. This polygon received then
    #the information in geomdata. The problem of this method is that a big violation is related to only one polygons. It can be different from reality but it is a rare case to have somebody who violates
    #two polygons in a big way.
    #The output is geomdata with the new colums
    gdf_trajectory_spat = gdf_trajectory[gdf_trajectory['Unintended spat behaviour'] == 1]
    gdf_trajectory_temp = gdf_trajectory[gdf_trajectory['Unintended temp behaviour'] == 1]
    geomdata['total violation spat'] = 0
    geomdata['total violation temp'] = 0
    for i in range(len(gdf_trajectory_spat)) :
        list_loc_i = gdf_trajectory_spat.iloc[i]['list_loc']
        maximum = 0
        loc_max = 0
        trackid_i = gdf_trajectory_spat.iloc[i]['TrackID']
        for loc in list_loc_i :
            positions = gdf2.loc[(gdf2['loc']==loc) & (gdf2['TrackID']==trackid_i)]
            spat_problem = positions.loc[positions['spatially']==1]
            if len(spat_problem) > maximum :
                maximum = len(spat_problem)
                loc_max = loc
                #loc_max is the loc where the violation happened
        line_geomdata = geomdata.loc[geomdata['name'] == loc_max].index[0]
        geomdata.at[line_geomdata,'total violation spat'] += 1
    
    for i in range(len(gdf_trajectory_temp)) :
        list_loc_i = gdf_trajectory_temp.iloc[i]['list_loc']
        maximum = 0
        loc_max = 0
        trackid_i = gdf_trajectory_temp.iloc[i]['TrackID']
        for loc in list_loc_i :
            positions = gdf2.loc[(gdf2['loc']==loc) & (gdf2['TrackID']==trackid_i)]
            temp_problem = positions.loc[positions['temporally']==1]
            if len(temp_problem) > maximum :
                maximum = len(temp_problem)
                loc_max = loc
                #loc_max is the loc where the violation happened
        line_geomdata = geomdata.loc[geomdata['name'] == loc_max].index[0]
        geomdata.at[line_geomdata,'total violation temp'] += 1
    return geomdata


# In[74]:


#2
trajectorycsv = r'C:\Users\alexf\Dropbox\RSIF Project\Results\2-Broadway-ChippendaleAve\Videos and DFS files\Trajectories/Broadway&ChippendaleAve_Trajectories.csv'
intersectionshapefile = r'C:\Users\alexf\OneDrive\Bureau\stage MSP\video\inter 2 dfs/inter2shapefileB.shp'
gateeventcsv = r'C:\Users\alexf\Dropbox\RSIF Project\Results\2-Broadway-ChippendaleAve\Videos and DFS files\CrossEvents/Broadway&ChippendaleAve_GatesEvent.csv'


# In[80]:


output_df, len_video = extraction_of_data(trajectorycsv)
gdf2 = creation_of_gdf2(output_df)
gdf_trajectory = creation_of_gdf_trajectory(gdf2)
geomdata = gpd.read_file(intersectionshapefile)
gdf2 = check_location(gdf2,geomdata)
geomdata, list_list_gate = light_gate_column(geomdata)
gate_df, list_list_gate = importation_gate_event(gateeventcsv)
list_gate_01 = list_01_gates(list_list_gate,gate_df,len_video)
list_gate_01 = smooth_data_1(list_gate_01)
list_gate_01 = smooth_data_2(list_gate_01)
geomdata = avg_pedestrian_RG(list_gate_01,geomdata)
geomdata = conversion_01_in_GR(list_gate_01,geomdata)
gdf2 = determination_of_phase(gdf2,geomdata)
gdf_trajectory = percentage_of_violation(gdf2,gdf_trajectory)
gdf2 = categories(gdf2)
geomdata = speed_data_polygons(geomdata,gdf2)
gdf_trajectory = successive_location_per_traj(gdf_trajectory,gdf2)
geomdata = violations_per_polygon(gdf2,geomdata)
geomdata = total_trajectory_violation(gdf_trajectory,geomdata,gdf2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




