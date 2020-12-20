import numpy as np 
import pandas as pd 
import re

if __name__ == "__main__":
    df = pd.read_csv('../storage/data/raw/us/US_Accidents_June20.csv')
    #The 10 columns that are almost entirely false and thus not useful
    df_FF = df_FF.drop(columns=['Amenity'],axis=1)
    df_FF = df_FF.drop(columns=['Bump'],axis=1)
    df_FF = df_FF.drop(columns=['Stop'],axis=1)
    df_FF = df_FF.drop(columns=['Give_Way'],axis=1)
    df_FF = df_FF.drop(columns=['No_Exit'],axis=1)
    df_FF = df_FF.drop(columns=['Railway'],axis=1)
    df_FF = df_FF.drop(columns=['Roundabout'],axis=1)
    df_FF = df_FF.drop(columns=['Station'],axis=1)
    df_FF = df_FF.drop(columns=['Traffic_Calming'],axis=1)
    df_FF = df_FF.drop(columns=['Turning_Loop'],axis=1)

    #Just a couple of extra columns that do not give us enough information on their own to warrant staying.
    df_FF = df_FF.drop(columns=['Nautical_Twilight'],axis=1) #Closely associated with Sunset, so they will not add to our analysis 
    df_FF = df_FF.drop(columns=['Astronomical_Twilight'],axis=1)
    df_FF = df_FF.drop(columns=['Civil_Twilight'],axis=1)

    #Now to convert the remaining Boolean objects into intiger objects. 
    df_FF['Junction']=df_FF['Junction'].astype(int)
    df_FF['Crossing']=df_FF['Crossing'].astype(int)
    df_FF['Traffic_Signal']=df_FF['Traffic_Signal'].astype(int)
    df_FF = df_FF.drop(columns=['ID'],axis=1) #Superfluous 
    df_FF = df_FF.drop(columns=['Country'],axis=1) #This is a uniform column displaying just the 'US'
    df_FF = df_FF.drop(columns=['State'],axis=1) #This is a uniform column displaying just 'CT' 
    df_FF = df_FF.drop(columns=['County'],axis=1)#This is a uniform column displaying just 'Fairfield'
    df_FF = df_FF.drop(columns=['Timezone'],axis=1) #This is a uniform column displaying just 'US/Eastern'
    df_FF = df_FF.drop(columns=['Airport_Code'],axis=1)#Who cares about airport code, am I right?
    df_FF = df_FF.drop(columns=['Weather_Timestamp'],axis=1)#This relates to what time the weather data was recorded
    df_FF = df_FF.drop(columns=['End_Lat','End_Lng', 'Number'],axis=1) 
    df_FF['Wind_Chill(F)'].fillna(value = df_FF['Wind_Chill(F)'].mean(),inplace = True)
    
    #For these columns we are keeping nan rows by replacing them with mean value
    df_FF['Wind_Speed(mph)'].fillna(value = df_FF['Wind_Speed(mph)'].mean(),inplace = True)
    df_FF['Visibility(mi)'].fillna(df_FF['Visibility(mi)'].mean(),inplace = True)
    df_FF['Precipitation(in)'].fillna(df_FF['Precipitation(in)'].mean(),inplace = True) 
    df_FF['Temperature(F)'].fillna(df_FF['Temperature(F)'].mean(),inplace = True)
    df_FF['Humidity(%)'].fillna(df_FF['Humidity(%)'].mean(),inplace = True)
    df_FF['Pressure(in)'].fillna(df_FF['Pressure(in)'].mean(),inplace = True)
    
    #Converting the giving time data into workable Date-time objects
    df_FF['Start_Time']=pd.to_datetime(df_FF['Start_Time'],infer_datetime_format=True)
    df_FF['End_Time']=pd.to_datetime(df_FF['End_Time'],infer_datetime_format=True)

    #Breaking the start end time into usable independant features 
    df_FF['Year'] = df_FF['Start_Time'].dt.year
    df_FF['Month'] = df_FF['Start_Time'].dt.month
    df_FF['Day'] = df_FF['Start_Time'].dt.day
    df_FF['Time_S'] = df_FF['Start_Time'].dt.hour
    df_FF['Weekday']=df_FF['Start_Time'].dt.weekday
    df_FF['Duration'] = df_FF['End_Time']-df_FF['Start_Time']
    df_FF['Duration'] = df_FF['Duration'].apply(lambda x : x.total_seconds())
    df_FF.drop(columns=['Start_Time','End_Time'],axis=1,inplace=True) # supurfluous now

    df_FF = df_FF[~df_FF['Weather_Condition'].isnull()]
    df_FF = df_FF[~df_FF['Wind_Direction'].isnull()]
    
    #Cleaning Wind Direction column
    df_FF['Wind_Direction'].replace('Calm','CALM',inplace=True)
    df_FF['Wind_Direction'].replace('North','N',inplace=True)
    df_FF['Wind_Direction'].replace('South','S',inplace=True)
    df_FF['Wind_Direction'].replace('East','E',inplace=True)
    df_FF['Wind_Direction'].replace('West','W',inplace=True)
    df_FF['Wind_Direction'].replace('Variable','VAR',inplace=True)
    
    #Binning of time of the day and seasons
    timeBins=[-1,6,12,18,24]
    tBin_names=['Early Morning','Morning','Afternoon','Evening']
    df_FF['TimeofDay']=pd.cut(df_FF['Time_S'],timeBins,labels=tBin_names)
    seasonBins=[-1,2,5,8,11,12]
    sBin_names=['Winter','Spring','Summer','Autumn','Winter']
    df_FF['Season']=pd.cut(df_FF['Month'],seasonBins,labels=sBin_names,ordered=False)
    
    seasonBins=[-1,4,6]
    sBin_names=['Weekday','Weekend']
    df_FF['Day_Type']=pd.cut(df_FF['Weekday'],seasonBins,labels=sBin_names,ordered=False)
    
    df_FF.drop(columns='TMC',inplace=True)
    
    df_FF = df_FF[~(df_FF['Side'] == ' ')]
    
    df_FF['Sunrise_Sunset'].replace('Night',0,inplace=True)
    df_FF['Sunrise_Sunset'].replace('Day',1,inplace=True)

    df_FF['Side'].replace('L',0,inplace=True)
    df_FF['Side'].replace('R',1,inplace=True)
    
    df_FF = df_FF[~(df_FF['Duration'] <= 0 )]
    
    #We further clean wind direction by grouping similar values into one
    df = df_FF
    df.loc[df['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
    df.loc[(df['Wind_Direction']=='West')|(df['Wind_Direction']=='WSW')|(df['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
    df.loc[(df['Wind_Direction']=='South')|(df['Wind_Direction']=='SSW')|(df['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
    df.loc[(df['Wind_Direction']=='North')|(df['Wind_Direction']=='NNW')|(df['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
    df.loc[(df['Wind_Direction']=='East')|(df['Wind_Direction']=='ESE')|(df['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
    df.loc[df['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'
    
    
    weather ='!'.join(df['Weather_Condition'].dropna().unique().tolist())
    df_FF = df
    
    #We clean weather condition by dividing all the values into the below 7 categories
    df['Clear'] = np.where(df['Weather_Condition'].str.contains('Clear', case=False, na = False), 1, 0)
    df['Cloud'] = np.where(df['Weather_Condition'].str.contains('Cloud|Overcast', case=False, na = False), 1, 0)
    df['Rain'] = np.where(df['Weather_Condition'].str.contains('Rain|storm', case=False, na = False), 1, 0)
    df['Heavy_Rain'] = np.where(df['Weather_Condition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms', case=False, na = False), 1, 0)
    df['Snow'] = np.where(df['Weather_Condition'].str.contains('Snow|Sleet|Ice', case=False, na = False), 1, 0)
    df['Heavy_Snow'] = np.where(df['Weather_Condition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls', case=False, na = False), 1, 0)
    df['Fog'] = np.where(df['Weather_Condition'].str.contains('Fog', case=False, na = False), 1, 0)

    # Assign NA to created weather features where 'Weather_Condition' is null.
    weather = ['Clear','Cloud','Rain','Heavy_Rain','Snow','Heavy_Snow','Fog']
    for i in weather:
      df.loc[df['Weather_Condition'].isnull(),i] = df.loc[df['Weather_Condition'].isnull(),'Weather_Condition']

    df.loc[:,['Weather_Condition'] + weather]

    df = df.drop(['Weather_Condition'], axis=1)
    weather = ['Clear','Cloud','Rain','Heavy_Rain','Snow','Heavy_Snow','Fog']
    for w in weather:
        df_FF[w]=df_FF[w].astype(int)
    
    df_FF = df
    featuers_removed=['Description', 'Street','Zipcode','Source','Year', 'Month', 'Day', 
          'Time_S', 'Weekday']

    features=['Severity', 'Distance(mi)',
              'Side','Temperature(F)','Wind_Chill(F)', 'Humidity(%)',
              'Pressure(in)', 'Visibility(mi)','Wind_Direction', 'Wind_Speed(mph)',
              'Precipitation(in)','Clear','Cloud','Rain','Heavy_Rain','Snow','Heavy_Snow','Fog', 'Junction', 'Crossing', 
              'Traffic_Signal','Sunrise_Sunset','TimeofDay', 'Season', 'Day_Type','Duration'] 
    #One-Hot Encoding 
    df_FF_Dummy=pd.get_dummies(df_FF[features],drop_first=True)
    
    df_FF_ML = df_FF_Dummy.reset_index()
    df_FF_ML=df_FF_ML.drop('index',axis=1)
    df_FF_ML.fillna(0,inplace = True)
    
    df = df_FF_ML
    target = 'Severity'
    y=df[target]
    X=df.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = MinMaxScaler()

    X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
    
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    
    X_train.to_csv("traning_data.csv",index = False)
    
    X_test.to_csv("testing_data.csv",index = False)
    
    np.save("y_train.npy",y_train)
    np.save("y_test.npy",y_test)
