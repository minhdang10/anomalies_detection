#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#create Snowflake Connection
def createConnection(user_id,passcode):
    import snowflake.connector
    import itertools
    import warnings

    warnings.filterwarnings('ignore')

    print("Attempting to establish a Secure Connection...")
    
    ctx = snowflake.connector.connect(
#         authenticator='externalbrowser',
        user=user_id,
        password=passcode,
        account='pitneybowes.us-east-1',
        warehouse="FUSION_CS_DWH",
        )
    #create a cursor object.
    global cs
    cs = ctx.cursor()
    print("Connected Successfully.")

#query data from Snowflake
def extractData(st,et,query):  
    print('Data Extraction Started')
 
    cs.execute(query)
    data = cs.fetch_pandas_all()

    print('Data Extraction Completed')
    return data
    
    
def exploratory_data_analysis(df):
    print(df.groupby(['CATEGORY_1']).count().sort_values('CATEGORY_2', ascending=False))
    print("---------------------------------")
    print(df.groupby(['CATEGORY_2']).count().sort_values('CATEGORY_1', ascending=False))
    print("---------------------------------")
    print(df.groupby(['CATEGORY_1','CATEGORY_2']).count().sort_values('ACTUAL', ascending=False))
    print("---------------------------------")
    df=df[(df['CATEGORY_1'].isin(['IND3', 'EWR2', 'ATL4', 'ONT1', 'RNO2', 'DFW4', 'CVG1', 'ORD2']))]
#     print("---------------------------------")
#     df=df[(df['CATEGORY_1'].isin(['7', '33', '334', '335']))]
#   print("---------------------------------")
    print(df.groupby(['CATEGORY_1','CATEGORY_2']).count().sort_values('ACTUAL', ascending=False))
    print("---------------------------------")


# In[4]:

def preprocessing(df,window):
    import pandas as pd
    from datetime import datetime, timedelta

    #change data type of EVENT_TIME to datetime
    df['EVENT_TIME']=pd.to_datetime(df['EVENT_TIME'])

    #added component to aggregate data in alignment with prediction windows
    df['hour'] = pd.to_datetime(df['EVENT_TIME']).dt.strftime("%H")
    df['EVENT_TIME2'] = pd.to_datetime(df['EVENT_TIME']).dt.strftime("%Y-%m-%d")
    df['hour2'] = (pd.to_numeric(df['hour']) // int(window[0]) * int(window[0]))
   
    def addhours(row):
        return  pd.to_datetime(row['EVENT_TIME2']) + timedelta(hours=row['hour2'])

    df['EVENT_TIME'] = df.apply(addhours, axis=1)

    #rearrange and reindex data after applying hour aggregation
    df2 = df.groupby(['EVENT_TIME','CATEGORY_1', 'CATEGORY_2']).sum()
    df2 = df2.drop(columns=['hour2'])
    df2 = df2.reset_index()
    df = df2

    return df

#modeling function for backtesting
def modeling_backtesting(category_1,category_2,confidence_interval,category_1_list,category_2_list,df,window,training_period,interval,number_of_forecast):
    import pandas as pd
    from tqdm import tqdm
    from datetime import datetime, timedelta
    from prophet import Prophet
    import numpy as np
    import warnings
    
    warnings.filterwarnings('ignore')

    #empty dataframe to store final results
    Final=pd.DataFrame()

    #looping through list of category_1
    #tqdm is to monitor the runtime
    for cat1 in tqdm(category_1_list):
        print(category_1,cat1)
        
        # filter category_1
        df2=df[df['CATEGORY_1']==cat1]

        #looping through list of category_2
        for cat2 in category_2_list:
            print(category_2,cat2)

            # filter category_2
            pre_train=df2[df2['CATEGORY_2']==cat2]

            if len(pre_train) == 0:
                print('Event is skipped')
                continue

            #drop category_1 and 2 columns
            pre_train.drop(['CATEGORY_2'],axis=1,inplace=True)
            pre_train.drop(['CATEGORY_1'],axis=1,inplace=True) 

            # the n+1 day - the day after as a start date for forecasting
            min_val=(min(pre_train['EVENT_TIME'])+timedelta(days=1)).round('D')
            max_val=max(pre_train['EVENT_TIME'])

            #empty dataframe for EVENT_TIME range, with the specified windows
            temp=pd.DataFrame()
            temp['EVENT_TIME']=pd.Series(pd.date_range(min_val,max_val,freq=window))

            #merge temp dataframe to pre_train, and fill null values with 0
            pre_train=pre_train.merge(temp,how='right',left_on='EVENT_TIME',right_on='EVENT_TIME')
            pre_train=pre_train.fillna(0)

            # specified training period, and sort EVENT_TIME
            len_train=pd.to_datetime(min_val)+timedelta(days=training_period) 
            flag=len_train
            pre_train=pre_train.sort_values(['EVENT_TIME'])

            #looping through the number of days available for testing
            while (flag+timedelta(days=interval))<=pd.to_datetime(max_val):
                #empty dataframe to store Prophet forecast output
                forecast=pd.DataFrame()

                # training set with datetime smaller than flag
                # rename column and convert to datetime format
                train=pre_train[pre_train['EVENT_TIME']<flag]
                train.columns = ['ds', 'y']
                train['ds']= pd.to_datetime(train['ds'])

                # define the model and loop through list of confidence intervals
                for conf_int in confidence_interval:
                    model = Prophet(yearly_seasonality=False, interval_width=conf_int)

                    # fit the model
                    model.fit(train)

                    # future dataframe with the beginning of training period as a start date
                    # with the specified number of forecast, windows and interval
                    future=pd.DataFrame(pd.Series(pd.date_range(start=flag, periods=number_of_forecast*interval,\
                                                                freq=window)),columns=['ds'])

                    #predict the values, and append category_1, category_2, and confidence interval
                    forecast = model.predict(future)
                    forecast['CATEGORY_1']=cat1
                    forecast['CATEGORY_2']=cat2
                    forecast['Sensitivity']=conf_int

                    #append forecast result
                    Final=Final.append(forecast)
                
                #increment flag by specified interval
                flag=flag+timedelta(days=interval)
    return(Final)      

def modeling_production(category_1,category_2,confidence_interval,category_1_list,category_2_list,df,window,training_period,interval,number_of_forecast):
    import pandas as pd
    from tqdm import tqdm
    from datetime import timedelta
    from prophet import Prophet
    import warnings
    
    warnings.filterwarnings('ignore')

    #empty dataframe to store final results
    Final=pd.DataFrame()

    #looping through list of category_1
    #tqdm is to monitor the runtime
    for cat1 in tqdm(category_1_list):
        print(category_1,cat1)

        # filter category_1
        df2=df[df['CATEGORY_1']==cat1]

        #looping through list of category_2
        for cat2 in category_2_list:
            print(category_2,cat2)

            # filter category_2
            pre_train=df2[df2['CATEGORY_2']==cat2]

            if len(pre_train) == 0:
                print('Event is skipped')
                continue
            
            #drop category_1 and 2 columns
            pre_train.drop(['CATEGORY_2'],axis=1,inplace=True)
            pre_train.drop(['CATEGORY_1'],axis=1,inplace=True) 

            # the n+1 day - the day after as a start date for forecasting
            min_val=(min(pre_train['EVENT_TIME'])+timedelta(days=1)).round('D')
            max_val=max(pre_train['EVENT_TIME'])

            #empty dataframe for EVENT_TIME range, with the specified windows
            temp=pd.DataFrame()
            temp['EVENT_TIME']=pd.Series(pd.date_range(min_val,max_val,freq=window))

            #merge temp dataframe to pre_train, and fill null values with 0
            pre_train=pre_train.merge(temp,how='right',left_on='EVENT_TIME',right_on='EVENT_TIME')
            pre_train=pre_train.fillna(0)

            # specified training period, and sort EVENT_TIME
            len_train=pd.to_datetime(min_val)+timedelta(days=training_period) 
            flag=len_train
            pre_train=pre_train.sort_values(['EVENT_TIME'])

            #empty dataframe to store Prophet forecast output
            forecast=pd.DataFrame()

            # training set with datetime smaller than flag
            # rename column and convert to datetime format
            train=pre_train[pre_train['EVENT_TIME']<flag]
            train.columns = ['ds', 'y']
            train['ds']= pd.to_datetime(train['ds'])

            # define the model and loop through list of confidence intervals
            for conf_int in confidence_interval:
                model = Prophet(yearly_seasonality=False, interval_width=conf_int)

                # fit the model
                model.fit(train)

                # future dataframe with the beginning of training period as a start date
                # with the specified number of forecast, windows and interval
                future=pd.DataFrame(pd.Series(pd.date_range(start=flag, periods=number_of_forecast*interval,\
                                                            freq=window)),columns=['ds'])
                
                #predict the values, and append category_1, category_2, and confidence interval
                forecast = model.predict(future)
                forecast['CATEGORY_1']=cat1
                forecast['CATEGORY_2']=cat2
                forecast['Sensitivity']=conf_int

                #append forecast result
                Final=Final.append(forecast)
            
            #increment flag by specified interval
            # flag=flag+timedelta(days=interval)
    return(Final)

def modeling_production_live(category_1,category_2,confidence_interval,category_1_list,category_2_list,df,window,training_period,interval,number_of_forecast):
    import pandas as pd
    from tqdm import tqdm
    from datetime import datetime,timedelta
    from prophet import Prophet
    import warnings
    import pytz
    
    warnings.filterwarnings('ignore')

    #empty dataframe to store final results
    Final=pd.DataFrame()

    #current time's interval
    now_interval = (datetime.now(pytz.timezone('US/Central')).hour)//int(window[0])

    #looping through list of category_1
    #tqdm is to monitor the runtime
    for cat1 in tqdm(category_1_list):
        print(category_1,cat1)

        # filter category_1
        df2=df[df['CATEGORY_1']==cat1]

        #looping through list of category_2
        for cat2 in category_2_list:
            print(category_2,cat2)

            # filter category_2
            pre_train=df2[df2['CATEGORY_2']==cat2]

            if len(pre_train) == 0:
                print('Event is skipped')
                continue
            
            #drop category_1 and 2 columns
            pre_train.drop(['CATEGORY_2'],axis=1,inplace=True)
            pre_train.drop(['CATEGORY_1'],axis=1,inplace=True) 

            # the n+1 day - the day after as a start date for forecasting
            min_val=(min(pre_train['EVENT_TIME'])+timedelta(days=1)).round('D')
            # max_val=max(pre_train['EVENT_TIME'])
            max_val=(datetime.now(pytz.timezone('US/Central')).strftime("%Y-%m-%d %H:%M:%S"))
            #print(max_val)
            
            #empty dataframe for EVENT_TIME range, with the specified windows
            temp=pd.DataFrame()
            temp['EVENT_TIME']=pd.Series(pd.date_range(min_val,max_val,freq=window))

            #merge temp dataframe to pre_train, and fill null values with 0
            pre_train=pre_train.merge(temp,how='right',left_on='EVENT_TIME',right_on='EVENT_TIME')
            pre_train=pre_train.fillna(0)

            # specified training period, and sort EVENT_TIME
            # revert to the day before if run time falls into the first interval (0)
            if now_interval != 0:
                flag=pd.to_datetime(max_val).replace(hour=0, minute=0, second=0)
            else:
                flag=pd.to_datetime(max_val).replace(hour=0, minute=0, second=0)-timedelta(days=1)
            #print(flag)
            
            pre_train=pre_train.sort_values(['EVENT_TIME'])

            #empty dataframe to store Prophet forecast output
            forecast=pd.DataFrame()

            # training set with datetime smaller than flag
            # rename column and convert to datetime format
            train=pre_train[pre_train['EVENT_TIME']<flag]
            train.columns = ['ds', 'y']
            train['ds']= pd.to_datetime(train['ds'])

            # define the model and loop through list of confidence intervals
            for conf_int in confidence_interval:
                model = Prophet(yearly_seasonality=False, interval_width=conf_int)

                # fit the model
                model.fit(train)

                # future dataframe with the beginning of training period as a start date
                # with the specified number of forecast, windows and interval
                future=pd.DataFrame(pd.Series(pd.date_range(start=flag, periods=number_of_forecast*interval,\
                                                            freq=window)),columns=['ds'])
                #print(future)
                #predict the values, and append category_1, category_2, and confidence interval
                forecast = model.predict(future)
                forecast['CATEGORY_1']=cat1
                forecast['CATEGORY_2']=cat2
                forecast['Sensitivity']=conf_int
                forecast['Interval']=[(d.time().hour)//int(window[0]) for d in forecast['ds']]

                #append forecast result
                Final=Final.append(forecast)
            
    if now_interval != 0: 
        # print('yes')
        Final=Final.loc[Final['Interval'] == now_interval-1]
    else:
        # print('no')
        Final=Final.loc[Final['Interval'] == max(Final['Interval'])]

    return(Final)

def output_cleaning(Final,df):
    import numpy as np
    import pandas as pd

    # make negatives values of yhat/lower/upper 0
    Final['yhat']=np.where(Final['yhat']<0,0,Final['yhat'])
    Final['yhat_lower']=np.where(Final['yhat_lower']<0,0,Final['yhat_lower'])
    Final['yhat_upper']=np.where(Final['yhat_upper']<0,0,Final['yhat_upper'])

    # temp dataframe, prepare to merge
    df_t=df
    df_t['EVENT_TIME']=pd.to_datetime(df_t['EVENT_TIME'])

    # merging everything to Final_CSV dataframe
    Final_CSV=Final.merge(df_t,how='left',left_on=['ds','CATEGORY_1','CATEGORY_2'],\
                        right_on=['EVENT_TIME','CATEGORY_1','CATEGORY_2'])

    # fill null values with existing data or 0
    Final_CSV['EVENT_TIME']=np.where(Final_CSV['EVENT_TIME'].isnull(),Final_CSV['ds'],Final_CSV['EVENT_TIME'])
    Final_CSV['CATEGORY_1']=np.where(Final_CSV['CATEGORY_1'].isnull(),Final_CSV['CATEGORY_1'],Final_CSV['CATEGORY_1'])
    Final_CSV['CATEGORY_2']=np.where(Final_CSV['CATEGORY_2'].isnull(),Final_CSV['CATEGORY_2'],Final_CSV['CATEGORY_2'])
    Final_CSV['ACTUAL']=Final_CSV['ACTUAL'].fillna(0)

    # write to CSV
    Final_CSV.to_csv('AWS_Hourly_Production.csv')

    return(Final_CSV)


def formatted_csv(Final_CSV,monitor_name,lob,version):
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import pytz

    Formatted_CSV=Final_CSV

    #Forecast Date/Time
    Formatted_CSV['EVENT_TIME']=pd.to_datetime(Formatted_CSV['EVENT_TIME'])
    Formatted_CSV['Forecast_Date']=[(d.date()) for d in Formatted_CSV['EVENT_TIME']]
    Formatted_CSV['Forecast_Time']=[(d.time()) for d in Formatted_CSV['EVENT_TIME']]

    #Lower/Upper boundary, Actuals, Forecast renames
    old_name=['yhat_lower',
              'yhat_upper',
              'ACTUAL',
              'yhat',
             ]
    new_name=['Lower_Boundary',
              'Upper_Boundary',
              'Actual',
              'Forecast'
             ]
    for old_col, new_col in zip(old_name,new_name):
        Formatted_CSV.rename(columns={old_col:new_col},inplace=True)

    # Model Performance/Sensitivity
    Formatted_CSV['Actual'] = Formatted_CSV['Actual'].astype('float')
    Formatted_CSV['Model_Performance']=round(100-np.sum(np.abs(Formatted_CSV['Actual']-Formatted_CSV['Forecast']))
                                             /np.sum(np.abs(Formatted_CSV['Actual'])),2)

    #Anomaly - Predited and Actual
    conditions = [
        Formatted_CSV['Actual'] < Formatted_CSV['Forecast'],
        Formatted_CSV['Actual'] > Formatted_CSV['Forecast'],
    ]
    choices = ["Lower than expected", 
               "Higher than expected", 
              ]
    Formatted_CSV['Anomaly'] = np.select(conditions,choices,default='')

    #Anomaly - out of prediction interval
    conditions = [
        Formatted_CSV['Actual'] < Formatted_CSV['Lower_Boundary'],
        Formatted_CSV['Actual'] > Formatted_CSV['Upper_Boundary']
    ]
    choices = ["Out of Lower Bound", 
               "Out of Upper Bound", 
               ]
    Formatted_CSV['Anomaly_Interval'] = np.select(conditions, choices,default='')

    #Run date/time
    Formatted_CSV['Run_Date']=(datetime.now(pytz.timezone('US/Central')).date())
    Formatted_CSV['Run_Time']=(datetime.now(pytz.timezone('US/Central')).time())

    #Comment - empty for now
    Formatted_CSV["Comment"] = ""

    #Monitor Name
    Formatted_CSV["Monitor_Name"] = monitor_name
    
    #Line of Business
    Formatted_CSV["LOB"] = lob

    #Version - increment after each update
    Formatted_CSV["Version"] = version
    
    Formatted_CSV.head()

    #Rearranging columns
    cols = Formatted_CSV.columns.tolist()
    cols = ['LOB',
            'Monitor_Name',
            'Version',
            'CATEGORY_1',
            'CATEGORY_2',
            'Run_Date',
            'Run_Time',
            'Forecast_Date',
            'Forecast_Time',
            'Model_Performance',
            'Sensitivity',
            'Lower_Boundary',
            'Upper_Boundary',
            'Forecast',
            'Actual',
            'Anomaly',
            'Anomaly_Interval',
            'Comment'
           ]
    Formatted_CSV=Formatted_CSV[cols]
    Formatted_CSV.to_csv('AWS_Hourly_Formatted_CI_Backtesting_Updatedgenrtest.csv',index=False)
    
    return(Formatted_CSV)      

def alert_rate(confidence_interval,category_1_list,category_2_list,Formatted_CSV,number_of_forecast):
    import pandas as pd

    for conf_int in confidence_interval:
        rate = pd.DataFrame(columns = [cat2 for cat2 in category_2_list])
        average_alert=pd.DataFrame(columns = [cat2 for cat2 in category_2_list])
        for cat1 in category_1_list:
            for cat2 in category_2_list:
                total_oob = len(Formatted_CSV[(Formatted_CSV['Anomaly_Interval'] != '')&\
                                            (Formatted_CSV['CATEGORY_2']==cat2)&\
                                            (Formatted_CSV['CATEGORY_1']==cat1)&\
                                            (Formatted_CSV['Sensitivity'] == conf_int)])
                total_tracking= len(Formatted_CSV[(Formatted_CSV['Sensitivity'] == conf_int)])
                
                alert_rate = round((total_oob/total_tracking)*100,2)
                avg_alert_weekly=alert_rate/100*number_of_forecast*7
                
                rate_tmp= pd.DataFrame([(cat1,alert_rate)], columns=['Category_1',cat2])
                avg_alert_tmp=pd.DataFrame([(cat1,avg_alert_weekly)], columns=['Category_1',cat2])
                
                rate=rate.append(rate_tmp)
                average_alert=average_alert.append(avg_alert_tmp)
                
        rate=rate.fillna(0)
        rate=rate.groupby(['Category_1']).sum()
        rate=rate.reset_index()
        
        average_alert=average_alert.fillna(0)
        average_alert=average_alert.groupby(['Category_1']).sum()
        average_alert=average_alert.reset_index()

        print("Alert Rate", conf_int*100, "%")
        print(rate,'\n')
        
        print("Average Alert Per Week", conf_int*100, "%")
        print(average_alert,'\n')