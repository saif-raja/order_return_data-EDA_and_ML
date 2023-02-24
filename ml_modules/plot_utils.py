import numpy as np
import pandas as pd 

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()



def treatoutliers(df_ori, columns=None, exclusion_fraction = 0.025 , treament='cap'): 
    ## Treat Outliers using InterQuartile Ranges
    ### RobustSclaing like method

    #### Quartile method used instead of standard deviation
    #### ie: Medians instead of Means 

    #mod code = only qualtile based exclusio of a fixed number of samples 
    # right now you can do the quantile clipping  multiple times and analyse how the original and geenrated DFs are differnet
    # the above code can quite easily be modified to be memory efficient , incaseyou are maxing out your RAM, 
    #     just replace df_pri with df , remove the copy line , and the original DF will be acted upon 
    #oofcourse , eachiteration will be final , and the calculation if done again , will be done upon this modified DF , 
    # so you have only one chance to do it , but offcourse , low RAM needs
    
    df = df_ori.copy()
    #del df_ori
    
    
    
    if not columns:
        columns = df.columns
    
    for column in columns:
        treatoutliers.floor = df[column].quantile(    exclusion_fraction/2 )
        treatoutliers.ceil  = df[column].quantile(1- (exclusion_fraction/2))
        
        
        
        if treament == 'remove':
            df = df[(df[column] >= floor) & (df[column] <= ceil)]
        elif treament == 'cap':
            df[column] = df[column].clip(treatoutliers.floor, treatoutliers.ceil)
            
            df.loc[df[column] ==  np.inf , column ] = treatoutliers.ceil
            df.loc[df[column] == -np.inf , column ] = treatoutliers.floor

    return df




def time_plots (time_level , df) :

        ax = df.groupby(time_level)['Order.ID'].count().plot(kind='bar', figsize=(16,8))
        ax.set_xlabel(time_level,fontsize=15)
        ax.set_ylabel('Number of Orders',fontsize=15)
        ax.set_title('Orders over '+time_level ,fontsize=15)
        plt.xticks(rotation=45)
        plt.show()


        ###############


        ax = df.groupby([time_level,'Return.Status'])['Order.ID'].count() \
                .unstack()\
                .plot.bar( stacked=True , figsize=(16,8))
                
        ax.set_xlabel(time_level,fontsize=15)
        ax.set_ylabel('Number of Orders',fontsize=15)
        ax.set_title('Orders over '+time_level+' by Return Status' ,fontsize=15)
        plt.xticks(rotation=45)
        plt.show()



        temp_df = df[[time_level,'Return.Status','Order.ID']].astype({time_level:'str'})#.sort_values( [time_level,'Return.Status'] )

        # sns.set(rc = {'figure.figsize':(20,10)})
        # ax = sns.histplot(temp_df , x=time_level , hue='Return.Status' , multiple="stack" , discrete=True)
        # ax.set_xlabel(time_level,fontsize=15)
        # ax.set_ylabel('Number of Orders',fontsize=15)
        # ax.set_title('Orders over '+time_level+' by Return Status' ,fontsize=15)
        # plt.xticks(rotation=45)
        # plt.show()


        sns.set(rc = {'figure.figsize':(20,10)})
        ax = sns.histplot(temp_df , x=time_level , hue='Return.Status' , multiple="fill" , discrete=True , color = 'Pastel1')
        ax.set_xlabel(time_level,fontsize=15)
        ax.set_ylabel('Number of Orders',fontsize=15)
        ax.set_title('Orders over '+time_level+' by Return Status' ,fontsize=15)
        plt.xticks(rotation=45)
        plt.show()

