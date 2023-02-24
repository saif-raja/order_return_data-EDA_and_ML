import pandas as pd
import numpy as np
import os
import pickle

np.seterr(all='warn', divide='warn', over='warn', under='warn', invalid='warn')

pd.set_option('display.max_columns', 75)
pd.set_option('display.max_rows', 75)


def preprocess_raw_data ( df ):


    df['Order.Date']= pd.to_datetime(df['Order.Date'], errors='coerce')
    df['Ship.Date']=  pd.to_datetime(df['Ship.Date'] , errors='coerce')

    # df['Return.Status']     = df['Return.Status']       .astype('bool')


    df['Customer.Name']     = df['Customer.Name']       .astype('category')
    df['Customer.Segment']  = df['Customer.Segment']    .astype('category')
    df['Product.Category']  = df['Product.Category']    .astype('category')
    df['Product.Container'] = df['Product.Container']   .astype('category')
    df['Region']            = df['Region']              .astype('category')
    df['Ship.Mode']         = df['Ship.Mode']           .astype('category')
    df['Order.Priority']    = df['Order.Priority']      .astype('category')

    df = df[[
        #  'Row.ID',
        #  'Order.ID',
        'Order.Date',
        'Ship.Date',

        'Order.Quantity',
        'Sales',
        'Discount',
        'Profit',
        'Unit.Price',
        'Shipping.Cost',
        'Product.Base.Margin',

        'Customer.Name',
        'Customer.Segment',
        'Product.Category',
        'Region',
        'Order.Priority',
        'Ship.Mode',
        'Product.Container',

        'Return.Status'
    ]]
    
    return df





def generate_features ( df ):


    df['Days_to_Ship'] = (df['Ship.Date'] - df['Order.Date']).dt.days.astype('int16')

    df["Order_Quarter"]       = df['Order.Date'].dt.quarter

    df["Order_Month"]         = df['Order.Date'].dt.month
    df["Order_Weekday"]       = df['Order.Date'].dt.weekday
    df["Order_Day"]           = df['Order.Date'].dt.day

    df["Ship_Month"]          = df['Ship.Date'].dt.month
    df["Ship_Weekday"]        = df['Ship.Date'].dt.weekday
    df["Ship_Day"]            = df['Ship.Date'].dt.day

    df["Order_Quarter"]       = df["Order_Quarter"]     .astype('category')
    df["Order_Month"]         = df["Order_Month"]       .astype('category')
    df["Order_Weekday"]       = df["Order_Weekday"]     .astype('category')
    df["Ship_Month"]          = df["Ship_Month"]        .astype('category')
    df["Ship_Weekday"]        = df["Ship_Weekday"]      .astype('category')

    df["Order_Day_cat"]       = df["Order_Day"]         .astype('category')
    df["Ship_Day_cat"]        = df["Ship_Day"]          .astype('category')

    df = df.drop('Order.Date', axis=1)
    df = df.drop('Ship.Date' , axis=1)

    return df






from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


PCA_features_num = 6

def get_Kernel_PCA():
    inst = KernelPCA(

          n_components= PCA_features_num ,
        
          kernel='rbf', 
          # “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”
          
          gamma=1.35, 
          # default=1/n_features
          # Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other kernels.
          
          #max_iter=10000 ,
          # Maximum number of iterations for arpack. If None, optimal value will be chosen by arpack.
          
          
          #eigen_solver='auto', 
          # Select eigensolver to use. If n_components is much less than the number of training samples, arpack may be more efficient than the dense eigensolver.
          
          #tol=0, 
          # Convergence tolerance for arpack. If 0, optimal value will be chosen by arpack.
          random_state=42,           
          copy_X=False, 
          # If True, input X is copied and stored by the model in the X_fit_ attribute. 
          # If no further changes will be done to X, setting copy_X=False saves memory by storing a reference.
          n_jobs=-1
          
         )

    return inst


def gen_PCA_features( df ):
    
    feature_subset = [
        'Order.Quantity',
        'Sales',
        'Discount',
        'Profit',
        'Unit.Price',
        'Shipping.Cost',
        'Product.Base.Margin',
        ] 
       
    if os.path.exists("ml_assets\\PCA_features_maker.pkl") :
        PCA_maker = pickle.load(open('ml_assets\\PCA_features_maker.pkl' , 'rb'))
    
        df_PCA_features = PCA_maker.transform ( df[feature_subset] )

    else :

        PCA_maker = Pipeline([
            ('RS'   , RobustScaler   () ),
            ('KPCA' , get_Kernel_PCA () ),
            ])
        
        df_PCA_features = PCA_maker.fit_transform ( df[feature_subset] )
        
        pickle.dump( PCA_maker , open( 'ml_assets\\PCA_features_maker.pkl' , 'wb'))


    col_names = [ 'PCA_feature_' + str(i) for i in range( PCA_features_num ) ] 

    df_PCA_features = pd.DataFrame( df_PCA_features , columns= col_names )

    df_final = pd.concat([df, df_PCA_features], axis=1)
    
    return df_final





    
def export_processed_data ( df ) :
    
    X = df.drop('Return.Status', axis=1)
    Y = df['Return.Status']
    
    X.to_csv(   'data_prepared\\X.csv'       , header=True , float_format='%.6f' ,index=False)
    Y.to_csv(   'data_prepared\\Y.csv'       , header=True , float_format='%.6f' ,index=False)
    df.to_csv(  'data_prepared\\Preprocessed_Data.csv'  , header=True , float_format='%.6f' ,index=False)

    return df , X , Y








