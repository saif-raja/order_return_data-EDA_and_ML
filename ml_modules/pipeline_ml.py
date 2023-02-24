from sklearn.model_selection import cross_val_score , cross_validate 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import pprint
import pickle 
pp = pprint.PrettyPrinter(indent=4)


import catboost
print('catboost %s' % catboost.__version__)
from catboost import CatBoostClassifier

import skopt
print('skopt %s' % skopt.__version__)
from skopt import BayesSearchCV

import sklearn
print('sklearn %s' % sklearn.__version__)


cat_features = [
    'Customer.Name',
    'Customer.Segment',
    'Product.Category',
    'Region',
    'Order.Priority',
    'Ship.Mode',
    'Product.Container',
    "Order_Quarter",
    "Order_Month",
    "Order_Weekday",
    "Ship_Month",
    "Ship_Weekday",
    "Order_Day_cat",
    "Ship_Day_cat",
    ]




def run_ml_pipeline( X,Y , tune_FLAG ) :

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1 , shuffle =True )


    print('X size : ' ,X.shape ) 
    print('Y size : ' ,Y.shape ) 

    print('X_train size : ' , X_train.shape )
    print('Y_train size : ' , Y_train.shape )
    print('X_test  size : ' , X_test.shape )
    print('Y_test  size : ' , Y_test.shape )


    model = get_model ( X_train, Y_train, tune_FLAG = tune_FLAG )

    Y_train_pred    = model.predict(X_train)
    Y_test_pred     = model.predict(X_test)
    
    # Y_train_pred    = Y_train_pred.astype('str')
    # Y_test_pred     = Y_test_pred .astype('str')
    
    Y_train_pred    = pd.Series(Y_train_pred , name='Return.Status')
    Y_test_pred     = pd.Series(Y_test_pred  , name='Return.Status')

    # Y_train    = Y_train.astype('str')
    # Y_test     = Y_test .astype('str')

    X_train.to_csv(      'data_ml_output\\X_train.csv'      , header=True , float_format='%.6f' ,index=False)
    Y_train.to_csv(      'data_ml_output\\Y_train.csv'      , header=True , float_format='%.6f' ,index=False)
    Y_train_pred.to_csv( 'data_ml_output\\Y_train_pred.csv' , header=True , float_format='%.6f' ,index=False)

    X_test .to_csv(      'data_ml_output\\X_test.csv'       , header=True , float_format='%.6f' ,index=False)
    Y_test .to_csv(      'data_ml_output\\Y_test.csv'       , header=True , float_format='%.6f' ,index=False)
    Y_test_pred .to_csv( 'data_ml_output\\Y_test_pred.csv'  , header=True , float_format='%.6f' ,index=False)


    print('X , Y , Y_Pred Data Exported as CSV ')
    
    
    pickle.dump( model , open( 'ml_assets\\Predictor.pkl' , 'wb'))

    
    return X_train, Y_train, Y_train_pred, X_test, Y_test, Y_test_pred ,model




def get_base_Classifer():
    inst = CatBoostClassifier(
            cat_features= cat_features,

            iterations= 500,#None,
            learning_rate=1,#None,
            max_depth=3,
            # depth=10,
            min_child_samples=20,
            subsample=1,
            colsample_bylevel=0.1,
            early_stopping_rounds=10,
            eval_metric='F1',
            # one_hot_max_size=10,
            # n_estimators=50,
            
            l2_leaf_reg=2,
            # reg_lambda=0.01,
            # use_best_model=True,
            loss_function=None,
            boosting_type=None,
            num_boost_round=None,
            objective=None,
            score_function=None,
            feature_weights=None,

            thread_count=10,
            verbose=0,
            
            bagging_temperature = 8,
            # class_weights= {1:10,0:1},
            # auto_class_weights= True
        )
    return inst



from skopt.utils import Integer, Real

Params_BayesSearch = {
    # 'iterations': Integer( 100 , 500 , 'log-uniform' ),
    'learning_rate':   Real( 0.5 , 5 , 'log-uniform'),
    'max_depth':         Integer( 2 , 4 , 'uniform'),
    # 'subsample':        Real( 0.8, 1.0, 'uniform'),
    # 'colsample_bylevel': Real( 0.01, 0.1, 'log-uniform'),
    'l2_leaf_reg':  Real( 0.01 , 0.05 , 'log-uniform'),
    # 'reg_lambda': Real(1e-4, 0.2  , 'log-uniform'),
    # 'bagging_temperature': Real(5, 10, 'uniform'),
    'random_strength': Real(0, 0.2, 'uniform'),
    # 'auto_class_weights': [True , False]
    # 'class_weights': [ {1:x,0:1} for x in [1,2,5,8,10]]
    # 'class_weights': [ [1,x] for x in [1,2,5,8,10]]
}



def get_model ( X_train, Y_train, tune_FLAG = False ) :
    if not tune_FLAG :
        model = get_base_Classifer()
        model.fit( X_train, Y_train, use_best_model=True, eval_set=(X_train,Y_train) )
        return model  

    else :
        global BayesExplore
        BayesExplore = BayesSearchCV(
                    estimator = get_base_Classifer()  , 
                    search_spaces = Params_BayesSearch ,
                    cv=RepeatedStratifiedKFold( n_splits = 3 , n_repeats = 1 , random_state = 42 ),  
                    scoring='f1' , 
                    n_iter=30, 
                    # Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution. 
                    # Consider increasing n_points if you want to try more parameter settings in parallel.
                    
                    n_jobs=10,#-1, 
                    # Number of jobs to run in parallel. 
                    # At maximum there are n_points times cv jobs available during each iteration.
                    
                    #  SO IF THERE ARE 4 CV FOLDS , THEN ONLY 4 CORES WILL BE USED UNLESS n_points is also set to 4 
                    #    then 4x4 16 fold will be properly used 
                    # hence n_points is also important to be set 
                    # be careful of RAM explosion though 
                    
                                                          
                    n_points=4, # only for CPU
                    # Number of parameter settings to sample in parallel. 
                    
                    pre_dispatch='1*n_jobs', # only for CPU
                    # Controls the number of jobs that get dispatched during parallel execution. 
                    # Reducing this number can be useful to avoid an explosion of memory consumption 
                    #  when more jobs get dispatched than CPUs can process. This parameter can be:

                    # None, in which case all the jobs are immediately created and spawned. 
                    #     Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
                    # An int, giving the exact number of total jobs that are spawned
                    # A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’                    
            
                    iid=True, 
                    refit=True,
                    error_score=np.nan,#'raise', 
                    return_train_score=True ,
                    random_state=42, 
                    verbose=1000, 
                )
        BayesExplore.fit(X=X_train, y=Y_train )
        
        print("val. score: %s" % BayesExplore.best_score_)
        #print("test score: %s" % BayesExplore.score(X_test, Y_test))
        print('best params:')
        pp.pprint(BayesExplore.best_params_)


        BE_results = pd.DataFrame(BayesExplore.cv_results_).dropna()
        BE_results.to_csv('ml_assets\\BayesSearchResults_1.csv')

        return BayesExplore
