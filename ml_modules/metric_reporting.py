from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import confusion_matrix as confmat
from sklearn.metrics import precision_recall_curve , classification_report
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, auc , balanced_accuracy_score , accuracy_score , f1_score , average_precision_score

Dclf = DummyClassifier(strategy='stratified', random_state=0)  # strategy='stratified', 'most_frequent'
    
import pandas as pd
import numpy as np


import seaborn as sns
import seaborn as sb
sns.set()

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.ticker import PercentFormatter



sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

import warnings
warnings.filterwarnings('ignore')



    
def get_confmat_analytics( Y_train ,  Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model ):
    print('''
Confusion matrix Format
[ [ tn , fp ]     all 0z
  [ fn , tp ] ]   all 1z
    ''')
    print()
    print('Confusion Matrix for Train :')
    #cm1 = regressor_confmat(Y_train , Y_train_pred);  #custom function find code below
    cm1 = confmat(Y_train , Y_train_pred )#, normalize='all');
    cm1norm = cm1/np.sum(cm1)
    print()
    print(cm1)
    print()
    print(cm1norm)
    print()
    print()

    ######

    print('Confusion Matrix for Test :')

    cm2 = confmat(Y_test , Y_test_pred )#, normalize='all');

    cm2norm = cm2/np.sum(cm2)
    print()
    print(cm2)
    print()
    print(cm2norm)
    print()
    print()


  ######


def get_f1_score_analytics( Y_train ,  Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model ):
    f1_train = f1_score(Y_train, Y_train_pred , zero_division='warn')

    f1_test = f1_score(Y_test, Y_test_pred , zero_division='warn')
    print()
    print ( 'F1 Score for Train : ' , f1_train )
    print ( 'F1 Score for Test  : ' , f1_test  )

    Dclf.fit(X_train, Y_train)

    f1_train_Dummy = f1_score(Y_train, Dclf.predict(Y_train) , zero_division='warn')            
    f1_test_Dummy = f1_score(Y_test, Dclf.predict(Y_test), zero_division='warn')

    print ( 'F1 Score for Train Dummy : ' , f1_train_Dummy )
    print ( 'F1 Score for Test  Dummy : ' , f1_test_Dummy  )
    print()

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#precision-recall

def get_AUCROC_plot( Y_train ,  Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model ):
    print('AUC-ROC Curve')

    plt.figure(figsize=(16,8)) 
    ax = plt.gca()
    
    rfc_disp = plot_roc_curve(model, np.array(X_test), np.array(Y_test), ax=ax , name = 'Test-ROC')
    rfc_disp = plot_roc_curve(model, np.array(X_train), np.array(Y_train), ax=ax , name = 'Train-ROC')
    rfc_disp = plot_roc_curve(Dclf , np.array(X_test), np.array(Y_test), ax=ax , name = 'Dummy_Clf')
    plt.legend(loc = 'lower right')

    rfc_disp.plot(ax=ax)
    print()

    
def get_AUCPRC_plot( Y_train ,  Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model ):
    print('AUC-PRC Curve')

    plt.figure(figsize=(16,8)) 
    ax = plt.gca()

    Dclf.fit(X_train, Y_train)
    disp = plot_precision_recall_curve(model, X_test, Y_test , ax=ax , name = 'Test_AUPRC' )
    disp = plot_precision_recall_curve(model, X_train, Y_train , ax=ax , name = 'Train_AUPRC' )
    disp = plot_precision_recall_curve(Dclf , X_train, Y_train , ax=ax , name = 'DummyClf' )

    disp.plot(ax=ax)
    print()
    
    
    
    
def get_classifier_analytics( Y_train ,  Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model ):
    get_confmat_analytics( Y_train ,  Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model )
    print()
    print('#'*75)
    get_f1_score_analytics( Y_train ,  Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model )
    print()
    print('Training Classification Report')
    print(classification_report(Y_train_pred,Y_train))
    print()
    print('Testing  Classification Report')
    print(classification_report(Y_test_pred,Y_test))
    print()
    print('#'*75)
          
    get_AUCROC_plot( Y_train , Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model )
    get_AUCPRC_plot( Y_train , Y_train_pred , Y_test , Y_test_pred , X_train , X_test , model )