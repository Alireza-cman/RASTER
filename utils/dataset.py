
import random as rd
import numpy as np
import pandas as pd 

from sklearn.metrics import accuracy_score
# import minirocket_multivarient
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# import rocket.minirocket_multivariate as mrm
# import rocket.rocket as rk
from tqdm.notebook import tqdm
from os import walk
import matplotlib.pyplot as plt
import seaborn as sns


from csv import writer
def write_to_csv(name,my_list):
    with open(name, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(my_list)
        f_object.close()
    

    
def show_all_dataset():
    directory ='./datasets/'
    archive_files = []
    for (dirpath, dirnames, filenames) in walk(directory):
        archive_files.extend(dirnames)
    files_name = list(filter(lambda x: '_' not in x, archive_files))   
    return files_name

def load_dataset_description():
    df = pd.read_csv('/Users/alirezakeshavarzian/ThesisProject/SW/TSCDescription.csv')
    df = df.iloc[:,:6]
    return df 


def load_dataset(dataset_name,verbose= True):
    directory ='/Users/alirezakeshavarzian/ThesisProject/Dataset/UCRArchive_2018/'
    archive_files = []
    for (dirpath, dirnames, filenames) in walk(directory):
        archive_files.extend(dirnames)
    files_name = list(filter(lambda x: '_' not in x, archive_files))   
    for dataset in files_name: 
        if dataset not in [dataset_name]:
            continue
        if verbose == True: 
            print(dataset, ": ",end='\t')
        TRAIN = directory + dataset +'/' + dataset +'_TRAIN.tsv' 
        TEST = directory + dataset +'/' + dataset +'_TEST.tsv' 

        X_train_pandas = pd.read_csv(TRAIN, sep='\t',header=None)
        X_test_pandas = pd.read_csv(TEST, sep='\t',header=None)
        #
        X_train = X_train_pandas.drop([0],axis=1).fillna(0).to_numpy().astype(np.float64)
        X_test = X_test_pandas.drop([0],axis=1).fillna(0).to_numpy().astype(np.float64)
    #         ########=============
        y_train = X_train_pandas[0].tolist()
        y_test = X_test_pandas[0].tolist()
        if verbose == True: 
            print(X_train.shape , X_test.shape)
            print('#labels: ',np.unique(y_train))
    return X_train, y_train , X_test, y_test

def load_dataset_multivariate(dataset_name,verbose= True):
    directory ='/home/akeshavarzian/Multivariate_ts/Dataset/'
    archive_files = []
    for (dirpath, dirnames, filenames) in walk(directory):
        archive_files.extend(dirnames)
    files_name = list(filter(lambda x: '_' not in x, archive_files))   
    for dataset in files_name: 
        if dataset not in [dataset_name]:
            continue
        if verbose == True: 
            print(dataset, ": ",end='\t')
        TRAIN = directory + dataset +'/' + dataset +'_TRAIN.ts' 
        TEST = directory + dataset +'/' + dataset +'_TEST.ts' 

        # dataset_name = 'SelfRegulationSCP2'
        X_train, y_train =load_from_tsfile_to_dataframe(TRAIN)
        X_test, y_test =load_from_tsfile_to_dataframe(TEST)
        X_train = from_nested_to_3d_numpy(X_train).transpose(0,2,1)
        X_test = from_nested_to_3d_numpy(X_test).transpose(0,2,1)
        if verbose == True: 
            print(X_train.shape , X_test.shape)
            print('#labels: ',np.unique(y_train))
    return X_train, y_train , X_test, y_test



def ppv(x,bias = 0):
    """ Calculate the positive portion value

    Parameters
    ----------
    x : array 1D
        input signal
    bias : float
        bias indicates from where we should assume positive values 

    Returns
    -------
    float
    """
    s = np.array(x)>bias
    return np.sum(s)/len(x)


def ppv_slop(x, bias , sr = 0.0):
    slop = np.linspace(-1,1,len(x))
    slop *= sr
    bias = bias+  slop
    s = np.array(x)> bias
    return np.sum(s)/len(x) 


def dilute(v,d_rate):
    """ dilute vector v with rate d_rate

    Parameters
    ----------
    v : array[]
        vector
    d_rate : int
        dilution rate, d_rate = 0 means no dilution
        

    Returns
    -------
    array[]
        diluted vector
    """
    if d_rate < 1 :
        return v
    length = len(v)
    result = np.zeros((length-1)*(d_rate)+ length)
    for i in range(length):
        result[i*(d_rate)+i] = v[i]
    return result


def plt_show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
#                 _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                _y = p.get_y() + p.get_height() + 2.5
                value = '{:.1f}'.format(p.get_height())
                rot = 90
                if value == '99.0':
                    rot = 0 
                ax.text(_x, _y, value, ha="center",rotation=rot) 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)
        
        
def describe_parameter(parameters):
    if len(parameters) == 5 : 
        s_v , s_b, s_d, s_s, s_i = parameters
    else: 
        s_v , s_b, s_d, s_i = parameters
        s_s = None
    
    print("number of features: ",len(s_v) )
    print("===============================")
    print("distribution of dilution:")
    a, b = np.unique(s_d, return_counts = True)
    print(a)
    print(b)
    print("===============================")
    if type(s_s) != type(None):   
        print("distribution of s_s:")
        a, b = np.unique(s_s, return_counts = True)
        print(a)
        print(b)
        df = {}
        df['slope']= a
        df['population'] = b
        df = pd.DataFrame(df)
        sns.barplot(df,x = 'slope', y = 'population')
    print("===============================")
    print('total sum score: ')
    print( np.sum(s_i))
    
    
def neaty_index(y):
    labels = np.unique(y)
    neaty_indexes = []
    for l in labels:
        indices = [i for i, e in enumerate(y) if e == l]
        neaty_indexes.append(indices)
    return neaty_indexes

def concatenate_cohorts(cohorts):
    x_train_list, y_train_list, bio_train_list = [], [], []
    x_valid_list, y_valid_list, bio_valid_list = [], [], []
    x_test_list, y_test_list, bio_test_list = [], [], []
    
    for train, valid, test in cohorts:
        train_data, train_label, train_bio = train
        valid_data, valid_label, valid_bio = valid
        test_data, test_label, test_bio = test

        x_train_list.append(train_data)
        y_train_list.append(train_label)
        bio_train_list.append(train_bio)
        
        x_valid_list.append(valid_data)
        y_valid_list.append(valid_label)
        bio_valid_list.append(valid_bio)
        
        x_test_list.append(test_data)
        y_test_list.append(test_label)
        bio_test_list.append(test_bio)

    x_train = np.concatenate(x_train_list)
    y_train = np.concatenate(y_train_list).ravel()
    bio_train = np.concatenate(bio_train_list)
    
    x_valid = np.concatenate(x_valid_list)
    y_valid = np.concatenate(y_valid_list).ravel()
    bio_valid = np.concatenate(bio_valid_list)
    
    x_test = np.concatenate(x_test_list)
    y_test = np.concatenate(y_test_list).ravel()
    bio_test = np.concatenate(bio_test_list)

    id_train = bio_train[:, 0]
    id_test = bio_test[:, 0]
    try:
        id_valid = bio_valid[:,0]
    except:
        id_valid = []

    return (x_train, y_train, bio_train, id_train), (x_valid, y_valid, bio_valid,id_valid), (x_test, y_test, bio_test, id_test)


