import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

import sklearn.preprocessing
from env import get_db_url


##############################################   ACQUIRE     ##############################################

def get_zillow_data():
    '''This function will check to see if a zillow file exists and read it.
       If the file doesn't exist it will run a sql query and cache query to csv.'''
    filename = 'zillow.csv'

    #checks if the file already exists
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    #if not, queries db
    else:
        sql = '''
                 SELECT bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, fips
                 FROM properties_2017
                 JOIN predictions_2017 USING (parcelid)
                 WHERE propertylandusetypeid = 261 AND transactiondate LIKE '2017%%';
                '''
        
        url = get_db_url('zillow')

        df = pd.read_sql(sql, url)

    #caches the the file
        df.to_csv(filename, index=False)

        return df
##############################################   CLEAN     ##############################################

def clean_zillow(df):
    '''This function cleans up the zillow data.'''
    # drop all of the null values
    df = df.dropna()

    #rename the columns
    df = df.rename(columns={'bedroomcnt':'bedcount',
                        'bathroomcnt':'bathcount',
                        'calculatedfinishedsquarefeet': 'sqfeet',
                        'taxvaluedollarcnt': 'value',
                        'fips': 'county'})

    # change the dtype for the necessary columns
    df['sqfeet'] = df.sqfeet.astype(int)

    df['county'] = df.county.astype(int).astype(str)

    #replace the values for readability
    df = df.replace({'6037': 'Los Angeles', '6059': 'Orange', '6111': 'Ventura'})

    #calculate iqr for removing outliers
    q3_bath, q1_bath = np.percentile(df.bathcount, [75, 25])
    iqr_bath = q3_bath - q1_bath

    q3_bed, q1_bed = np.percentile(df.bedcount, [75, 25])
    iqr_bed = q3_bed - q1_bed

    q3_sqft, q1_sqft = np.percentile(df.sqfeet, [75, 25])
    iqr_sqft = q3_sqft - q1_sqft

    q3_val, q1_val = np.percentile(df.value, [75, 25])
    iqr_val = q3_val - q1_val

    #remove the outliers
    df = df[~((df['bathcount']<(q1_bath-1.5*iqr_bath)) | (df['bathcount']>(q3_bath+1.5*iqr_bath)))]

    df = df[~((df['bedcount']<(q1_bed-1.8*iqr_bed)) | (df['bedcount']>(q3_bed+1.8*iqr_bed)))]  

    df = df[~((df['sqfeet']<(q1_sqft-42*iqr_sqft)) | (df['sqfeet']>(q3_sqft+42*iqr_sqft)))]

    df = df[~((df['value']<(q1_val-684*iqr_val)) | (df['value']>(q3_val+684*iqr_val)))]

    return df

##############################################   SPLIT     ##############################################

def split_zillow(df):
    '''This function splits the clean zillow data stratified on value'''
    #train/validate/test split
    
    train_validate, test = train_test_split(df, test_size = .2, random_state=311)

    train, validate = train_test_split(train_validate, test_size = .25, random_state=311)

    return train, validate, test


##############################################  PREPARE - SCALED    ##############################################

def scaled_data(train, validate, test):
    '''This function takes in the train, validate, and test dataframes and returns the scaled data as dataframes.'''
    
    # drop columns with str values
    train = train.drop(columns='county') 

    validate = validate.drop(columns='county')

    test = test.drop(columns='county')

    # make and fit
    scaler = sklearn.preprocessing.MinMaxScaler()

    scaler.fit(train)

    #scale
    train_scaled = pd.DataFrame(scaler.transform(train))
    validate_scaled = pd.DataFrame(scaler.transform(validate))
    test_scaled = pd.DataFrame(scaler.transform(test))
    
    return train_scaled, validate_scaled, test_scaled


##############################################  MODEL SPLIT    ##############################################

def zillow_model_split(train, validate, test):
    '''This function splits the train, validate, test datasets from the target variable to prepare it for model.'''

    #train_validate, test = train_test_split(df, test_size = .2, random_state=311)

   #train, validate = train_test_split(train_validate, test_size = .25, random_state=311)

    X_train = train.drop(columns=['value', 'county'])

    y_train = train.value

    X_validate = validate.drop(columns=['value', 'county'])

    y_validate = validate.value

    X_test = test.drop(columns='value')

    y_test = test.value

    return X_train, y_train, X_validate, y_validate, X_test, y_test

