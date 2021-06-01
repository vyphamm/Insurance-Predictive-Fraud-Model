import numpy as np
import pandas as pd
import regex as re
import calendar
from sklearn.preprocessing import OneHotEncoder


def process_data_gm(data, pipeline_functions, prediction_col):
    """Process the data for a guided model."""
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    X = data.drop(columns=[prediction_col])
    y = data.loc[:, prediction_col]
    return X, y 

def remove_duplicates_and_full_nulls(data):
    """Remove any duplicates or rows with all null entries"""
    data = data.copy()
    data = data.drop_duplicates()
    data = data.dropna(how ='all')
    return data 

def fill_drop_null_entries(data, col_mappings):
    """ Fill null entries for each column based on what is indicated in the col_mappings
        col_mappings: dictionary with key = column and 
                      value = value to fill null entries in column
    """
    data = data.copy()
    for col in col_mappings: 
        data[col] = data[col].fillna(col_mappings[col])
    data = data.dropna()
    return data 

def clean_name(data):
    data = data.copy() 
    # Process the `name` column. 
    data['name'] = (data['name']
                    .str.replace('-','')
                    .str.lower()
                    .str.rstrip())
    data['name'] = data['name'].str.replace("collision at intersection", 
                                            "collision at intersect")
    return data 

def clean_auto_make(data): 
    # Process `auto_make`
    data = data.copy()
    data['auto_make'] = data['auto_make'].str.replace("SUBUWU", "SUBARU").str.lower() # UWU
    return data 

def extract_month_hour(data, lossdate_col):
    """Convert string date to datetime object, then create a datetime object
    to extract month and hour. Creates a month and hour column. 
    """
    # Convert to datetime and extract necessary features 
    data = data.copy()
    data['lossdatetime'] = data[lossdate_col].apply(pd.to_datetime)
    data['month'] = data['lossdatetime'].apply(lambda x:x.month)
    data['hour'] = data['lossdatetime'].apply(lambda x: x.hour)
    return data 

def damage_ordinal_encoding(data):
    """Perform Ordinal Encoding on the damage Column"""
    data = data.copy()
    damages = ("Total Loss", "Major Damage", "Minor Damage", "Trivial Damage", "NA")
    damagesEncode = (4, 3, 2, 1, -1)
    data['incident_severity'] = data['incident_severity'].replace(damages,damagesEncode)
    return data

def ohe_name(data, target, expected_names):
    """
    One-hot-encodes name.  New columns are of the form 0x_QUALITY.
    Unknown names will be categorized into "other". This will help us retain the 
    shape of our input data since it must match the shape of the training data. 
    """
    data = data.copy()
    oh_enc = OneHotEncoder()
    data[target] = data[target].apply(lambda x: x if x in expected_names else "other")
    oh_enc.fit(data[[target]]) # determine specific values that a categorical feature can take
    dummies = pd.DataFrame(oh_enc.transform(data[["name"]]).todense(), 
                       columns = oh_enc.get_feature_names(),
                       index = data.index
                )
    data = data.join(dummies).drop(target, axis=1)
    data = make_ohe_required_cols(data, expected_names)
    return data
    
def make_ohe_required_cols(data, columns):
    """Create a the columns necessary for the input data so it matches the shape
    of the training data. 
    """
    data = data.copy()
    data_cols = set(data.columns)
    for col in columns:
        if "x0_" + col not in data_cols:
            data["x0_" + col] = 0
    return data 
        
    
def auto_make_target_encoding(data, col_name, encodings): 
    """ Target encode automake."""
    def determine_encoding(x):
        try: 
            return encodings[x]
        except:
            return np.mean(list(encodings.values()))
    
    data = data.copy()
    data[col_name] = data[col_name].apply(lambda x: determine_encoding(x))
    return data 

def select_cols(data, columns): 
    return data[columns]