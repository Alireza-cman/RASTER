import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif as sklearn_f_classif
from utils import RASTER
import matplotlib.pyplot as plt
from utils import miniROCKET as mr



def compare_2method(df,major, minor, threshold = 0.02):
    # Group the dataframe by Dataset
    grouped = df.groupby('Dataset')
    
    # Initialize a list to store the results
    results = []
    
    # Iterate through each group
    for dataset, group in grouped:
        # Find the accuracy for MiniROCKET and RASTER methods
        minor_acc = group[group['Method'] == minor]['Accuracy'].values
        major_acc = group[group['Method'] == major]['Accuracy'].values
        
        # Check if both methods exist for this dataset
        if len(minor_acc) > 0 and len(major_acc) > 0:
            # Calculate the difference in accuracy
            diff = major_acc[0] - minor_acc[0]
            
            # Add the result to the list
            results.append({
                'Dataset': dataset,
                major+'_Accuracy': major_acc[0],
                minor+'_Accuracy': minor_acc[0],
                'Difference': diff
            })
    
    # Create a dataframe from the results
    result_df = pd.DataFrame(results)
    
    # Sort the dataframe by the difference in descending order
    result_df = result_df.sort_values('Difference', ascending=False)
    
    # Filter for datasets where RASTER outperforms MiniROCKET by at least 2%
    result_df = result_df[result_df['Difference'] >= threshold]
    
    return result_df


