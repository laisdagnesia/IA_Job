# from sklearn.ensemble import IsolationForest
import sklearn as sk
# print(sk.__version__)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import time

def detect_outliers(df, contamination=0.05, random_state=42, n_jobs=-1):
    
    s = time.time()
    
    # Create a copy of the input data frame
    df_scaled = df.copy()

    # Store column names
    columns = df_scaled.columns

    # Perform Isolation Forest with hyperparameter tuning
    isolation_forest_params = {'n_estimators': [50, 100, 200], 'max_samples': ['auto', 0.5, 0.7], 'contamination': [contamination]}
    isolation_forest = GridSearchCV(IsolationForest(random_state=random_state), isolation_forest_params, scoring='neg_mean_squared_error', cv=3, n_jobs=n_jobs)
    isolation_forest.fit(df_scaled)
    df_scaled['isolation_forest'] = isolation_forest.predict(df_scaled)

    # Perform One-Class SVM with hyperparameter tuning
    svm_params = {'kernel': ['linear', 'rbf'], 'nu': [0.01, 0.05, 0.1]}
    one_class_svm = GridSearchCV(OneClassSVM(), svm_params, scoring='neg_mean_squared_error', cv=3, n_jobs=n_jobs)
    one_class_svm.fit(df_scaled)
    df_scaled['one_class_svm'] = one_class_svm.predict(df_scaled)

    # Perform Local Outlier Factor
    lof = LocalOutlierFactor(contamination=contamination)
    df_scaled['lof'] = lof.fit_predict(df_scaled)
    
    # Count the number of '-1' values in each row
    outlier_counts = df_scaled[['isolation_forest', 'one_class_svm', 'lof']].eq(-1).sum(axis=1)
    
    # Remove classification columns
    df_scaled = df_scaled.drop(columns=['isolation_forest', 'one_class_svm', 'lof'])
    
    # Identify rows with more than one classification as an outlier
    outlier_rows = df_scaled[outlier_counts > 1]
    outlier_rows['outlier_flag'] = 1

    # Keep only rows that are not classified as outliers by any model
    df_cleaned = df_scaled.drop(outlier_rows.index, axis=0)
    df_cleaned['outlier_flag'] = 0

    # Evaluate performance
    print('Total time for outlier detection: {:,.2f} seconds'.format(time.time()-s))
    print('{:,.2f}% of the base were outliers ({:,})'.format((len(outlier_rows)/len(df_cleaned))*100, len(outlier_rows)))

    return pd.concat([df_cleaned, outlier_rows]).sort_index()