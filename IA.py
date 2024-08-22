#!/usr/bin/env python
# coding: utf-8

# # Rotina de Preparação 

# In[80]:


#Preparação de dados para Machine Learning

#!pip install mlxtend
# !pip install featuretools

import pandas as pd
import numpy as np
import pickle
import ast
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import LabelEncoder
import featuretools as ft
import random 
import time

# data = dataframe para ser preparado
# target_column_name = coluna-alvo
# c_to_drop = lista de colunas que certamente não se relacionam com a Coluna alvo
# columns_to_slice = lista de tuplas com 2 strings: ('Nome da colna que pode ser splitada', 'delimitador') para ennriquecer a análise

# e.g.
# df,"value",['index','hash_code','unsable_column'],[('description',' ')]

def prepare_data_for_ML(o_data,target_column_name,c_to_drop,columns_to_slice):

    start = time.time()
    
    # Lista com mapeamentos, para usar o label encoder
    maps = []
    
    data = o_data

    # Step 1: Gather dataset(s) informaton and store in DataFrame
    print('Step 1: Gathering dataset(s) informaton and storing in DataFrame...')

    data = data.drop(c_to_drop, axis=1)

    for c_to_slice,slicer in columns_to_slice:
        data['sliced_{}'.format(c_to_slice)] = [str(list(string.split('{}'.format(slicer)))).lower() for string in data[c_to_slice]]


    # Step 2: Handeling Missing Values
    print('Step 2: Handeling Missing Values...')

    from sklearn.impute import SimpleImputer

    def handle_missing_values(df):
        # create a copy of the dataframe
        df_copy = df.copy()

        # loop through columns in dataframe
        for col in df_copy.columns:
            # check column data type

            try:

                if df_copy[col].dtype == object:
                    # Handle missing values for categorical columns
                    try:
                        categorical_imputer = SimpleImputer(strategy='most_frequent')
                        df_copy[col] = categorical_imputer.fit_transform(df_copy[[col]])
                    except:
                        pass

                elif df_copy[col].dtype == 'int64' or df_copy[col].dtype == 'float64':
                    # Handle missing values for numerical columns
                    try:
                        numeric_imputer = SimpleImputer(strategy='mean')
                        df_copy[col] = numeric_imputer.fit_transform(df_copy[[col]])
                    except:
                        pass

                elif df_copy[col].dtype == 'datetime64[ns]':
                    # Handle missing values for datetime columns
                    try:
                        datetime_imputer = SimpleImputer(strategy='most_frequent')
                        df_copy[col] = datetime_imputer.fit_transform(df_copy[[col]])
                    except:
                        pass
            except:
                pass

        try:
            return df_copy.reset_index(drop=True)
        except: 
            return df_copy


    import pandas as pd
    
    #avoiding internal structure inconsistences
    #data =  pd.DataFrame(data.to_dict())
    
    data = handle_missing_values(data)
    # display(data)
    
    #avoiding internal structure inconsistences
    #data =  pd.DataFrame(data.to_dict())


    # Step 3: Scale non-numerical data
    print('Step 3: Scaling non-numerical data...')

    import pandas as pd
    import numpy as np
    import ast

    #Função para fazer o encoding de uma lista de dicionários dentro de uma linha 
    def encode_dicts_in_column(df, data_col):

        def one_hot_encode_numeric_dicts_in_column(df, data_col):
            '''Dado um dataframe com uma coluna de listas de dicionários, identifica automaticamente todas as chaves cujos valores são
            numéricos (int ou float), cria uma coluna para cada chave, e o soma s valores de mesma chave.
            Se os valores encontrados na chave forem objeto, pula a chave.'''

            c_to_drop = list(df.columns)

            rows = []
            for _, row in df.iterrows():
                if isinstance(row[data_col], str):
                    list_of_dicts = ast.literal_eval(row[data_col])  #ensure list-like
                elif pd.isnull(row[data_col]).all():
                    list_of_dicts = []
                else:
                    list_of_dicts = row[data_col]

                row_dict = {}
                for d in list_of_dicts:
                    for key, value in d.items():
                        if isinstance(value, (int, float)):
                            row_dict[key] = row_dict.get(key, 0) + value

                rows.append(row_dict)

            new_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
            new_df = new_df.drop(c_to_drop+[data_col], axis=1)
            new_df = new_df.fillna(0)
            new_df.reset_index(drop=True, inplace=True)
            return new_df


        def one_hot_encode_object_dicts_in_column(df, data_col):
            """
            Dado um dataframe com uma coluna de listas de dicionários, identifica automaticamente todas as chaves cujos valores são
            objetos (strings), cria uma coluna para cada valor exclusivo encontrado na chave, mas apenas para valores do tipo objeto.
            Se os valores encontrados na chave forem numéricos, pula a chave.
            """
            c_to_drop = list(df.columns)
            new_df = df.copy()

            all_keys = set()
            all_objects = set()
            for row in new_df[data_col]:
                if row is not np.nan:
                    if isinstance(row, str):
                        row = ast.literal_eval(row) #ensure list-like
                    for d in row:
                        all_keys |= set(d.keys())
                        for k, v in d.items():
                            if isinstance(v, str):
                                all_objects.add(k)

            for obj in all_objects:
                # Verifica se a coluna já existe antes de criar uma nova coluna
                if obj not in new_df.columns:
                    new_df[obj] = 0

            for i, row in new_df.iterrows():
                log_data = row[data_col]
                if log_data is not np.nan:
                    if isinstance(log_data, str):
                            log_data = ast.literal_eval(log_data) #ensure list-like
                    for d in log_data:
                        for k, v in d.items():
                            if isinstance(v, str) and k in all_objects:
                                # Usa o operador de atribuição "+=" para adicionar valores a uma coluna existente
                                new_df.loc[i, v] = 1

            c_to_drop = c_to_drop + list(all_objects)
            new_df = new_df.drop(c_to_drop, axis=1)
            new_df = new_df.fillna(0)
            return new_df


        final_df = pd.concat([df, one_hot_encode_numeric_dicts_in_column(df, data_col), one_hot_encode_object_dicts_in_column(df, data_col)], axis=1)
        final_df = final_df.drop(data_col, axis=1)
        return final_df.reset_index(drop=True)


    from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder

    def auto_scaler(df, target_col):
        # create a copy of the dataframe
        df_scaled = df.copy()

        for col in [c for c in df_scaled.columns if c!=target_col]:
            # check column data type List of Dicts
            try:
                if str(df_scaled[col][0]).startswith('[{'):
                    # master one-hot encoding
                    df_scaled = encode_dicts_in_column(df_scaled, col)
            except:
                continue


        # loop through columns in dataframe
        for col in [c for c in df_scaled.columns if c!=target_col]:
            # check column data type
            try:
                #Testando e preparando dados em colunas de listas
                if str(df_scaled[col][0])[0]=='[':
                    
                    try:
                        # ensure list-like
                        df_scaled[col] = df_scaled[col].apply(lambda x: ast.literal_eval(x))
                        # one-hot encoding
                        one_hot = pd.get_dummies(df_scaled[col].apply(pd.Series).stack()).groupby(level=0).sum()
                        # concatenate the original data and one-hot encoding
                        df_scaled = pd.concat([df_scaled, one_hot], axis=1)
                        # drop the original 'lists' column
                        df_scaled.drop(col, axis=1, inplace=True)
                    
                    except Exception as err:
                        print(err)
                
                elif df_scaled[col].dtypes == 'object':
                    """
                    Dado um dataframe, tenta converter a coluna de strngs em objetos de data/hora usando a função pd.to_datetime.
                    Usa o parâmetro infer_datetime_format=True para tentar detectar automaticamente o formato da data na string.
                    Se isso Não for bem sucedido, usa one-hot encoding
                    """
                    
                    from dateutil import parser
                    
                    def is_date_column(series, threshold=0.9):
                        """
                        Verifica se uma coluna é provavelmente uma coluna de datas.

                        Parameters:
                        - series: pd.Series
                            A coluna a ser verificada.
                        - threshold: float, opcional (padrão=0.9)
                            Limite de confiança para decidir se a coluna é provavelmente uma coluna de datas.

                        Returns:
                        - bool
                            True se a coluna é provavelmente uma coluna de datas, False caso contrário.
                        """
                        try:
                            parsed_dates = series.apply(parser.parse)
                            valid_dates = parsed_dates[parsed_dates.notnull()]
                            date_ratio = len(valid_dates) / len(series)
                            return date_ratio >= threshold
                        except Exception as e:
                            return False
                    
                    if is_date_column(df_scaled[col]):
                        df_scaled[col] = pd.to_datetime(df_scaled[col], infer_datetime_format=True)
                    else:
                        # Decide se vai numerizar a coluna de objeto usando Get_dummies ou Label_encoder
                        
                        unique_values = df_scaled[col].nunique()
                        total_rows = len(df_scaled)

                        threshold = 0.1 #Label encoder só se até a coluna ter 10% de valores únicos
                        
                        if unique_values / total_rows < threshold:
                            # Usar LabelEncoder se a proporção de valores distintos for menor que o limiar
                            label_encoder = LabelEncoder()
                            
                            # Ajustar e transformar a coluna de strings
                            df_scaled[col] = label_encoder.fit_transform(df_scaled[col])

                            # Criar um dicionário de mapeamento entre valores originais e valores numéricos
                            mapeamento = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

                            # Exibir o dicionário de mapeamento
                            maps.append({col:mapeamento})
                            
                        else:
                            # Usar get_dummies se a proporção de valores distintos for maior ou igual ao limiar
                            df_scaled = pd.get_dummies(df_scaled, columns=[col], prefix=[col])


                elif df_scaled[col].dtypes == 'int64' or df_scaled[col].dtypes == 'float64':
                    # use MinMaxScaler for numerical variables
                    scaler = MinMaxScaler()
                    # df_scaled[[col]] = scaler.fit_transform(df_scaled[[col]])

            except:
                pass

        #Fazendo o tratamento da coluna-alvo para Modelos de Agrupamento
        if df_scaled[target_col].dtypes == 'object':
            le = LabelEncoder()
            df_scaled[target_col] = le.fit_transform(df_scaled[target_col])
        
        try:
            return df_scaled.reset_index(drop=True)
        except:
            return df_scaled

    
    #avoiding internal structure inconsistences
    #data =  pd.DataFrame(data.to_dict())
    
    data = auto_scaler(data, target_column_name)

    end = time.time()
    print("\n\nDados tratados em {:.2f} minutos".format((end-start)/60))

    return data, maps


# # Ranking V2

# In[83]:


# Consulta para descobrir todas as Ofertas coletadas pelo Scrapping da Carhunt

from pymongo import MongoClient
import pandas as pd
client = MongoClient('mongodb+srv://gabrielaraujon96:D3umaten0ve@carhunt.sweobl3.mongodb.net/')
db = client['carhunt']

#dados de ofertas:
base_v = pd.DataFrame(db['search_scraping'].find())

# Drop all rows that contain any NaN values
base_v = base_v.dropna(subset=['brand', 'city', 'km', 'model',
       'model_year', 'price', 'state'])


print(base_v)


# ## Pré engenharia de dados

# In[84]:


import datetime

#Clone da base original
coerced_base_v = base_v.copy()

#Preparação dos dados das ofertas para ML
import datetime


# Colunas que PRECISAM ser numéricas
num_cols = ['km', 'model_year', 'price']

# casting das colunas que devem ser numéricas por natureza
for c in num_cols:
    coerced_base_v[c] = [float(x) for x in coerced_base_v[c]]

# Criando coluna de age (anos desde a fabricação) que diminui a dimensionalidade dos dados
coerced_base_v['age'] = datetime.datetime.now().year-coerced_base_v['model_year'].astype(int)

# Dividir a coluna 'model' em duas colunas: 'model' e 'type'
coerced_base_v[['model', 'type']] = coerced_base_v['model'].str.split(pat=' ', n=1, expand=True)

# Adicionar uma nova coluna condicional
coerced_base_v['transmission'] = coerced_base_v['type'].apply(lambda x: 'Aut.' if 'aut' in x.lower() else 'Manual')

# Reajustar coluna de Type, removendo transmissão
coerced_base_v['type'] = coerced_base_v['type'].apply(lambda x: ' '.join([word for word in x.split() if(('aut' not in word.lower()) and ('manu' not in word.lower()))]))

# # Tipos de carrocerias de carros, para categorizar
# # https://www.icarros.com.br/catalogo/compare.jsp
# tipos_normalizados = ['Hatch (compacto)','Sedan','Picape','SUV (Utilitário esportivo)','Monovolume','Perua/Wagon (SW)','Van','Conversível (coupé)','Hibrido/Elétrico']

# norm_body_types = { 'Conversível': 'Conversível (coupé)',
#   'Coupé': 'Conversível (coupé)',
#   'Conversível (coupé)':'Conversível (coupé)',
#   'Coupê': 'Conversível (coupé)',
#   'Cupê': 'Conversível (coupé)',
#   'Hatch': 'Hatch (compacto)',
#   'Hatchback': 'Hatch (compacto)',
#   'Hatch (compacto)':'Hatch (compacto)',
#   'Compacto': 'Hatch (compacto)',
#   'Minivan': 'Van',
#   'Van/Utilitário': 'Van',
#   'Van':'Van',
#   'Furgão & Van': 'Van',
#   'Monovolume': 'Monovolume',
#   'Perua/SW': 'Perua/Wagon (SW)',
#   'SW & Perua': 'Perua/Wagon (SW)',
#   'Perua/Wagon (SW)':'Perua/Wagon (SW)',
#   'Picape': 'Picape',
#   'Picapes': 'Picape',
#   'Sedan': 'Sedan',
#   'Sedã': 'Sedan',
#   'Utilitário esportivo': 'SUV (Utilitário esportivo)',
#   'SUV (Utilitário esportivo)': 'SUV (Utilitário esportivo)',
#   'SUV': 'SUV (Utilitário esportivo)',
#   'SUV/Utilitário': 'SUV (Utilitário esportivo)',
#   'Hibrido':'Hibrido/Elétrico',
#   'Elétrico':'Hibrido/Elétrico',
#   'Hibrido/Elétrico':'Hibrido/Elétrico',
#   }

# # Fazendo a tradução na base:
# coerced_base_v['body'] = coerced_base_v['body'].map(norm_body_types)

#Setando todos os Body Types como SUv (ATE QUE O GABRIEL ADICIONE A COLUNA BODY)
coerced_base_v["body"] = 'SUV'

print(coerced_base_v)


# # Preparação dos dados das ofertas para ML

# In[85]:


# ------------------------------------------------
# Transformada para ML

#ARMAZENA A COLUNA ID ANTES DE DROPAR
id_column = coerced_base_v[['_id']].copy()
id_column['original_index'] = coerced_base_v.index
results_rank_df = coerced_base_v.drop(columns=['_id'])


#results_rank_df = coerced_base_v.drop(columns=['_id'])

#Colunas irrelevantes para a aprendizaem no que tange precificação
irrelevant_columns = ['_id', 'model_year', 'fab_year', 'link', 'source', 'ad_id', 'image', 'type']

target_col = "price"

base_v_toML, base_v_maps = prepare_data_for_ML(coerced_base_v,target_col,irrelevant_columns,[]) #[('type', ' ')]

# Aqui descobrimos quais colunas tiveram seus dados categóricos encoded como numericos
# pois para fins de exibição ao final das análises é necessário retraduzir
base_v_colunas_mapeadas = [list(m.keys())[0] for m in base_v_maps]
print('colunas mapeadas com Label Encoding:',base_v_colunas_mapeadas)
print(base_v_toML)


# # Detecção de Anomalias

# In[86]:


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


# ## Modelos de regressão

# In[87]:


# Testador de modelos de regressão

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, ARDRegression, PassiveAggressiveRegressor, TheilSenRegressor, SGDRegressor, Lars, LassoLars, LassoLarsCV, LassoLarsIC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import StackingRegressor#, RandomSubspace
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import time

def select_best_regression_model(data, target_col, cv_strategy='default', n_jobs=1, cv=None):
    
    start_global = time.time()
    
    scores = {}
    
    # Regression Models
    regression_models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=1.0),
        ElasticNet(alpha=1.0, l1_ratio=0.5),
        BayesianRidge(),
        HuberRegressor(),
        GaussianProcessRegressor(),
#         RandomSubspace(),  # Ensemble method
        StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor()),
                ('gb', GradientBoostingRegressor())
            ],
            final_estimator=LinearRegression()
        ),
        OrthogonalMatchingPursuit(),
        OrthogonalMatchingPursuitCV(),
        ARDRegression(),
        PassiveAggressiveRegressor(),
        TheilSenRegressor(),
        SVR(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        GradientBoostingRegressor(n_iter_no_change=10, tol=0.01),  # Early stopping
        MLPRegressor(max_iter=1000, early_stopping=True, n_iter_no_change=10),  # Early stopping
        SGDRegressor(),
        Lars(),
        LassoLars(),
        LassoLarsCV(),
        LassoLarsIC()
    ]

    # Create KFold cross-validation object
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for model in regression_models:
        model_name = model.__class__.__name__
        print('Testing model: {}'.format(model_name))

        try:
            if model_name in ['PCA', 'KernelPCA', 'TSNE']:
                model.fit(data.drop(target_col, axis=1))
                transformed_data = model.transform(data.drop(target_col, axis=1))
                reg_model = LinearRegression().fit(transformed_data, data[target_col])
                predictions = reg_model.predict(transformed_data)
            else:
                if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                    # Regularization for specific models
                    param_grid = {'alpha': [0.1, 1.0, 10.0]}
                    model = HalvingGridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=n_jobs)

                # Hyperparameter Tuning
                if cv_strategy == 'grid_search':
                    param_grid = {}  # Add hyperparameters for tuning
                    model = HalvingGridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=n_jobs)

                if hasattr(model, 'n_iter_no_change') and hasattr(model, 'tol'):
                    # Early Stopping for models that support it
                    model.n_iter_no_change = 10  # Number of consecutive iterations with no improvement
                    model.tol = 0.01  # Tolerance to declare convergence

                model.fit(data.drop(target_col, axis=1), data[target_col])
                predictions = model.predict(data.drop(target_col, axis=1))

            # Regression Model Scoring
            if cv_strategy != 'grid_search':
                # Cross-Validation Strategy
                if cv_strategy == 'stratified_kfold':
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                else:
                    cv_scores = cross_val_score(model, data.drop(target_col, axis=1), data[target_col], cv=kf, scoring='neg_mean_squared_error')
                    cv_score = cv_scores.mean()
            else:
                cv_score = None

            # Regression scores
            r2 = r2_score(data[target_col], predictions)
            mse = mean_squared_error(data[target_col], predictions)
            mae = mean_absolute_error(data[target_col], predictions)

            scores[model_name] = {'trained_model': model, 'r2_score': r2, 'mse_score': mse, 'mae_score': mae, 'cv_score': cv_score}

        except Exception as erro:
            print("Deu ruim: {}".format(erro))

            
    # -----------------------------------------------
    # Selecting the best model based on cost-benefit score
    scores_df = pd.DataFrame(scores).T.reset_index().rename(columns={'index':'model'})
    
    scores_df['final_score'] =  scores_df['mse_score'] / scores_df['r2_score']
    
    positive_scores_df = scores_df[scores_df['r2_score']>0].sort_values(by='final_score', ascending=True).reset_index(drop=True)
    
    best_model = list(positive_scores_df['trained_model'])[0]
            
    # Performance Visualization for the best model
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[target_col], y=best_model.predict(data.drop(target_col, axis=1)))
    plt.title(f"True vs. Predicted Values - {best_model.__class__.__name__}")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.show()

    # Learning Curve for the best model
    plt.figure(figsize=(10, 6))
    plot_learning_curve(best_model, best_model.__class__.__name__, data.drop(target_col, axis=1), data[target_col], cv=cv)
    plt.show()

    # Model Persistence
    # Save the best model to disk for later use
    #joblib.dump(best_model, 'best_model.pkl')

    
    end_global = time.time()
    print('\nThe best Regression model is:', list(positive_scores_df['model'])[0])
    
    print("\nEstudo finalzado em {:.2f} segundos".format((end_global - start_global)))
    
    return best_model, scores_df



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Curve - {title}")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# # Example usage:
# target_col = 'sre_value'
# best_model, scores_df = select_best_regression_model(data, target_col, cv_strategy='grid_search', n_jobs=-1, cv=None)
# display(scores_df)
# best_model


# In[88]:


# Visualização de modelo

from sklearn.tree import plot_tree
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
#import networkx as nx
import time

def visualize_best_model(best_model, data, target_col):
    start_global = time.time()
    
    if 'HalvingGridSearchCV' in str(best_model.__class__):
        # Extract the best estimator from HalvingGridSearchCV
        best_estimator = best_model.best_estimator_
        visualize_estimator(best_estimator, data, target_col)
    else:
        visualize_estimator(best_model, data, target_col)
        
    end_global = time.time()    
    print("\nPlot finalzado em {:.2f} min".format((end_global - start_global)/60))

    
def visualize_estimator(estimator, data, target_col):
    model_type = estimator.__class__.__name__

    if model_type == 'DecisionTreeRegressor':
        # Visualize decision tree and print number of nodes and edges
        plt.figure(figsize=(20, 10))
        plot_tree(estimator, filled=True, feature_names=data.drop(target_col, axis=1).columns)
        plt.show()

        num_nodes = estimator.tree_.node_count
        num_edges = num_nodes - 1
        print(f"Number of nodes: {num_nodes}, Number of edges: {num_edges}")

    elif model_type == 'MLPRegressor':
        # Visualize neural network architecture (simplified for illustration)
        plt.figure(figsize=(10, 6))
        plt.title("Neural Network Architecture")
        plt.imshow([len(data.drop(target_col, axis=1).columns), 10, 1], cmap='viridis', aspect='auto', extent=(0, 1, 0, 1))
        plt.axis('off')
        plt.show()

        num_neurons = sum(layer_size for layer_size in estimator.hidden_layer_sizes)
        print(f"Number of neurons: {num_neurons}")

    elif model_type in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
        # Visualize linear regression coefficients or other relevant information
        if hasattr(estimator, 'coef_'):
            plt.figure(figsize=(10, 6))
            plt.bar(data.drop(target_col, axis=1).columns, estimator.coef_)
            plt.title("Linear Regression Coefficients")
            plt.xlabel("Feature")
            plt.ylabel("Coefficient Value")
            plt.show()

            num_coefficients = len(estimator.coef_)
            print(f"Number of coefficients: {num_coefficients}")

    else:
        print(f"Visualization not implemented for {model_type}")


# Example usage:
# visualize_best_model(best_model, data, target_col)


# # Código de Ranking

# In[89]:


# Código de Ranking
#By DataMaster

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import numpy as np
import datetime
import math
import time
import joblib

# recebe um dataframe com abslutamente todas as ofertas da carhunt
def ML_ranking_gen2(df, df_maps, train_method='arbit', target_col = 'price', split_col = 'body'):
    
    #Clone dos dados de pesquisa, para não alterar os originais
    all_data = df.copy()
    
    start_global = time.time()

    # COnversão obrigatória dos valores dos carros
    all_data[target_col] = pd.to_numeric(all_data[target_col], errors="coerce")
        
    # Engenharia de recursos ----
    # usando Information gain para performar a divisão de categorias
    # ------
    
    # aqui vamos arbitrariamente usar 'sre_body_type' como divisão inicial da base
    categories = list(dict.fromkeys(all_data[split_col]))

    # Criação de um dicionário com modelos de machine learning treinados
    CarHuntML_models = {}
    
    # listapara salvar todos os chunks de dados separados no treiamneot dos modelos
    data_slices = []
    
    for cat in categories:
        
        data = all_data[all_data[split_col]==cat]
        
        # Se eu tenho dados:
        if len(data)>6:
            
            # qual o index em que a split_col está no mapa de dados

            splt_col_maps_idx = next((i for i, d in enumerate(df_maps) if list(d.keys())[0] == split_col), None)
            
            str_body_type = [k for k,v in df_maps[splt_col_maps_idx][split_col].items() if v==cat][0]
            print('\n{}'.format(str_body_type))

            original_data = all_data[all_data[split_col]==cat]

            # Se alguma das colunas passou reto na preparação para ML, remove aqui mesmo
            data = original_data.select_dtypes(include='number')

            print('Detecting outliers...')
            # Aqui é feita uma detecção de outliers na categoria
            data = detect_outliers(data)


            print('Training model...')

            # Create KFold cross-validation object
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            if train_method=='auto':
                # testes de modelos diferentes de Machine Learning para previsão do Preço (regressão)
                best_model, scores_df = select_best_regression_model(data, target_col, cv_strategy='grid_search', n_jobs=-1, cv=kf)


            elif train_method=='arbit':
                # Arbitratiamente setando o melhor modelo como DecisionTreeRegressor()

                # For HalvingGridSearchCV 
                scoring_m = 'neg_mean_squared_error' #'r2'

                if len(data)*len(data.columns)<1000000: #(menor que 1M amostras)
                    # sem evitar overfitting, pois o conjunto de dados é "pequeno" para ML 
                    print('Overfitting pois temos poucos dados')

                    try:
                        best_model = HalvingGridSearchCV(estimator=DecisionTreeRegressor(), n_jobs=-1, param_grid={}, scoring=scoring_m, cv=kf)
                        best_model.fit(data.drop(target_col, axis=1), data[target_col])
                    except:
                        pass
                else: #(maior que 1M amostras)
                    # Evitando overfitting, pois o conjunto de dados é grande o suficiente para maiores generalizações
                    print('Generalizando pois temos muitos dados')

                    # Tunando hiperparâmetros do modelo
                    # Automatically determine reasonable ranges for hyperparameters based on data size
                    max_depth_range = [None] + list(range(5, int((len(data.columns)-1)**1.75), int(len(data)/100)))
                    min_samples_split_range = list(range(2, int(len(data)**0.5), int(len(data)/100)))
                    min_samples_leaf_range = list(range(int(len(data)**0.15), int(len(data)**0.75), int(len(data)/100)))

                    # Define the parameter grid for the decision tree
                    param_grid = {
                        'max_depth': max_depth_range,
                        'min_samples_split': min_samples_split_range,
                        'min_samples_leaf': min_samples_leaf_range}

                    # Create the DecisionTreeRegressor and HalvingGridSearchCV objects
                    regressor = DecisionTreeRegressor()
                    grid_search = HalvingGridSearchCV(
                        estimator=regressor,
                        param_grid=param_grid,
                        scoring=scoring_m,
                        n_jobs=-1,
                        cv=kf)

                    # Fit the model to the data
                    grid_search.fit(data.drop(target_col, axis=1), data[target_col])

                    # Get the best model
                    best_model = grid_search.best_estimator_


            #---------------------------------------------------------------------
            # Salvando o modelo no dicionário, com o tipo de carroceria treinada
            CarHuntML_models[str_body_type] = best_model

            # Visualização
            #display(scores_df)    
            #visualize_best_model(best_model, data, target_col)

            # RANKING
            # Aqui vamos medir a distânica entre o valor previsto pelo modelo de precificação e o valor real
            # Se a diferença (valor_previsto - valor_real) for positiva, que dizer que o carro está com um desconto de diff
            # Agora se a diferença for negativa, que dizer que o carro está com um preço de diff acima do esperado

            # Desta maneira a maior diff será d melhor carro desa categoria, afinal está com o maior desconto

            predicted_values = list(best_model.predict(data.drop(target_col, axis=1)))

            data['diff_ml'] = [valor_previsto - valor_real  for valor_previsto, valor_real  in 
                            zip(predicted_values, list(data[target_col]))]

            # Em seguida vamos fazer uma escala de 0 a 10 com os valores de diff, 
            # e assim determinar notas para o RANKING CARHUNT

            # Create a MinMaxScaler instance and fit_transform the data
            scaler = MinMaxScaler(feature_range=(0, 10))

            # Reshape the data as MinMaxScaler expects a 2D array
            # Then Extract the normalized values from the resulting 2D array
            data['Ranking_CarHunt'] = [v[0] for v in scaler.fit_transform([[x/pv] for x,pv in zip(data['diff_ml'], predicted_values)])]

            data['predicted_ML_value'] = predicted_values

            print('Model trained!')

            # Readicionando as colunas não numéricas, se necessário
            inverse_selection = original_data.select_dtypes(exclude='number')

            # Concatenate the two DataFrames along the columns axis (axis=1)
            data = pd.concat([data, inverse_selection], axis=1).sort_index()

            #display(data)

            data_slices.append(data)
    
    
    # Reunificando dados -----
    all_data = pd.concat(data_slices).sort_index()

    end_global = time.time()
    print("Ranking finalizado em {:.2f} min".format((end_global - start_global)/60))
    
    return all_data, CarHuntML_models


# -----------------------------------
# CHAMADA

results_rank_df, CarHuntML_models = ML_ranking_gen2(base_v_toML, df_maps=base_v_maps, train_method='arbit', target_col = 'price', split_col = 'body')

# Model Persistence
# Save the best models to disk for later use
joblib.dump(CarHuntML_models, 'CarHuntML_models_{}.pkl'.format(datetime.datetime.now().strftime("%d%b%Y-%Hh%M")))    

print(results_rank_df)


#Retraduzindo os dados mapeados:
for col in [c for c in base_v_colunas_mapeadas if c in list(results_rank_df.columns)]:

    # encontrando o dicionário de mapeamento 
    mapping_dict = next(item.get(col) for item in base_v_maps if col in item)

    results_rank_df[col] = results_rank_df[col].map({v: k for k, v in mapping_dict.items()})
    

# ReCriando coluna de ano
results_rank_df['model_year'] = datetime.datetime.now().year-results_rank_df['age']



# # RECUPERAÇÃO

# In[ ]:


#==================================================
# COMO RECUPERAR AS COLUNAS ANTIGAS SEM EMBARALHAR? (LAÍS)


results_rank_df['original_index'] = results_rank_df.index

#renomea temporariamente a coluna '_id' p/ evitar conflitos 
results_rank_df['original_index'] = results_rank_df.index

id_column = id_column.rename(columns={'_id': '_id_temp'})

# merge com base no indice original
results_rank_df = pd.merge(results_rank_df, id_column, on='original_index')

#restaura o nome original da coluna '_id'
results_rank_df = results_rank_df.rename(columns={'_id_temp': '_id'})

#remove a coluna de indice orig usada para o merge
results_rank_df = results_rank_df.drop(columns=['original_index'])




#base_v_toML['_id'] = id_column

#Restaurar a coluna  de acordo com a ordem correta
#results_rank_df['_id'] = id_column

#results_rank_df = results_rank_df[['_id'] + [col for col in results_rank_df.columns if col != '_id']]


#====================================================

# Recuperando o ID original
#results_rank_df['_id'] = coerced_base_v['_id']

# Recolocando colunas anteriormente ignoradas na tratativa de ML (tentei com pandas merge)
final_full_sre_ranking = pd.merge(results_rank_df, coerced_base_v[irrelevant_columns], on='_id', how='inner')


#====================================================



# Ordenando campeões -----
final_full_sre_ranking = final_full_sre_ranking.sort_values("Ranking_CarHunt", ascending=False)

# Reorganizando as colunas
ml_cols=['outlier_flag','predicted_ML_value','price','diff_ml','Ranking_CarHunt']
final_full_sre_ranking = final_full_sre_ranking[[c for c in list(final_full_sre_ranking.columns) if c not in ml_cols]+ml_cols]

print(final_full_sre_ranking)


# In[ ]:


final_full_sre_ranking.values[100]


# In[ ]:


# f = ('sre_body_type','Hatch (compacto)')
f = ('sre_model', 'hb20')
threshold = 9

filt_fin_df = final_full_sre_ranking[final_full_sre_ranking[f[0]] == f[1]]
# filt_fin_df = filt_fin_df[filt_fin_df['sre_year'] == 2019]

top = filt_fin_df[filt_fin_df['Ranking_CarHunt'] > threshold]['sre_value'].mean()
general = filt_fin_df[filt_fin_df['Ranking_CarHunt'] <= threshold]['sre_value'].mean()

print('Tipo: {}'.format(f[1]))
print('Ofertas Analisadas: {:,}'.format(len(filt_fin_df)))

print('Preço médio (nota CarHunt acima de {}): R$ {:,.2f}'.format(threshold, top))
print('Preço médio (nota CarHunt abaixo de {}): R$ {:,.2f}'.format(threshold, general))

print('Diferença R$ {:,.2f}'.format(general-top))


# ## Exibição dos Melhores Avaliados

# In[ ]:


# # Testando para ver se a concatenação manteve os indextes, mantedo assim o cruzamento correto de dados
# sreid = 230246
# display(df_ALL_sre[df_ALL_sre['sre_id']==sreid])
# display(final_full_sre_ranking[final_full_sre_ranking['sre_id']==sreid])
#----------------------------------------------------------------------------------

# Filtrando o Olimpo: Carros com o Ranking CarHunt superior a nota 8, sem serem outliers  

olimpo_CarHunt = final_full_sre_ranking[(final_full_sre_ranking['Ranking_CarHunt']>9) & (final_full_sre_ranking['outlier_flag']==0)]

for bt in list(dict.fromkeys(olimpo_CarHunt['sre_body_type'])):
    
    print(bt,':')

    print(olimpo_CarHunt[olimpo_CarHunt['sre_body_type']==bt])


# ## Visualização do modelo de precificação

# In[ ]:


cat = 'Monovolume'

target_col = 'sre_value'

visualize_best_model(CarHuntML_models[cat], results_rank_df[results_rank_df['sre_body_type']==cat], target_col)


# # COMPARAÇÃO

# ### Clsuterizando carros similares

# In[ ]:


# Clsuterizando carros similares

import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

import os
import multiprocessing
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())


#Função auxiliar para determinar o núemro ideal de clsuters 
def elbow_method(data):
    
    #realizando a escala dos dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    min_clusters = 2
    
    inercia = []
    max_clusters = min_clusters + int(min(data_scaled.shape[0], data_scaled.shape[1])/1.5)
    range_clusters = range(min_clusters, max_clusters + 1)
    
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(data_scaled)
        inercia.append(kmeans.inertia_)

        
    def encontrar_ponto_mais_agudo(x_range, y_values):
        x = np.array(x_range)
        y = np.array(y_values)

        diff_y = np.diff(y)
        diff_x = np.diff(x)

        tangentes = diff_y / diff_x

        indice_ponto_agudo = np.argmin(tangentes) + 1

        return x[indice_ponto_agudo]
    
    
    
    def inflexao_segunda_derivada(x,y):
        # Calcular a segunda derivada
        second_derivative = np.gradient(np.gradient(y, x), x)

        # Encontrar o índice do ponto mais agudo
        index_max_second_derivative = np.argmax(second_derivative)

        # Recuperar as coordenadas do ponto mais agudo
        x_max_second_derivative = x[index_max_second_derivative]
        y_max_second_derivative = y[index_max_second_derivative]
        
        return x_max_second_derivative
    
            
        
    num_clusters = int((encontrar_ponto_mais_agudo(range_clusters, inercia) + inflexao_segunda_derivada(range_clusters, inercia))/2)
    
    #contando exceções de regras (colunas em que quase todas as linhas caíram, exceto até 10% das linhas)
    #Elas são clsuters novos
    count = ((data == 0).sum(axis=0) <= int(len(data)*0.1)).sum()
    num_clusters += count
    
    
    fig = px.line(x=range_clusters, y=inercia, title='Elbow')
    fig.add_trace(go.Scatter(x=[num_clusters], y=[inercia[num_clusters - min_clusters]], mode='markers', marker=dict(color='red'), name='Optimal K'))
    fig.update_xaxes(title='Número de Clusters')
    fig.update_yaxes(title='Inércia')
    pio.show(fig)

    print('Número ideal de clusters calculado: {} Clusters'.format(num_clusters))

    return num_clusters



def clusterize(data):
    
    #replicando os dados orginais para não executar alterações indevidas por acidente
    dataframe = data.copy()
    
    #eliminando todas as linhas em que todos os valores são zero
    dataframe = (dataframe.loc[~(dataframe.select_dtypes(include='number') == 0).all(axis=1)]).reset_index(drop=True)
    
    # Determinar o número mínimo de clusters utilizando o método do cotovelo
    num_clusters = elbow_method(dataframe.select_dtypes(include='number'))
        
    # Realizar clusterização com K-means
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(dataframe.select_dtypes(include='number'))

    # Adicionar rótulos de cluster como coluna no dataframe original
    dataframe['cluster'] = kmeans.labels_

    return dataframe


# CHAMADA DA CLSUTERIZAÇÃO
# --------------------------------------------

clustered_base_v_toML = clusterize(detc_base_v_toML)

clustered_base_v_toML


# ### PREÇO Calssificação de Ofertas
# 
# Escala de preços comparativos com 5 categorias:
# 
# Posicionamento de Preço em Comparação:
# 
# - Super Preço:
# Preço muito abaixo da média para modelos similares.
# 
# - Bom Preço:
# Preço abaixo da média para modelos similares.
# 
# - Preço Justo:
# Preço em linha com a média para modelos similares.
# 
# - Preço Elevado:
# Preço ligeiramente acima da média para modelos similares.
# 
# - Caro:
# Preço significativamente acima da média para modelos similares.
# 
# Essa escala simplificada oferece uma visão rápida e clara do posicionamento de preço em comparação com modelos semelhantes, permitindo que os usuários avaliem rapidamente se o carro está sendo oferecido a um preço atrativo ou se está na faixa mais elevada em termos de valor.

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px

# Function to display histograms for each cluster
def define_cluster_histograms(df, target_column, cluster_column, bin_tags):
    
    clusters = []
    
    for cluster_id in df[cluster_column].unique():
        
        cluster_data = df[df[cluster_column] == cluster_id]
        #display(cluster_data)
        
        if len(list(dict.fromkeys(cluster_data[target_column])))>=len(bin_tags):

            # Calculate histogram center and standard deviation
            hist_center = cluster_data[target_column].mean()
            std_dev = cluster_data[target_column].std()
            
            # Define the bin edges based on standard deviation
            bin_edges = np.linspace(hist_center - (2*std_dev), hist_center + (2*std_dev), len(bin_tags)-1)

#             # Extract the number of bins from the bin_edges list
#             num_bins = int(round(len(cluster_data) ** 0.5))

#             # Create histogram using Plotly Express
#             fig = px.histogram(cluster_data, x=target_column, nbins=num_bins, title='Histogram with {} Equal Sliced Areas'.format(len(bin_tags)))

#             # Add vertical lines for bin edges
#             for edge in bin_edges:
#                 fig.add_shape(
#                     type='line',
#                     x0=edge,
#                     x1=edge,
#                     y0=0,
#                     y1=100,
#                     line=dict(color='red', width=2)
#                 )

#             fig.show()

            # Categorize values into bins
            # 1a Bin é a mais extrema em valores superiores, a última Bin é a mais extrema em valores inferiores
            cluster_data['bin_category'] = pd.cut(cluster_data[target_column], bins=[-np.inf] + list(bin_edges) + [np.inf], labels=bin_tags)

        else:
            print('Could not categorize cluster {}'.format(cluster_id))
        
        clusters.append(cluster_data)            
    
    return pd.concat(clusters)

# --------------------------------------------------
# Definindo as tags

price_tags = ['Super Preço','Bom Preço','Preço Justo','Preço Elevado','Caro']
target_column = 'sre_value'

priced_clustered_base_v_toML = define_cluster_histograms(clustered_base_v_toML, target_column, 'cluster', price_tags)

# Display the resulting dataframe
print(priced_clustered_base_v_toML)


# ### Exibindo super ofertas

# In[ ]:


super_orefas_df = priced_clustered_base_v_toML[priced_clustered_base_v_toML['bin_category']=='Super Preço']

#Retraduzindo os dados mapeados:
for col in [c for c in base_v_colunas_mapeadas if c in list(super_orefas_df.columns)]:
    
    # encontrando o dicionário de mapeamento 
    mapping_dict = next(item.get(col) for item in base_v_maps if col in item)
        
    super_orefas_df[col] = super_orefas_df[col].map({v: k for k, v in mapping_dict.items()})

    
# ReCriando coluna de ano
super_orefas_df['sre_year'] = datetime.datetime.now().year-super_orefas_df['sre_age']
    
super_orefas_df


# # Ranking

# In[ ]:


# CHAMADA
def save_mongo_ranking(final_full_sre_ranking,db, collection):
    client = MongoClient('mongodb+srv://gabrielaraujon96:D3umaten0ve@carhunt.sweobl3.mongodb.net/?retryWrites=true&w=majority&appName=carhunt')
    db = client['carhunt']
    collection = db['car_ranking']
    data = final_full_sre_ranking.to_dict(orient= 'records')
    collection.insert_many(data)
    print("Dados inserido!")

results_rank_df, CarHuntML_models = ML_ranking_gen2(df_ALL_sre_toML, train_method='arbit')

# Model Persistence
# Save the best models to disk for later use
joblib.dump(CarHuntML_models, 'CarHuntML_models_{}.pkl'.format(datetime.datetime.now().strftime("%d%b%Y-%Hh%M")))    



#Retraduzindo os dados mapeados:
for col in [c for c in colunas_mapeadas_df_ALL_sre_maps if c in list(results_rank_df.columns)]:

    # encontrando o dicionário de mapeamento 
    mapping_dict = next(item.get(col) for item in df_ALL_sre_maps if col in item)

    results_rank_df[col] = results_rank_df[col].map({v: k for k, v in mapping_dict.items()})
    
    
# ReCriando coluna de ano
# results_rank_df['sre_year'] = datetime.datetime.now().year-results_rank_df['sre_age']

# ReColocanod colunas anteriormente ignoradas na tratativa de ML
final_full_sre_ranking = pd.concat([results_rank_df.sort_index(), df_ALL_sre[drop_cols_df_ALL_sre].sort_index()], axis=1)

# Ordenando campeões -----
final_full_sre_ranking = final_full_sre_ranking.sort_values("Ranking_CarHunt", ascending=False)

# Reorganizando as colunas
ml_cols=['outlier_flag','predicted_ML_value','price','diff_ml','Ranking_CarHunt']
final_full_sre_ranking = final_full_sre_ranking[[c for c in list(final_full_sre_ranking.columns) if c not in ml_cols]+ml_cols]

save_mongo_ranking(final_full_sre_ranking, 'carhunt', 'car_ranking')

print(final_full_sre_ranking)

df_ALL_sre_maps[-1]['body'].items()

# HISTORY DO RANKING
import pandas as pd
import os

def get_mongodb_history(records):
    try:
        # Configurar a conexão com o MongoDB
        client = MongoClient('mongodb+srv://gabrielaraujon96:D3umaten0ve@carhunt.sweobl3.mongodb.net/?retryWrites=true&w=majority&appName=carhunt')
        db = client['carhunt']
        collection = db['car_history']
    except Exception as e:
            print(f"Ocorreu um erro ao enviar os dados para o MongoDB: {e}")
            
#historico_path = 'historico_ranking.csv'
def save_mongo_history_ranking(data,db, collection):
    client = MongoClient('mongodb+srv://gabrielaraujon96:D3umaten0ve@carhunt.sweobl3.mongodb.net/?retryWrites=true&w=majority&appName=carhunt')
    db = client['carhunt']
    collection = db['car_history']
    data = historico_ranking.to_dict(orient= 'records')
    collection.insert_many(data)


if os.path.exists(historico_path):
    historico_ranking = pd.read_csv(historico_path)
else:
    historico_ranking = pd.DataFrame(columns=['brand', 'model', 'city', 'state', 'km', 'transmission', 'body', 'year',
                                         'link', 'image', 'source', 'type', 'outlier_flag', 'predicted_ML_value', 
                                         'price', 'diff_ml', 'Ranking_CarHunt', 'rank_date'])


final_full_sre_ranking['rank_date'] = pd.to_datetime('today')

# Selecionar os primeiros 50 carros do ranking e adiciona ao histórico
add_rank = final_full_sre_ranking.head(50)


if 'Ranking_CarHunt' in historico_ranking.columns:
    historico_ranking = add_rank[add_rank['Ranking_CarHunt']>8]
    #historico_ranking = pd.concat([historico_ranking, add_rank_filtered], ignore_index=True)

if 'link' in historico_ranking.columns:
    # Adiciona novos carros ao histórico
    #historico_ranking = pd.concat([historico_ranking, add_rank_filtered], ignore_index=True)
    # Remove duplicatas baseadas na coluna 'link'
    historico_ranking = historico_ranking.drop_duplicates(subset=['link'], keep='last')

# Ranking_CarHunt em ordem decrescente
historico_ranking = historico_ranking.sort_values(by='Ranking_CarHunt', ascending=False)

# Salvar o histórico atualizado
#historico_ranking.to_csv(historico_path, index=False)

    
print("Histórico atualizado com novos carros.")

save_mongo_history_ranking(historico_ranking, 'carhunt', 'car_history')

print(historico_ranking.head(100))  


# # Criar Rotina
# 

# In[ ]:


# import schedule

# def tarefa_atualizacao():
#     final_full_sre_ranking = ML_ranking_gen2()
#     atualizar_historico(final_full_sre_ranking)
#     print("Atualização concluída.")


# schedule.every(1).hours.do(tarefa_atualizacao)

# while True:
#     schedule.run_pending()
#     time.sleep(1)


# import schedule
# import time
# def rotina_atualizacao(intervalo_segundos):
#     while True:
#         print("Iniciando atualização...")
#         print("DataFrame atual:")
#         # print(final_full_sre_ranking)  # Debugging: Exibe o DataFrame
#         final_full_sre_ranking = ML_ranking_gen2(final_full_sre_ranking)
#         save_mongo_history_ranking(final_full_sre_ranking)
#         print("Atualização concluída!")
#         time.sleep(intervalo_segundos)
# rotina_atualizacao(3600)

