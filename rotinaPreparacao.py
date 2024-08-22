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