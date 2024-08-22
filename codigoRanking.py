from preparacaoDadosDasOfertasML import base_v_toML,base_v_maps,base_v_colunas_mapeadas
from deteccaoDeAnomalias import detect_outliers
from modeloRegressao import select_best_regression_model 
#from visualizacaoModelo import best_model,DecisionTreeRegressor,HalvingGridSearchCV

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

    # Conversão obrigatória dos valores dos carros
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
