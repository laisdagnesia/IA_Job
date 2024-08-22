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