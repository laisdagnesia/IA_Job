# CHAMADA
def save_mongo(final_full_sre_ranking,db, collection):
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

save_mongo(final_full_sre_ranking, 'carhunt', 'car_ranking')

display(final_full_sre_ranking)

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
def save_mongo_ranking(data,db, collection):
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

save_mongo_ranking(historico_ranking, 'carhunt', 'car_history')

print(historico_ranking.head(100))  