from preengenhariadedados import coerced_base_v
from rotinaPreparacao import prepare_data_for_ML

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