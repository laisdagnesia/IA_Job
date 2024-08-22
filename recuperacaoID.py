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

#final_full_sre_ranking.values[100]

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