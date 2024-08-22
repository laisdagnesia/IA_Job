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