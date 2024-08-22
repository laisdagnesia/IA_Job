super_orefas_df = priced_clustered_base_v_toML[priced_clustered_base_v_toML['bin_category']=='Super Preço']

#Retraduzindo os dados mapeados:
for col in [c for c in base_v_colunas_mapeadas if c in list(super_orefas_df.columns)]:
    
    # encontrando o dicionário de mapeamento 
    mapping_dict = next(item.get(col) for item in base_v_maps if col in item)
        
    super_orefas_df[col] = super_orefas_df[col].map({v: k for k, v in mapping_dict.items()})

    
# ReCriando coluna de ano
super_orefas_df['sre_year'] = datetime.datetime.now().year-super_orefas_df['sre_age']
    
super_orefas_df