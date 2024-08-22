cat = 'Monovolume'

target_col = 'sre_value'

visualize_best_model(CarHuntML_models[cat], results_rank_df[results_rank_df['sre_body_type']==cat], target_col)