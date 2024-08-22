# import schedule

def tarefa_atualizacao():
    final_full_sre_ranking = ML_ranking_gen2()
    atualizar_historico(final_full_sre_ranking)
    print("Atualização concluída.")


schedule.every(1).hours.do(tarefa_atualizacao)

while True:
    schedule.run_pending()
    time.sleep(1)


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
