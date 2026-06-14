# main.py
import os
import sqlite3
# Importação dos submódulos do ecossistema estruturado
from ingestion import DataIngestionModule
from simulation import StochasticSimulationModule
from integration import OnlineLearningAdaptationModule
from visualization import VisualizationModule

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
ARQUIVO_ESTACAO = os.path.join(DIRETORIO_ATUAL, "dados_estacao.csv")
ARQUIVO_GLOBAL = os.path.join(DIRETORIO_ATUAL, "temp_global.csv")
DB_NAME = os.path.join(DIRETORIO_ATUAL, "meteorologia.db")
CAMINHO_GRAFICO = os.path.join(DIRETORIO_ATUAL, 'grafico_vies_real_meteorologia.png')

def executar_pipeline():
    print("=== PIPELINE COMPUTACIONAL METEOROLÓGICO MODULAR ===")
    
    # 1. Fase de Validação de Legado Estocástico (Simulação do TCC)
    print("\n[INFO]: Executando rotinas estocásticas de validação preliminar...")
    dados_sinteticos_chuva = StochasticSimulationModule.simular_cadeia_markov_chuva(n_amostras=100)
    dados_sinteticos_temp = StochasticSimulationModule.simular_temperatura_ar1(n_amostras=100)
    print(f"-> Geração de Cadeias de Markov e AR(1) concluída em cache histórico.")

    # 2. Ingestão de Dados Reais
    print("\n[Módulo 1]: Iniciando ingestão e sincronização de dados reais de superfície...")
    ingestao = DataIngestionModule(ARQUIVO_ESTACAO, ARQUIVO_GLOBAL)
    df_sincronizado = ingestao.executar_sincronizacao()
    print(f"-> Sucesso! {len(df_sincronizado)} registros horários alinhados.")

    # 3. Processamento via Online Learning
    print("\n[Módulo 2]: Executando algoritmo adaptativo de integração ponderada...")
    adaptador = OnlineLearningAdaptationModule()
    df_calculado = adaptador.processar_fluxo_incremental(df_sincronizado)

    # 4. Cálculo de Indicadores Estatísticos
    mae_g, mae_a, rmse_g, rmse_a, ganho = adaptador.calcular_metricas_erro(df_calculado)
    print(f"\n===== RELATÓRIO DE MÉTRICAS DE CONTROLE (n = {len(df_calculado)}) =====")
    print(f"MAE Open-Meteo:  {mae_g:.2f}°C | MAE Corrigido:  {mae_a:.2f}°C")
    print(f"RMSE Open-Meteo: {rmse_g:.2f}°C | RMSE Corrigido: {rmse_a:.2f}°C")
    print(f"Ganho de Acurácia no Período Avaliado: {ganho:.2f}%")

    # 5. Persistência de Dados em SQLite
    print("\n[Módulo 3]: Armazenando dados históricos de validação no banco de dados...")
    conn = sqlite3.connect(DB_NAME)
    df_calculado[['datetime', 'temperature_2m (°C)', 'Temperatura Méd.', 'temp_ajustada']].to_sql(
        'validacao_real', conn, if_exists='replace', index=False
    )
    conn.close()
    print(f"-> Banco SQLite '{os.path.basename(DB_NAME)}' atualizado com sucesso.")

    # 6. Módulo Gráfico
    print("\n[Módulo 4]: Renderizando relatórios e plots estatísticos...")
    VisualizationModule.plotar_comparativo_temporal(df_calculado, CAMINHO_GRAFICO)
    print(f"-> Gráfico salvo com sucesso em: {CAMINHO_GRAFICO}\n")
    print("================ PIPELINE FINALIZADO ================")

if __name__ == "__main__":
    executar_pipeline()