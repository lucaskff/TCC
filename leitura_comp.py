import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

ARQUIVO_ESTACAO = os.path.join(DIRETORIO_ATUAL, "dados_estacao.csv")
ARQUIVO_GLOBAL = os.path.join(DIRETORIO_ATUAL, "temp_global.csv")
DB_NAME = os.path.join(DIRETORIO_ATUAL, "meteorologia.db")

print("1. Carregando e tratando os dados reais da Estação CEPLAN...")
try:
    df_local = pd.read_csv(ARQUIVO_ESTACAO, skiprows=1, sep=';', decimal=',', encoding='cp1252')
except FileNotFoundError:
    print(f"\n[ERRO]: O arquivo 'dados_estacao.csv' não foi encontrado na pasta: {DIRETORIO_ATUAL}")
    exit()

df_local['datetime'] = pd.to_datetime(df_local['Data e Hora'], dayfirst=True, errors='coerce')
df_local = df_local.dropna(subset=['datetime', 'Temperatura Méd.'])

print("2. Carregando os dados globais reais do Open-Meteo...")
try:
    df_global = pd.read_csv(ARQUIVO_GLOBAL, skiprows=3, encoding='utf-8')
    df_global['datetime'] = pd.to_datetime(df_global['time'], errors='coerce')
except UnicodeDecodeError:
    df_global = pd.read_csv(ARQUIVO_GLOBAL, skiprows=3, encoding='cp1252')
    df_global['datetime'] = pd.to_datetime(df_global['time'], errors='coerce')
except FileNotFoundError:
    print(f"\n[ERRO]: O arquivo 'temp_global.csv' não foi encontrado na pasta: {DIRETORIO_ATUAL}")
    exit()

print("3. Sincronizando séries temporais por data e hora (Merge)...")
df_merged = pd.merge(df_local, df_global, on='datetime', how='inner')
df_merged = df_merged.sort_values(by='datetime').reset_index(drop=True)

if df_merged.empty:
    print("[ERRO]: A sincronização retornou zero linhas. Verifique se as datas coincidem.")
    exit()

print(f"-> Sucesso! {len(df_merged)} registros horários alinhados perfeitamente.")

print("4. Executando algoritmo de aprendizado incremental...")
temp_ajustada = []
w1, w2 = 0.5, 0.5  

for index, row in df_merged.iterrows():
    t_global = float(row['temperature_2m (°C)'])
    t_local = float(row['Temperatura Méd.'])
    
    t_ajustada = (w1 * t_global) + (w2 * t_local)
    temp_ajustada.append(t_ajustada)
    
    erro_g = abs(t_global - t_local)
    erro_l = 0.15  
    inv_g = 1.0 / (erro_g + 1e-5)
    inv_l = 1.0 / (erro_l + 1e-5)
    
    w1 = inv_g / (inv_g + inv_l)
    w2 = 1.0 - w1

df_merged['temp_ajustada'] = temp_ajustada

mae_global = np.mean(np.abs(df_merged['temperature_2m (°C)'] - df_merged['Temperatura Méd.']))
mae_ajustado = np.mean(np.abs(df_merged['temp_ajustada'] - df_merged['Temperatura Méd.']))
rmse_global = np.sqrt(np.mean((df_merged['temperature_2m (°C)'] - df_merged['Temperatura Méd.'])**2))
rmse_ajustado = np.sqrt(np.mean((df_merged['temp_ajustada'] - df_merged['Temperatura Méd.'])**2))

print(f"\n===== RESULTADOS REAIS OBTIDOS =====")
print(f"MAE original Open-Meteo: {mae_global:.2f}°C  | MAE Corrigido: {mae_ajustado:.2f}°C")
print(f"RMSE original Open-Meteo: {rmse_global:.2f}°C | RMSE Corrigido: {rmse_ajustado:.2f}°C")

conn = sqlite3.connect(DB_NAME)
df_merged[['datetime', 'temperature_2m (°C)', 'Temperatura Méd.', 'temp_ajustada']].to_sql('validacao_real', conn, if_exists='replace', index=False)
conn.close()

plt.rcParams["figure.figsize"] = (12, 6)
df_merged['data_hora_str'] = df_merged['datetime'].dt.strftime('%d/%m %H:%M')

plt.plot(df_merged['data_hora_str'], df_merged['temperature_2m (°C)'], label='Previsão Global Real (Open-Meteo)', color='red', alpha=0.6, linestyle='--')
plt.plot(df_merged['data_hora_str'], df_merged['Temperatura Méd.'], label='Estação Local Real (CEPLAN)', color='blue', alpha=0.5)
plt.plot(df_merged['data_hora_str'], df_merged['temp_ajustada'], label='Resultado Ajustado (Modelo Ponderado)', color='green', linewidth=2)

plt.title('Comparativo Temporal de Temperatura e Ajuste de Viés Local', fontsize=12, fontweight='bold')
plt.xlabel('Linha do Tempo (Amostragem Sincronizada Horária)', fontsize=10)
plt.ylabel('Temperatura (°C)', fontsize=10)

plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

CAMINHO_GRAFICO = os.path.join(DIRETORIO_ATUAL, 'grafico_vies_real_meteorologia.png')
plt.savefig(CAMINHO_GRAFICO, dpi=300)
print(f"\n-> Gráfico gerado e salvo em: {CAMINHO_GRAFICO}")
plt.show()