# ingestion.py
import os
import pandas as pd

class DataIngestionModule:
    """
    Subsistema responsável pela carga, tratamento ortográfico,
    tipagem temporal e sincronização (merge) das fontes heterogêneas.
    """
    def __init__(self, arquivo_estacao, arquivo_global):
        self.arquivo_estacao = arquivo_estacao
        self.arquivo_global = arquivo_global

    def carregar_dados_locais(self):
        if not os.path.exists(self.arquivo_estacao):
            raise FileNotFoundError(f"Arquivo local ausente: {self.arquivo_estacao}")
        
        df = pd.read_csv(self.arquivo_estacao, skiprows=1, sep=';', decimal=',', encoding='cp1252')
        df['datetime'] = pd.to_datetime(df['Data e Hora'], dayfirst=True, errors='coerce')
        return df.dropna(subset=['datetime', 'Temperatura Méd.'])

    def carregar_dados_globais(self):
        if not os.path.exists(self.arquivo_global):
            raise FileNotFoundError(f"Arquivo global ausente: {self.arquivo_global}")
        
        try:
            df = pd.read_csv(self.arquivo_global, skiprows=3, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.arquivo_global, skiprows=3, encoding='cp1252')
            
        df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
        return df.dropna(subset=['datetime'])

    def executar_sincronizacao(self):
        df_l = self.carregar_dados_locais()
        df_g = self.carregar_dados_globais()
        
        df_merged = pd.merge(df_l, df_g, on='datetime', how='inner')
        return df_merged.sort_values(by='datetime').reset_index(drop=True)