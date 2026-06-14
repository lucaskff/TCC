# integration.py
import numpy as np

class OnlineLearningAdaptationModule:
    def __init__(self, erro_sensor_local=0.15):
        self.erro_l = erro_sensor_local

    def processar_fluxo_incremental(self, df_merged):
        temp_ajustada = []
        w1, w2 = 0.5, 0.5  # Estado de pesos iniciais do sistema
        
        for index, row in df_merged.iterrows():
            t_global = float(row['temperature_2m (°C)'])
            t_local = float(row['Temperatura Méd.'])
            
            # Combinação linear ponderada adaptativa
            t_ajustada = (w1 * t_global) + (w2 * t_local)
            temp_ajustada.append(t_ajustada)
            
            # Cálculo recursivo do erro absoluto de passo único
            erro_g = abs(t_global - t_local)
            
            inv_g = 1.0 / (erro_g + 1e-5)
            inv_l = 1.0 / (self.erro_l + 1e-5)
            
            # Atualização dos pesos para t+1
            w1 = inv_g / (inv_g + inv_l)
            w2 = 1.0 - w1
            
        df_merged['temp_ajustada'] = temp_ajustada
        return df_merged

    @staticmethod
    def calcular_metricas_erro(df):
        mae_global = np.mean(np.abs(df['temperature_2m (°C)'] - df['Temperatura Méd.']))
        mae_ajustado = np.mean(np.abs(df['temp_ajustada'] - df['Temperatura Méd.']))
        
        rmse_global = np.sqrt(np.mean((df['temperature_2m (°C)'] - df['Temperatura Méd.'])**2))
        rmse_ajustado = np.sqrt(np.mean((df['temp_ajustada'] - df['Temperatura Méd.'])**2))
        
        melhoria_mae = ((mae_global - mae_ajustado) / mae_global) * 100
        return mae_global, mae_ajustado, rmse_global, rmse_ajustado, melhoria_mae