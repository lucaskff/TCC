# visualization.py
import os
import matplotlib.pyplot as plt

class VisualizationModule:
    """
    Subsistema responsável pela renderização e exportação automatizada
    dos relatórios gráficos comparativos de desempenho térmico.
    """
    @staticmethod
    def plotar_comparativo_temporal(df, caminho_saida):
        plt.rcParams["figure.figsize"] = (12, 6)
        df_plot = df.copy()
        df_plot['data_hora_str'] = df_plot['datetime'].dt.strftime('%d/%m %H:%M')

        plt.plot(df_plot['data_hora_str'], df_plot['temperature_2m (°C)'], 
                 label='Previsão Global Real (Open-Meteo)', color='red', alpha=0.6, linestyle='--')
        plt.plot(df_plot['data_hora_str'], df_plot['Temperatura Méd.'], 
                 label='Estação Local Real (CEPLAN)', color='blue', alpha=0.5)
        plt.plot(df_plot['data_hora_str'], df_plot['temp_ajustada'], 
                 label='Resultado Ajustado (Modelo Ponderado)', color='green', linewidth=2)

        plt.title('Comparativo Temporal de Temperatura e Ajuste de Viés Local', fontsize=12, fontweight='bold')
        plt.xlabel('Linha do Tempo (Amostragem Sincronizada Horária)', fontsize=10)
        plt.ylabel('Temperatura (°C)', fontsize=10)

        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        plt.savefig(caminho_saida, dpi=300)
        plt.close()