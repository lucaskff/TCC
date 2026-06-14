# simulation.py
import numpy as np
import pandas as pd

class StochasticSimulationModule:
    """
    Módulo de validação preliminar (legado). Gera séries temporais sintéticas
    via Cadeias de Markov (precipitação) e modelos autorregressivos AR(1) (temperatura).
    """
    @staticmethod
    def simular_cadeia_markov_chuva(n_amostras, p_seco_chuvoso=0.15, p_chuvoso_seco=0.40):
        # Estado 0: Seco, Estado 1: Chuvoso (Baseado em médias regionais Epagri)
        estados = [0]
        for _ in range(1, n_amostras):
            estado_atual = estados[-1]
            if estado_atual == 0:
                proximo = np.random.choice([0, 1], p=[1 - p_seco_chuvoso, p_seco_chuvoso])
            else:
                proximo = np.random.choice([0, 1], p=[p_chuvoso_seco, 1 - p_chuvoso_seco])
            estados.append(proximo)
        return np.array(estados) * np.random.exponential(scale=5.0, size=n_amostras)

    @staticmethod
    def simular_temperatura_ar1(n_amostras, media_clima=18.5, phi=0.85, sigma=1.2):
        # Modelo Autorregressivo Gaussiano de ordem 1: X_t = c + phi * X_{t-1} + erro
        temp = [media_clima]
        erros_gaussianos = np.random.normal(0, sigma, n_amostras)
        c = media_clima * (1 - phi)
        
        for t in range(1, n_amostras):
            valor_t = c + phi * temp[-1] + erros_gaussianos[t]
            temp.append(valor_t)
        return np.array(temp)