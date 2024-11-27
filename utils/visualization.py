import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple
import ipywidgets as widgets

class VisualizationManager:
    def __init__(self):
        sns.set_theme()
        self.default_figsize = (12, 6)
    def plot_training_progress(self, rewards: List[float], title: str = "Training Progress"):
        """
        Visualiza el progreso del entrenamiento.
        
        Args:
            rewards: Lista de recompensas por episodio
            title: Título del gráfico
        """
        plt.figure(figsize=self.default_figsize)
        
        # Gráfico de recompensas
        plt.subplot(1, 2, 1)
        plt.plot(rewards, label='Reward per Episode', alpha=0.6)
        
        # Promedio móvil
        window_size = 100
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(moving_avg, label=f'Moving Average (n={window_size})', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'{title}\nRewards Over Time')
        plt.legend()
        
        # Histograma de recompensas
        plt.subplot(1, 2, 2)
        plt.hist(rewards, bins=50, alpha=0.7)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        
        plt.tight_layout()
        plt.show()

    def plot_parameter_comparison(self, parameter_results: Dict):
        """
        Compara resultados con diferentes parámetros.
        
        Args:
            parameter_results: Diccionario con resultados para diferentes parámetros
        """
        plt.figure(figsize=(15, 5))
        
        # Comparación de recompensas promedio
        plt.subplot(131)
        for param, results in parameter_results.items():
            plt.plot(results['moving_avg'], label=f'Param: {param}')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Parameter Comparison')
        plt.legend()
        
        plt.tight_layout()
        plt.show()