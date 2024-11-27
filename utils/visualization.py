import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple
import ipywidgets as widgets
from IPython.display import display

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

    def plot_state_analysis(self, states: List[Tuple], actions: List[int]):
        """
        Analiza las distribuciones de estados y acciones.
        
        Args:
            states: Lista de estados visitados
            actions: Lista de acciones tomadas
        """
        states = np.array(states)
        
        plt.figure(figsize=(15, 10))
        
        # Distribución de posiciones del carro
        plt.subplot(221)
        plt.hist(states[:, 0], bins=50)
        plt.xlabel('Cart Position')
        plt.ylabel('Frequency')
        plt.title('Cart Position Distribution')
        
        # Distribución de ángulos del péndulo
        plt.subplot(222)
        plt.hist(states[:, 2], bins=50)
        plt.xlabel('Pole Angle')
        plt.ylabel('Frequency')
        plt.title('Pole Angle Distribution')
        
        # Diagrama de fase
        plt.subplot(223)
        plt.scatter(states[:, 0], states[:, 1], c=actions, alpha=0.1)
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title('Phase Diagram (Position vs Velocity)')
        
        plt.tight_layout()
        plt.show()

    def plot_convergence_analysis(self, rewards: List[float], lengths: List[int]):
        """
        Analiza la convergencia del entrenamiento.
        
        Args:
            rewards: Lista de recompensas
            lengths: Lista de duraciones de episodios
        """
        plt.figure(figsize=(15, 5))
        
        # Evolución de la recompensa
        plt.subplot(131)
        plt.plot(rewards)
        plt.axhline(y=195, color='r', linestyle='--', label='Success Threshold')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward Evolution')
        plt.legend()
        
        # Duración de episodios
        plt.subplot(132)
        plt.plot(lengths)
        plt.xlabel('Episode')
        plt.ylabel('Episode Length')
        plt.title('Episode Length Evolution')
        
        # Histograma 2D de recompensa vs duración
        plt.subplot(133)
        plt.hist2d(lengths, rewards, bins=50)
        plt.xlabel('Episode Length')
        plt.ylabel('Total Reward')
        plt.title('Reward vs Episode Length')
        plt.colorbar(label='Frequency')
        
        plt.tight_layout()
        plt.show()

    def create_training_report(self, results: Dict):
        """
        Genera un reporte completo del entrenamiento.
        
        Args:
            results: Diccionario con todos los resultados del entrenamiento
        """
        print("=== Training Report ===")
        print(f"Total Episodes: {len(results['rewards'])}")
        print(f"Best Reward: {max(results['rewards']):.2f}")
        print(f"Average Reward (last 100): {np.mean(results['rewards'][-100:]):.2f}")
        print(f"Success Rate: {np.mean([r >= 195 for r in results['rewards'][-100:]]):.2%}")
        print("\nGenerating visualizations...")
        
        # Generar todas las visualizaciones
        self.plot_training_progress(results['rewards'])
        self.plot_state_analysis(results['states'], results['actions'])
        self.plot_convergence_analysis(results['rewards'], results['lengths'])
        
        print("\nReport generation complete.")
    
    def plot_interactive_analysis(results: Dict):
        """
        Crea una visualización interactiva de los resultados (requiere ipywidgets).
        
        Args:
            results: Diccionario con resultados del entrenamiento
        """
        try:        
            def update_plot(episode_range):
                plt.figure(figsize=(10, 5))
                plt.plot(results['rewards'][episode_range[0]:episode_range[1]])
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title(f'Training Progress (Episodes {episode_range[0]} to {episode_range[1]})')
                plt.show()
            
            episode_slider = widgets.IntRangeSlider(
                value=[0, len(results['rewards'])],
                min=0,
                max=len(results['rewards']),
                step=1,
                description='Episodes:',
                continuous_update=False
            )
            
            widgets.interactive(update_plot, episode_range=episode_slider)
            
        except ImportError:
            print("Interactive plotting requires ipywidgets. Please install it with:")
            print("pip install ipywidgets")