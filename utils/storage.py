import pickle

class ValueFunctionStorage:
    def __init__(self, filename="value_function.pkl"):
        self.filename = filename
    def save(self,agent):
        """
        Guarda el value_function del agente en un archivo.
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(agent.value_function, f)
        print(f"Value function saved to {self.filename}")
    def load(self,agent):
        """
        Carga el value_function del agente desde un archivo.
        Si el archivo no existe, inicializa un diccionario vacío.
        """
        try:
            with open(self.filename, 'rb') as f:
                agent.value_function = pickle.load(f)
            print(f"Value function loaded from {self.filename}")
        except FileNotFoundError:
            print(f"No value function found at {self.filename}")
            agent.value_function = {} # Inicializa un diccionario vacío si no se encuentra el archivo