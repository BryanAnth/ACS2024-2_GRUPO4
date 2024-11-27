import pickle

class ValueFunctionStorage:
    def __init__(self, filename="value_function.pkl"):
        self.filename = filename
    def save(self,agent):
        with open(self.filename, 'wb') as f:
            pickle.dump(agent.value_function, f)