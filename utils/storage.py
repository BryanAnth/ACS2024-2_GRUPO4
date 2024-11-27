import pickle

class ValueFunctionStorage:
    def __init__(self, filename="value_function.pkl"):
        self.filename = filename