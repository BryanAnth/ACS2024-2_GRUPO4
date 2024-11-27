import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple
import ipywidgets as widgets

class VisualizationManager:
    def __init__(self):
        sns.set_theme()
        self.default_figsize = (12, 6)