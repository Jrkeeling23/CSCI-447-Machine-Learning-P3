from Data import Data
import pandas as pd

def load_data():
    """
    loads the data (csv) files
    :return: list of Data instances
    """
    data_list = [Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8),
                 Data('car', pd.read_csv(r'data/car.data', header=None), 5),
                 Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None), 0),
                 Data('machine', pd.read_csv(r'data/machine.data', header=None), 0),
                 Data('forest_fires', pd.read_csv(r'data/forestfires.data', header=None), 12),
                 Data('wine', pd.read_csv(r'data/wine.data', header=None), 0)]
    return data_list


class Main:
    def __init__(self):
        self.data_list = load_data()
