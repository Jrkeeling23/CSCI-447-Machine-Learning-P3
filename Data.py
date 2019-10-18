import pandas as pd
import numpy as np


class Data:
    def __init__(self, name, df, label_col):
        self.name = name
        self.df = df
        self.test_df = None
        self.train_df = None
        self.label_col = label_col
        self.k_dfs = None

    @staticmethod
    def get_row_size(df):
        """
        call to get any df row size
        """
        return df.shape[0]

    @staticmethod
    def get_col_size(df):
        """
        call to get any df column size
        """
        return df.shape[1]

    def split_data(self, data_frame, train_percent=.8):
        """
        splits the data according to the train percent.
        :return:
        """
        # TODO:if dataframe or train_percent are empty, use if statement to split data in a universal way
        # use numpys split with pandas sample to randomly split the data
        self.train_df, self.test_df = np.split(data_frame.sample(frac=1), [int(train_percent * len(data_frame))])

    def split_k_fold(self, k_val, dataset):
        """
        Split data into list of K different parts
        :param k_val: k value to set size of folds.
        :return: list of lists where arranged as follows [[train,test], [train, test]] repeated k times
        where train is traing data (index 0) and test is testing data (index 1)
        """
        k__split_data = np.array_split(dataset, k_val)  # splits dataset into k parts
        # now we need to split up data into 1 list and k others combined into 1 list for test/train
        test_train_sets = []
        temp_list = [None] * 2
        length = len(k__split_data)
        # create these new lists and add them to test_train_sets
        for i in range(length):  # go through every split list
            # APPARENTLY PYTHON DEVS THOUGHT IT WAS A GOOD FUCKING IDEA TO MAKE LISTS THAT HAVE DIFFERENT NAMES BOTH
            # REMOVE VALS WHEN THE REMOVE FUNCTION IS APPLIED TO ONE OF THEM.   WHY GOD WHY
            data_to_combine = np.array_split(dataset, k_val)
            temp_list[0] = k__split_data[i]
            del data_to_combine[i]
            temp_list[1] = pd.concat(data_to_combine)
            test_train_sets.append(temp_list)

            # TODO: I don't think we need  i +=1, but we can check

    def k_fold(self, df, k_val):
        # TODO: Test this and see if the data is split accordingly and no elements in a df have are equal to another df
        column_size = self.get_col_size(df)  # get column size
        group = int(column_size / k_val)  # get number of data points per group
        grouped_data_frames = []
        for g, k_df in df.groupby(np.arange(len(column_size)) // group):
            grouped_data_frames.append(k_df)
        return grouped_data_frames

    def quick_setup(self):
        self.split_data(train_percent=0.8)
