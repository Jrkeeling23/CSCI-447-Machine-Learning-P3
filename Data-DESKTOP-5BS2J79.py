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

    def removeMissing(self):  # remove missing values, needs to be done for each array:  Methodology: generate average for attribute and use that
        # go through rows and check for NaN values, if nan change to average for that column (attribute)
        for row in range(self.df.iterrows()):  # go through rows
            for col in range(len(row)):  # go through cols
                true_val = self.df.at[row, col]  # var for holding col val (str int compare issues)
                if str(true_val) is None:  # check if it is a "missing" value
                    col_av = 0  # init for column average
                    # get the average of the column
                    no_str = [x for x in self.df[col].tolist() if not isinstance(x, str)]  # get a list of all non str values (not '?')
                    if len(no_str) > 0:  # make sun length of that list is > 0 to avoid divide by 0
                        # NOTE::  Only time we should see a length 0 with no nums would be in the class rows
                        # so for now ignoring the "else" case for this if as irrelevant
                        col_av = (sum(no_str) / len(no_str)).__round__()  # get the average of that list and round it
                    self.df.at[row, col] = col_av  # set our new row to the new average
        # set class data list equal to data_list from method

    def split_data(self, data_frame, train_percent=.8):
        """
        splits the data according to the train percent.
        :return:
        """
        # TODO:if dataframe or train_percent are empty, use if statement to split data in a universal way
        # use numpys split with pandas sample to randomly split the data
        # self.train_df = temp_df.sample(frac=0.75, random_state=0)
        # self.test_df= temp_df.split(self.train_df)
        self.train_df, self.test_df = np.split(data_frame.sample(frac=1), [int(.8 * len(data_frame))])
        # print("Train ", self.train_df.shape)
        # print("Test ", self.test_df.shape)

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
