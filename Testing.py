import unittest

from KMeans import Kmeans
from PAM import PAM
from Data import Data, DataConverter
import pandas as pd
import numpy as np
from Cluster import KNN

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_determine_closest_in_dictionary(self):
        """
        Test for PAM function
        """
        test_dict = {  # arbitrary set dictionary to test
            "1": 55, "2": 22, "3": 11, "4": 1
        }
        result_list = PAM.order_by_dict_values(test_dict)  # returns list of tuples (not dictionary)
        result = result_list[0][1]  # obtain first value dictionary is ordered by
        self.assertEqual(result, 1, "minimum value")  # determines if it is smallest element
        self.assertNotEqual(result, 55, "Not Max Val")

    def test_PAM_super(self):
        """
        Test if inheritance is working properly
        test if euclidean is working properly for both single and dictionary returns
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=10)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        pam = PAM(k_val=2, data_instance=data)  # create PAM instance to check super
        pam_train_data = pam.train_df  # train_df is an instance from parent class
        self.assertTrue(pam_train_data.equals(data.train_df), "Same data")
        self.assertFalse(pam_train_data.equals(data.test_df))

        row_c, row_q = np.split(pam_train_data, 2)  # split the same data into size of
        _, row_comp = next(row_c.copy().iterrows())  # get a row
        _, row_query = next(row_q.copy().iterrows())  # get a row
        dict_dist = pam.get_euclidean_distance_dict(row_query, row_c)

        single_distance = pam.get_euclidean_distance(row_query, row_comp)  # get distance
        self.assertTrue(isinstance(single_distance, float))  # check the it returns a float
        self.assertTrue(isinstance(dict_dist, dict))  # check if it is a dictionary

    def test_KNN(self):
        """
        Test if KNN is returning a class
        :return:
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=10)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        k_val = 5
        knn = KNN(k_val, data)
        nearest = knn.perform_KNN(k_val, df.iloc[1], data.train_df)
        print(nearest)

    def test_euclidean(self):
        """
        Test if euclidean distance is working
        :return:
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=10)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        knn = KNN(5, data)
        print(knn.get_euclidean_distance(df.iloc[1], df.iloc[2]))

    def test_edit(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)
        df = data.df.sample(n=50)
        data.split_data(data_frame=df)
        knn = KNN(5, data)
        knn.edit_data(data.train_df, 5, data.test_df, data.label_col)

    def test_data_conversion(self):
        data = Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None), 8)
        df = data.df.sample(n=50)
        data.split_data(data_frame=df)
        print(data.train_df)
        converter = DataConverter()
        print(converter.convert_data_to_original(data.train_df))

    def test_data_conversion_to_original(self):
        data = Data('car', pd.read_csv(r'data/car.data', header=None), 8)
        df = data.df.sample(n=50)
        data.split_data(data_frame=df)
        print(data.train_df)
        converter = DataConverter()
        print(converter.convert_data_to_original(data.train_df))

    def test_k_means(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        data_copy = data
        df = data.df.sample(n=4177)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        k_val = 5
        knn = KNN(k_val, data)
        # nearest = knn.perform_KNN(k_val, df.iloc[1], data.train_df)
        kmeans = Kmeans(k_val, data)
        kmeans.k_means(data.test_df, 5)

if __name__ == '__main__':
    unittest.main()
