import unittest
from PAM import PAM
from Data import Data
import pandas as pd
import numpy as np
from Cluster import KNN
import collections


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

    def test_assigning_to_medoids(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=100)  # minimal data frame
        self.assertEqual(df.shape[0], 100)
        data.split_data(data_frame=df)  # sets test and train data
        self.assertEqual((data.train_df.shape[0] + data.test_df.shape[0]), 100)
        pam = PAM(k_val=3, data_instance=data)  # create PAM instance to check super
        pam.current_medoids = pam.assign_random_medoids(pam.train_df, pam.k)
        size = len(pam.current_medoids)
        self.assertEqual(size, 3)
        pam.assign_data_to_medoids(pam.train_df, pam.current_medoids)  # set medoids list equal to return value
        size = 0  # reset size variable
        for medoid in pam.current_medoids:
            size += medoid.encompasses.shape[0]  # get number of data points assigned to a medoid
        self.assertEqual(size, pam.train_df.shape[0] - 3)  # make sure all data_points are assigned to a medoid
        bool_type = pam.compare_medoid_costs(pam.current_medoids[0], pam.current_medoids[1])
        # new_med_list = pam.perform_pam(pam.train_df, pam.current_medoids)
        self.assertIsInstance(bool_type, bool)  # shows whether it swapped or not
        # self.assertNotEqual(new_med_list, pam.current_medoids)

    def test_medoid_swapping(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=500)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        pam = PAM(k_val=3, data_instance=data)  # create PAM instance to check super
        pam.current_medoids, pam.current_medoid_indexes = pam.assign_random_medoids(pam.train_df, pam.k)

        print(
            "__________________________________________________\n__________ Begin Finding Better Medoids "
            "__________\n__________________________________________________\n")
        print("Initial Medoid Indexes: ", pam.current_medoid_indexes)
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
        checker = False
        temp1 = pam.current_medoid_indexes
        temp2 = pam.current_medoid_indexes
        while True:
            if checker:
                temp1 = pam.current_medoid_indexes
                checker = False
            else:
                temp2 = pam.current_medoid_indexes
                checker = True

            pam.assign_data_to_medoids(pam.train_df, pam.current_medoids)  # do this every time.
            changed_list, indexes = pam.compare_medoids(pam.current_medoids.copy(), pam.train_df)
            if compare(temp1, indexes) or compare(temp2, indexes):
                break
            else:
                print("\nInitial Medoid list: ", pam.current_medoid_indexes, "\nReturned Medoid List: ", indexes,
                      " Cache[0]: ", temp1, " Cache[1]: ", temp2)
                pam.current_medoids = changed_list
                pam.current_medoid_indexes = indexes
                print("\n---------- Continue Finding Better Medoids ----------")

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


if __name__ == '__main__':
    unittest.main()
