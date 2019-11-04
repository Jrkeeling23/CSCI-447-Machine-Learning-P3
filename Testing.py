import unittest

from KMeans import Kmeans
from PAM import PAM
from Data import Data, DataConverter
import pandas as pd
import numpy as np
from Cluster import KNN
import collections
from RBFNet import RBFReg
from RBFNet import RBFClass

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

    def test_medoid_swapping(self):
        """
        Just run to see values being swapped
        :return:
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=300)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        pam = PAM(k_val=3, data_instance=data)  # create PAM instance to check super
        index, distort, medoids = pam.perform_pam()

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

        print(type(df.iloc[1][1]))
        print(knn.get_euclidean_distance(df.iloc[1], df.iloc[2]))

    def test_edit(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)
        df = data.df.sample(n=50)
        data.split_data(data_frame=df)
        knn = KNN(5, data)
        knn.edit_data(data.train_df, 5, data.test_df, data.label_col)

    def test_data_conversion_to_numerical(self):
        data = Data('machine', pd.read_csv(r'data/machine.data', header=None), 8)
        df = data.df.sample(n=209)
        data.split_data(data_frame=df)
        print(data.test_df)
        # converter = DataConverter()
        # converter.convert_to_numerical(data.train_df)
        # converter.convert_to_numerical(data.test_df)

    def test_data_conversion_to_original(self):
        data = Data('forestfires', pd.read_csv(r'data/forestfires.data', header=None), 8)
        df = data.df.sample(n=209)
        data.split_data(data_frame=df)
        print(data.train_df)
        converter = DataConverter()
        converted = converter.convert_data_to_original(data.train_df)
        print(converted)
        mismatch = False
        dt = converter.convert_data_to_original(data.train_df.copy())
        for convert in converted.values:
            if convert not in dt.values:
                mismatch = True
        self.assertFalse(mismatch)

    def test_k_means(self):
        data = Data('machine', pd.read_csv(r'data/machine.data', header=None), 8)  # load data
        df = data.df.sample(n=209)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        k_val = 2
        knn = KNN(k_val, data)
        kmeans = Kmeans(k_val, data)
        clusters = kmeans.k_means(data.train_df, 10)
        converter = DataConverter()
        dt = converter.convert_data_to_original(data.train_df.copy())
        mismatch = False
        for cluster in clusters.values:
            if cluster not in dt.values:
                mismatch = True
        self.assertFalse(mismatch)

    def test_knn_condensed(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=350)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        cluster_obj = KNN(5, data)
        condensed_data = cluster_obj.condense_data(data.train_df)
        size_after = condensed_data.shape[0]
        size_prior = data.train_df.shape[0]
        self.assertGreater(size_prior, size_after)

    def test_discretize(self):
        data = Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None, skiprows=4), 8)
        data.regression_data_bins(4, quartile=True)
        data.regression_data_bins(4, quartile=False)

    def test_rbfReg(self):
            #data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 8)  # load data
            data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 11)  # load data
            df = data.df.sample(n=100)  # minimal data frame

            test_df = pd.read_csv('data/winequality-white.csv', header=None)
            test2_df = test_df.iloc[:101, :]
            #print(test2_df[11])
            print("Checking DF set")
            print(df[df.columns[-1]])

            cols = df.columns
            for col in cols:
                df[col] = df[col].astype(float)
            expected = df[df.columns[-1]]

            #print(expected[1])

            df = df.iloc[:, :-1]
            test2_df = test2_df.iloc[:, :-1]
            data.split_data(data_frame=df)  # sets test and train data
            # will have high error due to small dataset, but just a test to show how this works
            rbf = RBFReg(clusters=12, maxruns=1000)

            rbf.trainReg(data.train_df, expected, data)

            predicts = rbf.predictReg(data.test_df, data)
            expc_list = expected.values.tolist()
            print("predicts")
            print(predicts)
            print("expected")
            print(expc_list)

            print("MSE")
            mse = rbf.mean_squared_error(predicts, expc_list)
            print(mse)


    def test_rbfClass(self):
            #data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 8)  # load data
            data = Data('abalone', pd.read_csv('data/forestfires.data', header=None), 8)  # load data
            df = data.df.sample(n=100)  # minimal data frame

            cols = df.columns
            for col in cols:
                df[col] = df[col].astype(float)
            expected = df[df.columns[-1]]


            df = df.iloc[:, :-1]
            data.split_data(data_frame=df)  # sets test and train data
            # will have high error due to small dataset, but just a test to show how this works
            rbf = RBFClass(clusters=12, maxruns=8000)

            class_vals = list(range(1, 29))
            rbf.train(data, data.train_df, class_vals)

            predicts = rbf.predictClass(data.test_df, data)
            expc_list = expected.values.tolist()
            print("predicts")
            print(predicts)
            print("expected")
            print(expc_list)






if __name__ == '__main__':
    unittest.main()
