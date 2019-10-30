import random

import numpy as np
import pandas as pd

from Cluster import KNN
from Data import CATEGORICAL_DICTIONARY, DataConverter


class Kmeans(KNN):


    def __init__(self, k_val, data_instance):
        super().__init__(k_val, data_instance)
        self.converter = DataConverter()
    def k_means(self, data_set, k_val):  # Method for K-Means
        print("\n-----------------Starting K-Means Function-----------------")
        # centroid_points = self.create_initial_clusters(self.k_random_rows(data_set,
        #                                                                   k_val))  # Get random rows for centroid points then create the initial centroid point pd.DataFrames
        centroid_points = self.k_random_point(data_set, k_val)
        # print(data.convert_data_to_original(data_set))
        while True:
            current_points = []  # A list of the current data points for a cluster
            previous_points = centroid_points  # Sets a previous value to check if K-means has converged
            clusters = []  # List of clusters for use below
            initiate_list = 0
            for point in centroid_points:  # Make a list of DataFrame clusters
                clusters.append([])  # Instantiates a list for the clusters
                clusters[initiate_list].append(point)  # Adds the centroid points to the list
                initiate_list += 1
            for _, data in data_set.iterrows():  # Loops through the rows of the data set
                distance = None  # Initializes distance
                current_closest_point = []  # Keeps track of the current closes point
                iterator = 0
                for centroid in centroid_points:  # Loops through the k centroid points
                    euclid_distance = KNN.get_euclidean_distance(centroid,
                                                              data)  # Gets the distance between the centroid and the data point
                    if distance is None or euclid_distance < distance:  # Updates the distance to keep track of the closest point
                        distance = euclid_distance
                        current_closest_point = [data, iterator]
                    iterator += 1
                clusters[current_closest_point[1]].append(
                    list(current_closest_point[0]))  # Appends the list to the clusters for specific centroids
            for closest_cluster in clusters:  # Loops through the closest cluster list
                current_points.append(closest_cluster)
            centroid_points = self.get_new_cluster(
                current_points)  # Calls the get new cluster function to get the mean values and run through the updated centroid points
            print("Previous Clusters:")
            print(pd.DataFrame(previous_points))
            print("\nUpdated Clusters:")
            print(pd.DataFrame(centroid_points))
            if centroid_points == previous_points:
                print("\n----------------- K-Means has converged! -----------------")
                break

        return centroid_points

    def k_random_point(self, data_set, k_val):  # Method to grab k_random rows for centroid method
        data_set = data_set
        print("\n-----------------Getting K Random Cluster Points-----------------")
        centroid_points = []  # List of centroid points type Series
        for k in range(k_val):  # Grabs k Centroids
            # Following row iteration with iteritems() sourced from https://stackoverflow.com/questions/28218698/how-to-iterate-over-columns-of-pandas-dataframe-to-run-regression/32558621 User: mdh and mmBs
            for _, row in data_set.iterrows():  # Loop through the rows of the dataframe
                current_point = []  # List for current random point in loop

                for item in row:  # Loop through each item in the row
                    current_point.append(random.uniform(0, float(len(CATEGORICAL_DICTIONARY)))) # radom uniform source: https://pynative.com/python-get-random-float-numbers/
                centroid_points.append(current_point)  # Appends the point to a list to be returned

        return centroid_points  # Returns a Series of centroid points

    def get_new_cluster(self, current_clusters):  # Method to get the sum of values of the clusters
        print("\n----------------- Updating K-Means Clusters -----------------\n")
        mean_cluster = []  # Instantiates a list of the updated clusters
        for cluster in current_clusters:  # Loop through the current clusters to get the sum of the values
            current_point = []
            cluster_length = len(cluster)  # Passed to mean_current_cluster
            str_dict = {}  # Dictionary of the first column str labels
            for point in cluster:  # Loop through each data point in a cluster
                iterator = 0
                for index in point:
                    if type(index) is str:
                        try:
                            if index in str_dict.keys():
                                str_dict[index] += 1  # Increments the count of a particular string
                            else:
                                str_dict[index] = 1  # Instantiates a value for a particular string
                            current_point[iterator] = index  # Place holder in the list
                        except:
                            current_point.append(index)  # Place holder in the list
                    elif type(index) is np.float64 or type(index) is float:  # Handles float values
                        try:
                            current_point[iterator] = current_point[iterator] + float(
                                index)  # Sums the value for this particular column in the loop
                        except:
                            current_point.append(index)  # Instantiates a value for the index location

                    elif type(index) is int or type(index) is np.int64:  # Handles Int values
                        try:
                            current_point[iterator] += index  # Sums the value for this column
                        except:
                            current_point.append(index)  # Instantiates a value for this location in the list.
                    iterator += 1
            mean_cluster.append(self.mean_current_cluster(cluster_length, current_point,
                                                          str_dict))  # Appends the new cluster value to be returned.

        return mean_cluster

    def mean_current_cluster(self, cluster_length, current_point,
                             str_dict):  # This function does the math for the new centroid
        highest_char_count = 0  # Decided to use the highest occurring string.
        if str_dict.keys().__len__() > 0:
            iterator = 1
        else:
            iterator = 0
        for char in str_dict.keys():  # Loops through the string dictionary
            if str_dict[char] > highest_char_count:
                highest_char_count = str_dict[char]
                current_point[0] = char  # Sets the first location in the centroid list to the most occurring string.

        for index in range(iterator, len(current_point)):  # Loops through the values for the mean of the cluster
            current_point[
                index] /= cluster_length  # Divides the sum by the length of the columns in the cluster data set.
            if index == len(current_point) - 1:
                current_point[index] = int(
                    current_point[index])  # Last value in the data set is an INT. This is a type cast.
        return current_point

    def predict_centroids(self, centroids, data_set):  # Method to return closest cluster to test data
        print("\n----------------- Predicting Closes Cluster on Test Data -----------------\n")

        for _, data in data_set[data_set].iterrows():  # Loops through the rows of the data set
            distance = None  # Initializes distance
            closest_centroid = None  # Keeps track of the current closes centroid cluster
            closest_centroid_euclidian_distance = None  # Keeps track of the closest euclidian distance.
            cluster_val = 1
            for centroid in centroids:  # Loops through the k centroid points
                euclid_distance = self.euclidean_distance(centroid,
                                                          data)  # Gets the distance between the centroid and the data point

                if distance is None or euclid_distance < distance:  # Updates the distance to keep track of the closest point
                    distance = euclid_distance
                    # closest_centroid = centroid
                    closest_centroid = cluster_val
                    closest_centroid_euclidian_distance = distance
                cluster_val += 1
            # Print closest cluster to the test data point.
            print("\nEuclidian Distance to Closest K-Means Cluster: ", closest_centroid_euclidian_distance)
            print("Closest Cluster: Cluster ", closest_centroid)


