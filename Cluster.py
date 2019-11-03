import operator

class KNN:
    def __init__(self, k_val, data_instance):
        self.k = k_val
        self.train_df = data_instance.train_df
        self.test_df = data_instance.test_df
        self.data_instance = data_instance

    @staticmethod
    def get_euclidean_distance(query_point, comparison_point):
        """
        Performs the Euclidean distance function for a single data point against a query point
        :param data_name:
        :param query_point: a data point
        :param comparison_point: a comparison point
        :param df: used to get the label columns
        :return: a SINGLE  distance
        """
        temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part
        for feature_col in range(len(query_point)):
            # print(isinstance(query_point[feature_col], float))
            if isinstance(query_point[feature_col], float) or isinstance(query_point[feature_col], int):
                # print(query_point[feature_col])
                # print(comparison_point[feature_col])
                # print("\n")
                temp_sub = (query_point[feature_col] - comparison_point[feature_col]) ** 2  # x2 -x1 and square
                temp_add += temp_sub  # continuously add until square root

        return temp_add ** (1 / 2)  # square root ... return the specific distance

    @staticmethod
    def get_euclidean_distance_dict(query_point, comparison_data):
        """
        Performs the Euclidean distance function for a all the data needed to compare against query
        :param comparison_data: all data to be compared to the query point
        :param query_point: a data point
        :return: a dict of all distances given all the data
        """
        distance_dict = {}
        for index, comparison_point in comparison_data.iterrows():  # iterate  through all data for one query point
            temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part

            for feature_col in range(len(query_point)):
                # TODO: exclude labels
                # TODO: bin data and preprocess to handle strings....
                if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int:
                    temp_sub = (query_point[feature_col] - comparison_point[feature_col]) ** 2  # x2 -x1 and square
                    temp_add += temp_sub  # continuously add until square root

            distance_dict[temp_add] = (temp_add ** (1 / 2))  # square root ... return the specific distance
        return distance_dict

    @staticmethod
    def get_k_closest(distance_dict, k_val, data_frame, label_col):
        """
        get the k closest distances and labels associated with it.
        :param k_val: number of values to grab
        :param sort_this: dictionary of distances from query point to medoids
        :return: k clostest distances and their associated labels
        """
        count = 0  # stops for loop
        v_label_list = []
        v_distance_list = []
        for key, value in sorted(distance_dict.items(), key=lambda item: item[1]):
            # key is the index and value is the distance. Ordered least to greatest by sort().
            # if statement to grab the k number of distances and labels
            if count > k_val:
                break
            elif count is 0:
                count += 1  # first value is always 0.
                continue
            else:
                v_distance_list.append(value)  # add distance
                v_label_list.append(data_frame.loc[key, label_col])  # add label
                count += 1
        return v_distance_list, v_label_list

    def perform_KNN(self, k_val, query_point, train_data):
        distances = {}
        for index, row in train_data.iterrows():
            # print(row)
            distances[index] = self.get_euclidean_distance(query_point, row)
        nearest_neighbors_distances, nearest_neighbors = self.get_k_closest(distances, k_val, train_data, self.data_instance.label_col)
        seen = []
        for index in nearest_neighbors:
            # print (index)
            if index not in seen:
                seen.append(index)
        see_count = {}
        for i in range(len(seen)):
            see_count[seen[i]] = 0
        for j in nearest_neighbors:
            temp = see_count[j]
            temp += 1
            see_count[j] = temp

        return max(see_count, key=lambda k: see_count[k])

    def edit_data(self, data_set, k_value, validation, label_col):
        """
        Edit values for edit_knn by classifying x_initial; if wrong, remove x_initial. (option1)
        OR... if correct remove (option 2)
        :param data_set: the training data that will be edited
        :param k_value: the number of neighbors being checked against
        :param validation: the test data, so that there is a measurement of performance to know when to stop
        :param the column of the data that has the classifier
        :return: Edited data_set back to KNN
        """
        # TODO: edit data according to pseudo code from class on 9/23
        # prev_set = data_set
        data_set_perform = 0  # for getting an initial measure on performance
        print(data_set.shape)
        print(validation.shape)
        print(data_set)
        print(validation)
        for index, row in validation.iterrows():  # loops through the validation set and if it matches, then it adds one to the score
            knn = self.perform_KNN(k_value, row, data_set)
            # print(knn)
            # print(row[label_col])
            if knn == row[label_col]:
                data_set_perform += 1
                # print(data_set_perform)
        # data_set_perform = 20
        prev_set_perform = data_set_perform  # for allowing the loop to occur
        reduce_data = data_set
        prev_set = None
        while data_set_perform >= prev_set_perform:  # doesn't break until the performance drops below the previous set
            # print(data_set.shape)

            prev_set_perform = data_set_perform  # sets the previous set and previous set performance
            prev_set = reduce_data
            list_to_remove = []  # initializes the list of items that will be removed
            for index, row in reduce_data.iterrows():  # does knn on itself
                knn_value = self.perform_KNN(k_value, row, reduce_data)
                actual_value = row[label_col]
                if knn_value != actual_value:  # comparing the knn done on itself to it's actual value.  If it doesn't match, it will be removed
                    list_to_remove.append(index)
            reduce_data = reduce_data.drop(list_to_remove)  # removes the data points that don't match
            data_set_perform = 0  # resets the performance measure
            print(list_to_remove)
            for index, row in validation.iterrows():  # gets the performance measure
                knn = self.perform_KNN(k_value, row, reduce_data)
                if knn == row[label_col]:
                    data_set_perform += 1
            if len(list_to_remove) is 0:
                break
        print(prev_set_perform)
        print(str(data_set_perform) + "\n\n")
        print(data_set.shape)
        print(prev_set.shape)
        return prev_set  # returns the set with the best performance

