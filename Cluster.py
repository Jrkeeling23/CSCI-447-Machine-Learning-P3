class KNN:
    def __init__(self, k_val, data_instance):
        self.k = k_val
        self.train_df = data_instance.train_data
        self.test_df = data_instance.test_data
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
            if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int:
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
        distances = self.get_euclidean_distance_dict(query_point, train_data)
        nearest_neighbors_distances, nearest_neighbors = self.get_k_closest(distances, k_val, train_data, self.data_instance.label_col)
        seen = []
        for index in nearest_neighbors:
            if index not in seen:
                seen.append(index)
        see_count = {}
        for i in seen:
            see_count[nearest_neighbors.count(i)]=0
        for j in nearest_neighbors:
            see_count[j]+=1
        return max(see_count)[0]

