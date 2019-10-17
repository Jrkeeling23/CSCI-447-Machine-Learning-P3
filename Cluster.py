class KNN:
    def __init__(self, k_val, train_df):
        self.k = k_val
        self.train_df = train_df

    def get_euclidean_distance(self, query_point, comparison_point, data_name):
        """
        Performs the Euclidean distance function
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

    def get_euclidean_distance_dict(self, query_point, comparison_data):
        """
        Performs the Euclidean distance function
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

    def get_k_closest(self, sort_this, k_val, data_frame, label_col):
        """
        determines the smallest distance to the query point
        :param k_val: number of values to grab
        :param sort_this: dictionary of distances from query point to medoids
        :return: k clostest distances and their associated labels
        """
        count = 0  # stops for loop
        v_label_list = []
        v_distance_list = []
        for key, value in sorted(sort_this.items(), key=lambda item: item[1]):
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