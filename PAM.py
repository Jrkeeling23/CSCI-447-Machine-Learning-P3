from Cluster import KNN


class PAM(KNN):
    """
    Inheritance allows PAM to use functions in KNN, or override them, and use it class variables.
    """

    # TODO check that when an instance of PAM is made, an instance of KNN is also made.
    def __init__(self, k_val, data_instance):
        super().__init__(k_val, data_instance)
        self.current_medoids = []

    def assign_random_medoids(self, df, k):
        """
        randomly selects examples to represent the medoids
        :param k: number of medoids to instantiate
        :return: k number of medoids
        """
        medoid_list = []
        rand_med = df.sample(n=k)
        for index, row in rand_med.iterrows():
            Medoid.static_medoid_indexes.append(index)  # add to static variable; eases checking if data in medoids
            medoid_list.append(Medoid(row, index))
        return medoid_list

    def assign_data_to_medoids(self, df, medoid_list):
        """
        Assigns the remaining data points to medoids
        :param df: data to add to medoids
        :param medoid_list: list of medoids
        :return: None
        """
        for index, row in df.iterrows():  # iterate through all the data
            if index in Medoid.static_medoid_indexes:  # do not assign a medoid to a medoid
                continue  # next index if a medoid
            temp_distance_dict = {}  # contains the distances of a data point to all the medoids
            for medoid in medoid_list:  # iterate through all the medoids (For one data point)
                temp_distance_dict[medoid] = super().get_euclidean_distance(row, medoid.row)  # med : distance
            list_of_tuples = self.order_by_dict_values(temp_distance_dict)  # sorts dictionary by value
            med = list_of_tuples[0][0]  # closest medoid
            cost = list_of_tuples[0][1]  # distance from closest medoid
            med.cost += cost  # append to medoid
            med.encompasses.append(index)  # append to the closest medoid point

    @staticmethod
    def order_by_dict_values(dictionary):
        """
        Orders least to greatest a given dictionary by value (not key) and returns a list of tuples [(key, val_min),
        ..., (key, val_max)]
        :param dictionary: dictionary to sort
        :return: an ordered list of tuples by value
        """
        return sorted(dictionary.items(), key=lambda item: item[1])


class Medoid:
    static_medoid_indexes = []  # eases checking if data in medoids

    def __init__(self, row, index):
        """
        Medoids instance. Initializes the medoids current point, what data is contained within that medoid and an index
        :param row: the point representing the medoid
        :param index: index of the point
        """
        self.row = row
        # self.encompasses = {}  # dictionary to store data of medoid's encompassed data points.
        self.encompasses = []
        self.index = index  # index of data frame
        self.cost = 0  # individual medoid cost
        self.next_medoid = None
        self.recently_used = []  # list of test_medoids to potentially

    def medoid_encompasses(self, index, distance, row):
        """
        Assigns data to the medoid
        :param row:
        :param distance:
        :param index: index of data
        :return:
        """
        self.encompasses[index] = row
        self.cost += distance

    def get_medoid_encompasses(self):
        """
        Getter function to use the encompassed list
        :return: the data points the medoid encompasses
        """
        return self.encompasses

    def reset_cost(self):
        """
        reset the costs when recalculating the cost of medoids
        :return: None
        """
        self.cost = 0

    def get_cost(self):
        """
        getter function for the cost of the medoid
        :return: cost from THIS medoid to all other medoids it encompasses
        """
        return self.cost

    def reset_recently_used(self):
        """
        reset list of one iteration's tested 'potential' medoid
        :return: None
        """
        self.recently_used = []

    def get_recently_used(self):
        """
        return list of recently tested potential medoids
        :return: recently tried medoids (for one iteration)
        """
        return self.recently_used

    def add_tested_medoid(self, index):
        """
        append to the list of tested 'potential' medoids
        :param index: append an index of the data point tried as a medoid
        :return: None
        """
        self.recently_used.append(index)

    def better_test_medoid(self, index, row, cost):
        """
        change the values of the medoid to a better fitting one
        :param index: index to change to
        :param row: row to change to
        :param cost: cost to change to
        :return:
        """
        self.cost = cost
        self.index = index
        self.row = row
