from Cluster import KNN
import pandas as pd


class PAM(KNN):
    """
    Inheritance allows PAM to use functions in KNN, or override them, and use it class variables.
    """

    # TODO check that when an instance of PAM is made, an instance of KNN is also made.
    def __init__(self, k_val, data_instance):
        super().__init__(k_val, data_instance)
        self.current_medoids = pd.DataFrame().reindex_like(data_instance.train_df)

    @staticmethod
    def assign_random_medoids(df, k):
        """
        randomly selects examples to represent the medoids
        :param df: data frame to get random points from
        :param k: number of medoids to instantiate
        :return: k number of medoids
        """
        medoid_list = []
        rand_med = df.sample(n=k)
        for index, row in rand_med.iterrows():
            Medoid.static_medoid_indexes.append(index)  # add to static variable; eases checking if data in medoids
            medoid_list.append(Medoid(row, index, pd.DataFrame(columns=df.columns, index=None)))
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
            med.encompasses.loc[index] = row  # append to the closest medoid point
            Medoid.static_cost += cost

    @staticmethod
    def order_by_dict_values(dictionary):
        """
        Orders least to greatest a given dictionary by value (not key) and returns a list of tuples [(key, val_min),
        ..., (key, val_max)]
        :param dictionary: dictionary to sort
        :return: an ordered list of tuples by value
        """
        return sorted(dictionary.items(), key=lambda item: item[1])

    def test_new_medoid(self, df, medoid_list):
        """
        Test ALL non-medoids as a replacement for a current medoid
        :param df: data frame to use
        :param medoid_list: medoids
        :return:
        """
        for med_index in range(len(medoid_list)):  # iterate through indexes of medoid list
            print("\nCurrent Medoid being updated: ", medoid_list[med_index].index)
            initial_medoid = medoid_list[med_index]
            test_encompass = initial_medoid.encompasses.copy
            for index, row in df.iterrows():  # iterate through data
                # if index in Medoid.static_medoid_indexes or index in medoid_list[med_index].recently_used:
                if index in Medoid.static_medoid_indexes:
                    continue  # do not use a medoid
                test_medoid = Medoid(row, index, test_encompass)  # testing_medoid
                temp_medoid_list = medoid_list.copy()  # copy actual medoid_list
                temp_medoid_list[med_index] = test_medoid  # replace actual medoid from temp list with testing medoid
                Medoid.resets(temp_medoid_list, df)
                self.assign_data_to_medoids(df, temp_medoid_list)
                swap_bool = self.compare_medoid_costs(medoid_list[med_index], test_medoid)
                if swap_bool:
                    Medoid.static_medoid_indexes[med_index] = test_medoid.index
                    medoid_list = temp_medoid_list  # swap the lists
                else:
                    continue
        return medoid_list

    def better_fit_medoids(self, df, medoid_list):
        """
        Updates the medoids to a better fit medoid if need be!
        :param df: data frame to use
        :param medoid_list: list of medoids to use
        :return: Medoid list, so that PAM instance can update it for training.
        """
        first_indexes = Medoid.static_medoid_indexes.copy()
        print(
            "__________________________________________________\n__________ Begin Finding Better Medoids "
            "__________\n__________________________________________________\n")
        print("Initial Medoid Indexes: ", Medoid.static_medoid_indexes)

        while True:
            initial_list = Medoid.static_medoid_indexes.copy()  # printing purposes
            change_in_list = self.test_new_medoid(df, medoid_list.copy())  # used to compare
            if medoid_list != change_in_list:  # continue finding better fits iff a better medoid was found
                print("\nInitial Medoid list: ", initial_list, "\nReturned Medoid List: ", Medoid.static_medoid_indexes)
                print("\n---------- Continue Finding Better Medoids ----------")
                medoid_list = change_in_list  # must update list
                continue
            else:
                print("\nNo More Changes!\nFinal Medoid Indexes: ", Medoid.static_medoid_indexes, " Compared to ",
                      first_indexes)
                break  # no more changes ands the loop
        return medoid_list

    def perform_pam(self, df, medoid_list):
        """
        Updates the medoids to a better fit medoid if need be!
        :param df: data frame to use
        :param medoid_list: list of medoids to use
        :return: Medoid list, so that PAM instance can update it for training.
        """
        first_indexes = Medoid.static_medoid_indexes.copy()
        print(
            "__________________________________________________\n__________ Begin Finding Better Medoids "
            "__________\n__________________________________________________\n")
        print("Initial Medoid Indexes: ", Medoid.static_medoid_indexes)

        while True:
            initial_list = Medoid.static_medoid_indexes
            changed_list, changed_static_list = self.compare_medoids(medoid_list.copy(), df)
            if changed_list != medoid_list:
                medoid_list = changed_list
                initial_list = changed_static_list
                print("\nInitial Medoid list: ", initial_list, "\nReturned Medoid List: ", Medoid.static_medoid_indexes)
                print("\n---------- Continue Finding Better Medoids ----------")
                self.assign_data_to_medoids(df, medoid_list)
                continue
            else:
                break
            pass

    def compare_medoids(self, medoid_list, df):
        temp = Medoid.static_medoid_indexes.copy()
        temp_medoid_list = medoid_list
        for med_index in range(len(temp_medoid_list)):

            initial_medoid = medoid_list[med_index]
            for index, row in df.iterrows():
                test_medoid = Medoid(row, index, initial_medoid.encompasses)
                test_medoid_list = medoid_list.copy()  # copy actual medoid_list
                test_medoid_list[med_index] = test_medoid  # replace actual medoid from temp list with testing medoid
                test_medoid.cost += self.distortion_from_encompassed(test_medoid)
                swap_bool = self.compare_medoid_costs(initial_medoid, test_medoid)
                if not swap_bool:
                    continue
                initial_medoid = test_medoid
                temp_medoid_list = test_medoid_list
                temp[med_index] = test_medoid.index
        return temp_medoid_list, temp

    def distortion_from_encompassed(self, test_medoid):
        distortion = 0
        for index, query in test_medoid.encompasses.iterrows():
            distortion += self.get_euclidean_distance(query, test_medoid.row)
        return distortion

    @staticmethod
    def compare_medoid_costs(actual, test):
        """
        Compares two medoid costs
        :param actual: actual medoid in list
        :param test: testing medoid to potentially swap in
        :return: boolean to determine if a swap occurred
        """
        if actual.cost > test.cost and test.cost is not 0:  # do not want 0; means that they are the same!
            print("Swap Justification\t\t-------->\t\tInitial Medoid", actual.index, " cost: ", actual.cost,
                  "\t\tcomparing to\t\t Test Medoid ", test.index,
                  " cost: ", test.cost)
            return True
        else:
            return False

    def calcualte_distortions(self, medoid, medoid_list):
        # TODO: above, change the static indexes to test indexes (be sure to save the state of the initial indexes).
        #  Assign initial_medoid to a cluster. Then, iterate through each encompassed list element and get the cost
        #  excluding the indexes in test indexes to get distortion'. Finally compare the distortions. Choose
        #  accordingly. If swapped... use variable to keep tab on which medoid it belonged to, and remove it. Assign
        #  the swapped out medoid to a medoid. And repeat.
        pass


class Medoid:
    static_medoid_indexes = []  # eases checking if data in medoids
    static_cost = 0

    def __init__(self, row, index, encompasses):
        """
        Medoids instance. Initializes the medoids current point, what data is contained within that medoid and an index
        :param row: the point representing the medoid
        :param index: index of the point
        """
        self.row = row
        # self.encompasses = {}  # dictionary to store data of medoid's encompassed data points.
        self.encompasses = encompasses  # Dataframe of its medoids
        self.index = index  # index of data frame
        self.cost = 0  # individual medoid cost
        self.recently_used = []  # list of test_medoids to potentially

    def get_medoid_encompasses(self):
        """
        Getter function to use the encompassed list
        :return: the data points the medoid encompasses
        """
        return self.encompasses

    @staticmethod
    def resets(medoid_list, df):
        # TODO Make sure cost is actually being changed for medoids (since static) Use function below if need be
        """
        reset the costs when recalculating the cost of medoids
        :return: None
        """
        for medoid in medoid_list:
            medoid.cost = 0
            medoid.encompasses = pd.DataFrame(columns=df.columns, index=None)

    def reset_cost(self):
        self.cost = 0

    def get_cost(self):
        """
        getter function for the cost of the medoid
        :return: cost from THIS medoid to all other medoids it encompasses
        """
        return self.cost
