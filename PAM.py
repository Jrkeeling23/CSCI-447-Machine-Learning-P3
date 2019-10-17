from Cluster import KNN


class PAM(KNN):
    """
    Inheritance allows PAM to use functions in KNN, or override them, and use it class variables.
    """
    # TODO check that when an instance of PAM is made, an instance of KNN is also made.
    def __init__(self, k_val, data_instance):
        super().__init__(k_val, data_instance)


class Medoids:

    def __init__(self, row, index):
        """
        Medoids instance. Initializes the medoids current point, what data is contained within that medoid and an index
        :param row: the point representing the medoid
        :param index: index of the point
        """
        self.medoid_row = row
        self.encompasses = {}  # dictionary to store data of tested "potential" medoids. 
        self.index = index  # index of data frame
        self.cost = 0  # individual medoid cost
        self.next_medoid = None
        self.recently_used = []  # list of test_medoids to potentially

    def assign_to_medoid(self, index, distance, row):
        """
        Assigns data to the medoid
        :param row:
        :param distance:
        :param index: index of data
        :return:
        """
        self.encompasses[index] = row
        self.cost += distance

    def reset_cost(self):
        """
        reset the costs when recalculating the cost of medoids
        :return: None
        """
        self.cost = 0

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

    def get_cost(self):
        """
        getter function for the cost of the medoid
        :return: cost from THIS medoid to all other medoids it encompasses
        """
        return self.cost

    def set_better_medoid(self, index, row, cost):
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



