class PAM:
    def __init__(self):
        pass
    
class Medoids:

    def __init__(self, row, index):
        """
        Medoids instance. Initializes the medoids current point, what data is contained within that medoid and an index
        :param row: the point representing the medoid
        :param index: index of the point
        """
        self.medoid_row = row
        self.encompasses = {}
        self.index = index
        self.cost = 0  # individual medoid cost
        self.next_medoid = None
        self.recently_used = []

    def assign_to_medoid(self, index, distance, row):
        """
        Assigns data to the medoid
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

    def set_next(self, next_medoid):
        """
        set a medoids next medoid for a circularly linked list
        :param next_medoid: medoid to be next in linked list
        :return: None
        """
        self.next_medoid = next_medoid

    def get_next(self):
        """
        getter function for the next medoid in linked list
        :return: next medoid in list
        """
        return self.next_medoid

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


class MedoidsLinkList:

    def __init__(self, max_size):
        """
        Create a medoid circularly linked list
        :param max_size: number of medoids to use
        """
        self.head = None
        self.current = None
        self.size = 0
        self.max_size = max_size

    def insert(self, row, index):
        """
        insert a medoid into the list
        :param row: row, used to instantiate a new medoid
        :param index: index, used to instantiate a new medoid
        :return: None
        """
        new_medoid = Medoids(row, index)  # new medoid of type medoids
        if self.head is None:  # if list is empty
            self.head = new_medoid
            self.current = new_medoid
        else:
            self.current.set_next(new_medoid)
            new_medoid.set_next(self.head)
        self.size += 1  # increment size of list