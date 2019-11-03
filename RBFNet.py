import numpy as np;
from PAM import PAM
from Cluster import KNN
import math

class RBFReg:
    """
          Class to function as an RBF regression network (one output node)
          :param clusters, # of clusters (hidden nodes)
          :param isReg,  bool to check if regression or not
          :param learning_rate,  tuning paramater for our RBF net
          :param maxruns, maximum amount of cycles we want the RBF to run for


           """

    def __init__(self,  clusters,  maxruns=1000, learning_rate=.01, ):

        self.clusters = clusters
        self.learning_rate = learning_rate
        # weight array,  use out_nodes size 1  for reg
        self.weights = np.random.uniform(-self.learning_rate, self.learning_rate, size=(self.clusters))
        # bias term, vector of size out_nodes (so size 1 for reg as 1 output node)
        self.bias = np.random.randn(1)
        self.maxruns = maxruns
        self.std = None
        self.medoids = None

    def gaus(self, x, c, s):
            """ gausian"""
            return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

    def calcHiddenOutputs(self, input, center, std):
            knn = KNN
            dist_between = knn.get_euclidean_distance(input, center)


            return np.exp(-1 / (2 * std ** 2) * dist_between ** 2)


    ":param medoids_list,  a list of medoids in a cluster"
    # function to get max dist
    def getMaxDist(self, medoids_list):
        maxDist = 0
        knn = KNN
        for medoid in medoids_list:
            for medoid2 in medoids_list:
                # compare against all other medoids
                curDist = knn.get_euclidean_distance(medoid.row, medoid2.row)
                if curDist > maxDist:
                    maxDist = curDist
        return maxDist



    """
    :param data_set : data set to train on
    :param expected_value : expected values for that data set
    :param data_instance : instance of data class
    """
    def trainReg(self, data_set, expected_values, data_instance):
        pam = PAM(k_val=self.clusters, data_instance=data_instance)
        medoids_list = pam.assign_random_medoids(data_set, self.clusters)
        pam.assign_data_to_medoids(data_set, medoids_list)
        # set the STD of the clusters  (doing once for now)
        self.std = self.getMaxDist(medoids_list) / np.sqrt(2 * self.clusters)
        self.medoids = medoids_list
        # train the regression model
        converged = False
        iterations = 0
        while not converged:

            if(iterations % 100 is 0):
                print("Iteration: ")
                print(iterations)
            for index, row in data_set.iterrows():
                # calculate the activation functions for each of the examples for each hidden node (cluster)
                a = []

                for medoid in medoids_list:
                    medoidAct = self.calcHiddenOutputs(row, medoid.row, self.std)

                    a.append(medoidAct)

                a = np.array(a)
                F = a.T.dot(self.weights) + self.bias
                #backwards update pass

                error = -(expected_values[index] - F)
                self.weights = self.weights - self.learning_rate * a * error
                self.bias = self.bias - self.learning_rate * error


            if iterations > self.maxruns:
                # break out if we hit the maximum runs
                converged = True

            iterations += 1


    # function to predict values given current weights and a dataset
    def predictReg(self, data_set):
        predictions = []
        for index, row in data_set.iterrows():
            a = []

            for medoid in self.medoids:
                medoidAct = self.calcHiddenOutputs(row, medoid.row, self.std)

                a.append(medoidAct)

            a = np.array(a)
            F = a.T.dot(self.weights) + self.bias

            predictions.append(F.item())
        return predictions


    def mean_squared_error(self, predicted_data, actual_data):
        """
        :param predicted_data:  list of predicted values for datapoints (assume same order)
        :param actual_data: actual values for those said data points  (assume same order)
        :return MSE from the predicted data
         """
        n = len(actual_data)  # get out n for MSE
        sum = 0  # sum of the MSE squared values

        for (predict, true) in zip(predicted_data, actual_data): # go through all the points at the same time
            currentSum = (true - predict) ** 2  # square it
            sum += currentSum # add current to total sum

        # divide by n
        sum = sum/n
        return sum # done, return sum
# output node class for RBF classification
class output:
        def __init__(self, weights, classval,):
            self.weights = weights
            self.classval = classval
            self.bias = np.random.randn(1) # bias term for this node

class RBFClass:
        """
         Class to function as an RBF netwwork
         :param out_nodes, # of nodes in output layer
         :param clusters, # of clusters (hidden nodes)
         :param isReg,  bool to check if regression or not
         :param learning_rate,  tuning paramater for our RBF net
         :param maxruns, maximum amount of cycles we want the RBF to run for


          """
        def __init__(self, out_nodes, clusters,  maxruns=1000, learning_rate=.01, ):
            self.out_nodes = out_nodes
            self.clusters = clusters
            self.learning_rate = learning_rate
            # weight array,  use out_nodes size 1  for reg
            self.outputnodes = []
            # bias term, vector of size out_nodes (so size 1 for reg as 1 output node)
            self.bias = np.random.randn(clusters)
            self.maxruns = maxruns
            self.std = None

        '''
        :param x : an example
        :param c : center of a cluster
        :param s standard deviation of a cluster
        '''

        def gaus(self, x, c, s):
            """ gausian"""
            return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

        def calcHiddenOutputs(self, input, center, std):
            knn = KNN
            dist_between = knn.get_euclidean_distance(input, center)

            return np.exp(-1 / (2 * std ** 2) * dist_between)

        ":param medoids_list,  a list of medoids in a cluster"
        # function to get max dist
        def getMaxDist(self, medoids_list):
            maxDist = 0
            knn = KNN
            for medoid in medoids_list:
                for medoid2 in medoids_list:
                    # compare against all other medoids
                    curDist = knn.get_euclidean_distance(medoid.row, medoid2.row)
                    if curDist > maxDist:
                        maxDist = curDist
            return maxDist

        #function to create list of outputs
        # param:  class_values,  the list of class values
        def createOutputs(self, class_values):
            outputs = []
            for outclass in class_values:
                node = output(classval=outclass, weights=np.random.uniform(-self.learning_rate, self.learning_rate, size=self.clusters))
                outputs.append(node)
            return outputs

        # get class of some point
        def getClass(self, point, data):
            this_class = point[data.label_col]
            print(this_class)
            return this_class

        """ Training function  to train our RBF
            :param data_instance, instance of data object
            :param data_set. set of training data
            :param actual_set,  set of actual outputs to compare training data too
            :param class_values, the values for all classes in the dataset
        """
        def train(self, data_instace, data_set, class_values):
            # getting the clusters (medoids)
            pam = PAM(k_val=self.clusters, data_instance=data_instace)
            medoids_list = pam.assign_random_medoids(data_set, self.clusters)
            pam.assign_data_to_medoids(data_set, medoids_list)
            # set the STD of the clusters  (doing once for now)
            self.std = self.getMaxDist(medoids_list) / np.sqrt(2*self.clusters)
            # create output nodes
            self.outputnodes = self.createOutputs(class_values)
            # var to represent convergence
            converged = False
            iterations = 0
            while not converged:
                if iterations % 100 is 0:
                    print("Iteration: ")
                    print(iterations)

                for node in self.outputnodes:  # row represents the weights of a given end node

                    for index, row in data_set.iterrows():
                        # calculate the activation functions for each of the examples for each hidden node (cluster)
                        a = []
                        for medoid in medoids_list:
                            medoidAct = self.calcHiddenOutputs(row, medoid.row, self.std)
                            a.append(medoidAct)

                        # convert a to a numpy array
                        a = np.array(a)
                        # add in the bias term to current row
                        a = np.add(a, node.bias)
                        # change all values where class of
                        # get the value of F(x) using the weights
                        temp = a.T.dot(node.weights)
                        # temp = temp.add(temp, node.bias)

                        # calcuate sigmoid for each output, then use sigmoidal values to perform gradient descent
                        F = 1 / (1 + math.exp((-temp)))

                        # TODO:  Calculate weights for this output



                        # row = row - self.learning_rate * a * error
                        # self.bias = self.bias - self.learning_rate * error

                if iterations > self.maxruns:
                    # break out if we hit the maximum runs
                    converged = True

                iterations += 1




