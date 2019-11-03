import numpy as np;
from PAM import PAM
from Cluster import KNN



class RBF:
        """
         Class to function as an RBF netwwork
         :param out_nodes, # of nodes in output layer
         :param clusters, # of clusters (hidden nodes)
         :param isReg,  bool to check if regression or not
         :param learning_rate,  tuning paramater for our RBF net
         :param maxruns, maximum amount of cycles we want the RBF to run for


          """
        def __init__(self, out_nodes, clusters, isReg, maxruns=1000, learning_rate=.01, ):
            self.out_nodes = out_nodes
            self.clusters = clusters
            self.isReg = isReg
            self.learning_rate = learning_rate
            # weight array,  use out_nodes size 1  for reg
            self.weights = np.random.uniform(-self.learning_rate, self.learning_rate, size=(self.out_nodes, self.clusters))
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




        """ Training function  to train our RBF
            :param data_instance, instance of data object
            :param data_set. set of training data
            :param actual_set,  set of actual outputs to compare training data too
        """
        def train(self, data_instace, data_set, actual_set):
            # getting the clusters (medoids)
            pam = PAM(k_val=self.clusters, data_instance=data_instace)
            medoids_list = pam.assign_random_medoids(data_set, self.clusters)
            pam.assign_data_to_medoids(data_set, medoids_list)
            # set the STD of the clusters  (doing once for now)
            self.std = self.getMaxDist(medoids_list) / np.sqrt(2*self.clusters)


            # var to represent convergence
            converged = False
            iterations = 0
            while not converged:
                # start training the model here
                # go through each of the output "nodes" (values for weights are stored as vectors in the weights matrix
                # this way I can avoid having to create a large number of classes
                for output in self.weights:  # row represents the weights of a given end node

                    for index, row in data_set.iterrows():
                        # calculate the activation functions for each of the examples for each hidden node (cluster)
                        a = []
                        for medoid in medoids_list:
                            medoidAct = self.calcHiddenOutputs(row, medoid.row, self.std)
                            a.append(medoidAct)

                        # convert a to a numpy array
                        a = np.array(a)
                        a = np.add(a, self.bias)
                        # add in the bias term to current row
                        F = a.T.dot(output)



                       

                        # TODO:  Impelemt the gradient descent rules using the activation values a as input
                        # backward pass
                        # if we are in regression do reg error, using expected value of the function
                       # if self.isReg:
                       #    error = 0
                           # error = -(y[i] - F).flatten()
                        # otherwise set 0 / 1 depending on class values
                       # else:
                        #    error = 0



                       # row = row - self.learning_rate * a * error
                        #self.bias = self.bias - self.learning_rate * error

                if iterations > self.maxruns:
                    # break out if we hit the maximum runs
                    converged = True

                iterations += 1




