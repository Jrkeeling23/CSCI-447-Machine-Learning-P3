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
        def __init__(self, out_nodes, clusters, isReg, maxruns, learning_rate=.01, ):
            self.out_nodes = out_nodes
            self.clusters = clusters
            self.isReg = isReg
            self.learning_rate = learning_rate
            # weight array,  use out_nodes size 1  for reg
            self.weights = np.random.uniform(-self.learning_rate, self.learning_rate, size=(self.out_nodes, self.clusters))
            # bias term, vector of size out_nodes (so size 1 for reg as 1 output node)
            self.bias = np.random.randn(out_nodes)
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
            :param data_set set of data to train on
        """
        def train(self, data_instace, data_set):
            # getting the clusters (medoids)
            pam = PAM(k_val=self.clusters, data_instance=data_instace)
            medoids_list = pam.assign_random_medoids(data_set, 5)
            pam.assign_data_to_medoids(data_set, medoids_list)
            # set the STD of the clusters  (doing once for now)
            self.std = self.getMaxDist(medoids_list) / np.sqrt(2*self.clusters)


            # var to represent convergence
            converged = False
            iterations = 0
            while not converged:
                # start training the model here


                if iterations > self.maxruns:
                    # break out if we hit the maximum runs
                    converged = True






