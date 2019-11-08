from Data import Data
import pandas as pd
from RBFNet import RBFReg
from Cluster import KNN
from loss_functions import LF
from RBFNetKMean import RBFRegK
import matplotlib.pyplot as plt


def load_data():
    """
    loads the data (csv) files
    :return: list of Data instances
    """
    data_list = [Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8),
                 Data('car', pd.read_csv(r'data/car.data', header=None), 5),
                 Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None, skiprows=4), 0),
                 Data('machine', pd.read_csv(r'data/machine.data', header=None), 0),
                 Data('forest_fires', pd.read_csv(r'data/forestfires.data', header=None), 12),
                 Data('wine', pd.read_csv(r'data/wine.data', header=None), 0)]
    return data_list


# run RBF regression on 4 experiments (diff clusters)
def RBFREG_exp(data_config, data):
    # setup data var
    # data = Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None), 0)
    # load data
    df = data.df  # get the dataframe from df

    print("Checking DF set")
    print(df[df.columns[-1]])
    # double check data is numerical
    cols = df.columns
    for col in cols:
        df[col] = df[col].astype(float)
    # split into test/train
    data.split_data(data_frame=df)
    if data_config == 'condensed':  # Run RBF on condensed data set
        cluster_obj = KNN(5, data)
        data.train_df = cluster_obj.condense_data(data.train_df)

        print("\n---------------- Running Condensed Nearest Neighbor RBF -----------------")
        print('Size of data: ', data.train_df.shape)
        rbf = RBFReg(clusters=4, maxruns=1000)
        rbf2 = RBFReg(clusters=6, maxruns=1000)
        rbf3 = RBFReg(clusters=8, maxruns=1000)
        rbf4 = RBFReg(clusters=12, maxruns=1000)
    elif data_config == 'edited':  # Run RBF on edited dataset
        knn = KNN(5, data)
        data.train_df = knn.edit_data(data.train_df, 5, data.test_df, data.label_col)
        print("\n---------------- Running Edited Nearest Neighbor RBF -----------------\n")
        print('Size of data: ', data.train_df.shape)

        rbf = RBFReg(clusters=4, maxruns=1000)
        rbf2 = RBFReg(clusters=6, maxruns=1000)
        rbf3 = RBFReg(clusters=8, maxruns=1000)
        rbf4 = RBFReg(clusters=12, maxruns=1000)
    elif data_config == 'k-means':  # Run RBF on K-means
        print("\n---------------- Running K-Means RBF -----------------\n")
        rbf = RBFRegK(clusters=4, maxruns=1000)
        rbf2 = RBFRegK(clusters=6, maxruns=1000)
        rbf3 = RBFRegK(clusters=8, maxruns=1000)
        rbf4 = RBFRegK(clusters=12, maxruns=1000)
    elif data_config == 'medoids':  # Run RBF on Medoids
        print("\n---------------- Running Medoids RBF -----------------\n")
        rbf = RBFReg(clusters=4, maxruns=1000)
        rbf2 = RBFReg(clusters=6, maxruns=1000)
        rbf3 = RBFReg(clusters=8, maxruns=1000)
        rbf4 = RBFReg(clusters=12, maxruns=1000)
    # setup expected values for testings
    expected = data.train_df[data.train_df.columns[-1]]
    actual = data.test_df[data.test_df.columns[-1]]

    # sets test and train data
    # will have high error due to small dataset, but just a test to show how this works

    expc_list = actual.values.tolist()

    rbf.trainReg(data.train_df, expected, data)
    predicts = rbf.predictReg(data.test_df, data)

    print("predicts RBF 1")
    print(predicts)
    print("expected")
    print(expc_list)

    lf = LF()
    lf.mean_squared_error(predicts, expc_list)
    lf.zero_one_loss(predicts, expc_list)
    # print("MSE RBF 1")
    # mse = rbf.mean_squared_error(predicts, expc_list)
    # print(mse)

    rbf2.trainReg(data.train_df, expected, data)
    predicts2 = rbf.predictReg(data.test_df, data)

    print("predicts RBF 2")
    print(predicts2)
    print("expected")
    print(expc_list)

    # print("MSE RBF 2")
    # mse2 = rbf2.mean_squared_error(predicts2, expc_list)
    # print(mse2)
    lf.mean_squared_error(predicts, expc_list)
    lf.zero_one_loss(predicts, expc_list)

    rbf3.trainReg(data.train_df, expected, data)
    predicts3 = rbf.predictReg(data.test_df, data)

    print("predicts RBF 3")
    print(predicts3)
    print("expected")
    print(expc_list)

    # print("MSE RBF 3")
    # mse3 = rbf.mean_squared_error(predicts3, expc_list)
    # print(mse3)
    lf.mean_squared_error(predicts, expc_list)
    lf.zero_one_loss(predicts, expc_list)

    rbf4.trainReg(data.train_df, expected, data)
    predicts4 = rbf.predictReg(data.test_df, data)

    print("predicts RBF 4")
    print(predicts4)
    print("expected")
    print(expc_list)

    # print("MSE RBF 4")
    # mse4 = rbf.mean_squared_error(predicts4, expc_list)
    # print(mse4)
    lf.mean_squared_error(predicts, expc_list)
    lf.zero_one_loss(predicts, expc_list)


# run RBF regression on small dataset for video

def RBFREG_vid(data_config, data, best_performing):
    # data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
    df = data.df.sample(100)  # get the dataframe from df, take small subsection
    data_name = data.name
    print("\nChecking DF set")
    print(df[df.columns[-1]])
    # double check data is numerical
    cols = df.columns
    for col in cols:
        df[col] = df[col].astype(float)
    # split into test/train
    data.split_data(data_frame=df)

    # setup expected values for testings
    expected = data.train_df[data.train_df.columns[-1]]
    actual = data.test_df[data.test_df.columns[-1]]

    # sets test and train data
    # will have high error due to small dataset, but just a test to show how this works
    if data_config == 'condensed':  # Run RBF on condensed data set
        cluster_obj = KNN(5, data)
        data.train_df = cluster_obj.condense_data(data.train_df)

        print("\n---------------- Running Condensed Nearest Neighbor RBF -----------------")
        print('Size of data: ', data.train_df.shape)
        rbf = RBFReg(clusters=8, maxruns=600)

    elif data_config == 'edited':  # Run RBF on edited dataset
        knn = KNN(5, data)
        data.train_df = knn.edit_data(data.train_df, 5, data.test_df, data.label_col)
        print("\n---------------- Running Edited Nearest Neighbor RBF -----------------\n")
        print('Size of data: ', data.train_df.shape)

        rbf = RBFReg(clusters=8, maxruns=600)

    elif data_config == 'k-means':  # Run RBF on K-means
        print("\n---------------- Running K-Means RBF -----------------\n")
        rbf = RBFRegK(clusters=8, maxruns=600)

    elif data_config == 'medoids':  # Run RBF on Medoids
        print("\n---------------- Running Medoids RBF -----------------\n")
        rbf = RBFReg(clusters=8, maxruns=600)

    rbf.trainReg(data.train_df, expected, data)

    print('Calculate predictions for the RBF')
    predicts = rbf.predictReg(data.test_df, data)

    expc_list = actual.values.tolist()
    print("predicts RBF")

    print(predicts)
    print("expected")
    print(expc_list)
    lf = LF()
    mse = lf.mean_squared_error(predicts, expc_list)
    if best_performing is None:
        best_performing = [data_config, mse, data.name]
    elif mse < best_performing[1]:
        best_performing = [data_config, mse, data.name]
    zeroone = lf.zero_one_loss(predicts, expc_list)
    plt.plot(predicts, label=data_name + ' ' + data_config + ' prediction')
    plt.plot(expc_list, label=data_name + ' ' + data_config + ' expected')
    plt.plot(mse, label='MSE: ' + str(mse))
    plt.plot(zeroone, label='0-1 Loss: ' + str(zeroone))

    plt.legend()
    plt.title('Data: ' + data_name)
    plt.ylabel('Expected value/ Predicted Value')
    plt.xlabel('# Predictions')
    plt.savefig(
        data_name + '_' + data_config)  # Code for saving a plot to image sourced from: https://pythonspot.com/matplotlib-save-figure-to-image-file/
    plt.clf()
    # print("MSE RBF")
    # mse = rbf.mean_squared_error(predicts, expc_list)
    # print(mse)


class Main:
    def __init__(self):
        self.data_list = load_data()
    # def perform_KNN(self, k_val, query_point, train_data):


if __name__ == '__main__':
    data = load_data()
    rbf_list = ['k-means', 'medoids', 'edited']
    best_performing = None
    for dataset in data:
        print(dataset.name)
        for rbf_version in rbf_list:  # Run RBF
            # run video rbg freg
            RBFREG_vid(rbf_version, dataset, best_performing)

            # run experiment
            # RBFREG_exp(rbf_version)

    print('Best performing ', best_performing)
