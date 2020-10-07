from sklearn import datasets

if __name__ == '__main__':
    iris = datasets.load_iris()
    irisFeatures = iris["data"]
    irisFeaturesName = iris["feature_names"]
    irisLabels = iris["target"]

    print('Iris feature name:', irisFeaturesName)
    print('Iris data size :', irisFeatures.shape)
    print('Iris label size :', irisLabels.shape)



