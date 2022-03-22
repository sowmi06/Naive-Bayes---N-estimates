import pandas as pd
import numpy as np


# function to read and preprocess the dataset
def monks_Dataset_Preprocessing(path):
    # reading the dataset file
    cols = ['classLabels', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    df = pd.read_csv(path, delimiter=',', names=cols)
    # df = np.asarray(df)
    # shuffle dataset before making a train and test split
    shuffled_dataset = df.sample(frac=1).reset_index(drop=True)
    shuffled_dataset = np.asarray(shuffled_dataset)
    # out of 432 instances, take 70% instances (i.e) "302" of the total samples as train instance
    train_dataset = shuffled_dataset[:302, :]
    np.savetxt("Training_dataset.csv", train_dataset, delimiter=",", fmt='%s')
    # splitting x_train and y_train (class and features) from train dataset
    x_train = train_dataset[:, 1:]
    y_train = train_dataset[:, :1]
    # take rest of 30% instances (i.e) "103" as test instance
    test_dataset = shuffled_dataset[302:, :]
    np.savetxt("Testing_dataset.csv", test_dataset, delimiter=",", fmt='%s')
    # splitting  x_test and y_test (class and features) from the test dataset
    x_test = test_dataset[:, 1:]
    y_test = test_dataset[:, :1]
    # printing the x_train, y_train, x_test and y_test shape
    print("-------------------------------------------------")
    print("Data Preprocessing:")
    print("The shape of x_train:", x_train.shape)
    print("The shape of y_train:", y_train.shape)
    print("The shape of x_test:", x_test.shape)
    print("The shape of y_test:", y_test.shape)
    print("Total instance in dataset:", shuffled_dataset.shape[0])
    print("Number of Features in dataset:", x_train.shape[1])
    print("Number of classes in dataset:", np.unique(y_train))
    print("-------------------------------------------------")
    return x_train, y_train, x_test, y_test


# function to calculate the probablity frequency
def calculate_probablity(data, total_instances):
    # create a dictionary to store probablity values
    dict_of_probablity = {}
    # finding the unique values in a given 1d-array
    unique_data_values = np.unique(data)
    # looping for all the uniques values in an array
    for values in unique_data_values:
        # finding the total uniques values form the given values
        NoOfTotalUniqueValues = sum(data == values)
        # calculating the probablity
        probablity = NoOfTotalUniqueValues / total_instances
        # adding it to dictionary with respective label information
        dict_of_probablity[values] = probablity
    return dict_of_probablity


# function to calculate priors from the train samples
def prior(y_train, label):
    # calculating the total number of instance in dataset
    total_instances = y_train.shape[0]
    # calculating prior probablity of (Y=y_i) using the created calculate_probablity() funaction
    priors = calculate_probablity(y_train, total_instances)
    priorOf_Class = priors[label]
    return priorOf_Class

# calculate the possible probablity vales for each unique feature in the dataset to calaculate the liklihood
def liklehood_values(x_train, y_train, label):

    # fetch values of features and labels with respect to a certain class
    # getting the indices of the label(y) in a dataset
    index_of_y_with_label = np.where(y_train == label)
    index_of_y_with_label = index_of_y_with_label[0]
    # calculating total instances in each class
    total_sample_with_singleclass = index_of_y_with_label.shape[0]
    x_data_with_singleclass = []
    # for each index value in y, fetching the features and class labels with singel class values
    # (i.e) 0 and 1 from our dataset
    for index in index_of_y_with_label:
        temp = x_train[index]
        # appending the separated features and label to a list
        x_data_with_singleclass.append(temp)
    x_data_with_singleclass = np.asarray(x_data_with_singleclass)
    # transposing to access the values feature-wise
    x_train = np.transpose(x_train)
    x_data_with_singleclass = np.transpose(x_data_with_singleclass)

# estimating likehood probability values for each unique values in dataset with respect to each labels
    liklehood = []
    # calculating the 'n' total samples in each classes
    n = total_sample_with_singleclass
    # adding constant 'm' values; the virtual samples in to the dataset using m-estimate of probablity
    m = 10
    # looping for each unique labels in the dataset (i.e) for class0 and class1 in our dataset
    # for each instance in dataset
    for i in range(len(x_train)):
        feature = x_train[i]
        # calculating the unique features in dataset
        unique_features = np.unique(feature)
        dict_likelihood = {}
        for values in unique_features:
            #  calculating the n_c value for the particular class and unique value in feature
            n_c = sum(x_data_with_singleclass[i] == values)
            # assuming uniform prior by calculating number of possible values of the feature
            p = unique_features.shape[0]
            # assigning the m_p value for the m-estimate
            m_p = 1/p
            # calculating the formula (n_c + m_p) / (n + m)
            numerator = n_c + m_p
            denominator = n + m
            # calculating the liklihood with m-estimate of probablity
            likelihood_of_each_feature = numerator / denominator
            # appending the values of each class and each unique feature with its value in a dictionary of list
            dict_likelihood[values] = likelihood_of_each_feature
        liklehood.append(dict_likelihood)
    liklehood = np.asarray(liklehood)
    return liklehood


# function for Naive Bayes classifier
def NaiveBayes(x, y, x_train, y_train):

    # getting unique values in labels
    unique_label = np.unique(y)
    combinations = []

    # calculating the prior and likelihood from the created prior() and
    # likelihood_values() function to estimate prior and likelihood

    # prior of class 0
    priorOf_Class0 = prior(y_train, 0)
    # list of probability combinations of class 0
    likelihoodOf_Class0 = liklehood_values(x_train, y_train, 0)

    # prior of class 1
    priorOf_Class1 = prior(y_train, 1)
    # list of probability combinations of class 1
    likelihoodOf_Class1 = liklehood_values(x_train, y_train, 1)

    # calculating the posterior probability for each instances in the dataset to predict its class
    for label in unique_label:
        # if class value is 1 then prior and likelihood probability of class 1 is taken
        priorOf_Class = priorOf_Class1
        likelihood = likelihoodOf_Class1
        # else if class value is 0 then prior and likelihood probability of class 0 is taken
        if label == 0:
            priorOf_Class = priorOf_Class0
            likelihood = likelihoodOf_Class0

        posterior_list_for_all_instance = []
        # for each instance in the dataset and for each value in the instance calculate the posterior_ probablity
        # for each unique class and stores in a list
        for index in range(len(x)):
            instance = x[index]
            Likelihood_probablity = 1
            for index in range(len(instance)):
                # fetching the respective value in the likelihood value
                value = instance[index]
                prob_combinations = likelihood[index]
                probablity = prob_combinations[value]
                # calculating the posterior
                Likelihood_probablity = Likelihood_probablity * probablity
            # multiplying the prior with the calculated likehood values
            posterior_probablity = Likelihood_probablity * priorOf_Class
            posterior_list_for_all_instance.append(posterior_probablity)
        combinations.append(posterior_list_for_all_instance)
    # calculated posterior probability for class 0
    posterior_class0 = combinations[0]
    posterior_class0 = np.asarray(posterior_class0)
    # calculated posterior probability for class 1
    posterior_class1 = combinations[1]
    posterior_class1 = np.asarray(posterior_class1)

# calculating training and testing accuracy using the posterior_probablity array with 2 classes
    correct_value = 0
    # getting the total instances
    total_instances = x.shape[0]
    for i in range(len(posterior_class0)):
        # take the i_th value from posterior_probablity array for class 1 and 0
        prob0 = posterior_class0[i]
        prob1 = posterior_class1[i]
        # if posterior_probablity of class 0 is greater the posterior_probablity of class 1
        # set the predicted value as class 0 else set it to class 1
        y_predicted = 1
        if prob0 > prob1:
            y_predicted = 0
        y_actual = y[i]
        # compare the true y value with the prdicted y value
        if y_predicted == y_actual:
            correct_value = correct_value + 1
    # calculating accuracy using formula
    accuracy = correct_value / total_instances
    return accuracy


def main():
    # assigning dataset path
    dataset_path = "./NB_dataset.csv"

    # calling the creating dataset preprocessing function
    x_train, y_train, x_test, y_test = monks_Dataset_Preprocessing(dataset_path)

    print("Naive Bayes Algorithim:")
    print("-------------------------------------------------")


    # calling the Naive Bayes algorithim to train the dataset
    training_accuracy = NaiveBayes(x_train, y_train,x_train, y_train)
    print("Training accuracy:", training_accuracy)

    # calling the Naive Bayes algorithim to test the dataset
    testing_accuracy = NaiveBayes(x_test, y_test, x_train, y_train)
    print("Testing accuracy:", testing_accuracy)
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()
