"""
Number of samples in the training set:  "3":  5713;        "7": 5835
Number of samples in the testing set :  "3": 1428;         "7": 1458
"""
import math

import numpy.linalg
import scipy.io
import numpy as np

data = scipy.io.loadmat("train_data.mat")
test_data = scipy.io.loadmat("test_data.mat")

all_data = data["data"]
all_labels = data["label"][0]
all_test_data = test_data["data"]
all_test_labels = test_data["label"][0]

threshold_values = [150, 200]


# skewness g1=∑(Yi−mean(Y))^3/N) / s3
def calculate_skew(arr):
    mean = np.mean(arr)
    sum_of_cubic_deviation = 0
    for elem in arr:
        sum_of_cubic_deviation += pow((elem - mean), 3)
    numerator = sum_of_cubic_deviation / len(arr)
    standard_deviation = np.std(arr)
    denominator = pow(standard_deviation, 3)
    return numerator / denominator


# Ratio of brighter to darker pixels ri using a threshold T
def calculate_ratio(arr, threshold):
    greater_count = 0
    lesser_count = 0
    for elem in arr:
        if elem >= threshold:
            greater_count += 1
        else:
            lesser_count += 1
    return greater_count / lesser_count


def get_normalized_value(value, mean, std):
    return (value - mean) / std


# mle mean is the sample mean
def find_mle_mean(x_train):
    mle_mean = np.mean(x_train, axis=0)
    print("mle mean: ", mle_mean)
    return mle_mean


# using the mle covariance matrix formula (please refer project report)
def find_mle_covariance(x_train, mle_mean):
    squared_diff_sum = 0
    for x in x_train:
        diff = np.array(x - mle_mean).reshape((2, 1))  # calculate Xi - mle_mean
        diff_transpose = np.transpose(diff)
        squared_diff = np.dot(diff, diff_transpose)
        squared_diff_sum += squared_diff
    mle_covariance = squared_diff_sum / len(x_train)
    print("mle covariance: ", mle_covariance)
    return mle_covariance


def get_classifier_discriminant_value(x, mean, covariance_inverse, covariance_det, prior):
    d = 2  # in our case, normal density dimension = 2
    diff = np.array(x - mean).reshape((2, 1))
    diff_transpose = np.transpose(diff)
    first_term = - 0.5 * np.matmul(np.matmul(diff_transpose, covariance_inverse), diff)
    second_term = - (d / 2) * math.log(2 * math.pi)
    third_term = - 0.5 * np.log(covariance_det)
    fourth_term = math.log(prior)
    g = first_term + second_term + third_term + fourth_term
    return g


# calculate the error rate of the optimal classifier
def calculate_error_rate(image_data, mle_mean_3, mle_mean_7, cov_inverse_3, cov_inverse_7, cov_det_3, cov_det_7,
                         prior_3, prior_7):
    pred_correct = 0
    for idx in range(len(image_data)):
        x = image_data[idx]
        g_3 = get_classifier_discriminant_value(x, mle_mean_3, cov_inverse_3, cov_det_3, prior_3)
        g_7 = get_classifier_discriminant_value(x, mle_mean_7, cov_inverse_7, cov_det_7, prior_7)
        y_pred = 0
        if g_3 > g_7:
            y_pred = 3
        else:
            y_pred = 7
        if y_pred == all_labels[idx]:
            pred_correct += 1
    error_rate = 1 - (pred_correct / len(image_data))
    return error_rate


# calculating the skewness and ratio of brighter to darker pixels for each image
def transform_image_data(all_data, t):
    transformed_data = []
    for image_data in all_data:
        init = np.array(image_data)
        # First we convert the 28*28 matrix into a 784 dimension vector
        flattened_image = init.flatten()
        # calculating skew
        skew = calculate_skew(flattened_image)
        # calculating ratio
        ratio = calculate_ratio(flattened_image, t)
        transformed_data.append([skew, ratio])
    return transformed_data


# normalize the data using the overall mean and standard deviation for each of the two features
def normalize_data(transformed_data, M, S):
    M1 = M[0]
    M2 = M[1]
    S1 = S[0]
    S2 = S[1]
    normalized_image_data = []
    for x in transformed_data:
        x1 = x[0]
        y1 = get_normalized_value(x1, M1, S1)
        x2 = x[1]
        y2 = get_normalized_value(x2, M2, S2)
        y = [y1, y2]
        normalized_image_data.append(y)
    return normalized_image_data


for t in threshold_values:
    print("Considering threshold: ", t)
    transformed_image_data = transform_image_data(all_data, t)
    transformed_image_test_data = transform_image_data(all_test_data, t)

    M = np.mean(transformed_image_data, axis=0)
    S = np.std(transformed_image_data, axis=0)
    normalized_image_data = normalize_data(transformed_image_data, M, S)
    normalized_image_test_data = normalize_data(transformed_image_test_data, M, S)

    # separate out the training data for the two classes
    class_3_train = []
    class_7_train = []
    for label_idx in range(len(all_labels)):
        if all_labels[label_idx] == 3:
            class_3_train.append(normalized_image_data[label_idx])
        else:
            class_7_train.append((normalized_image_data[label_idx]))

    '''
    calculating the MLE parameters for the normal distributions for class 3 and 7 training samples separately
    mean = sample mean
    covariance matrix = (please refer project report)
    '''
    print("calculating the MLE parameters for class 3")
    class_3_mle_mean = find_mle_mean(class_3_train)
    class_3_mle_covariance = find_mle_covariance(class_3_train, class_3_mle_mean)

    print("calculating the MLE parameters for class 7")
    class_7_mle_mean = find_mle_mean(class_7_train)
    class_7_mle_covariance = find_mle_covariance(class_7_train, class_7_mle_mean)

    prior_values = [[0.5, 0.5], [0.3, 0.7]]
    # calculating error rate for each set of prior values
    for priors in prior_values:
        class_3_prior = priors[0]
        class_7_prior = priors[1]
        print("considering class 3 prior, class 7_prior: ", str(class_3_prior), str(class_7_prior))
        # calculating the discriminating function value for classes 3 and 7 on entire data
        class_3_cov_det = np.linalg.det(class_3_mle_covariance)
        class_3_cov_inverse = np.linalg.inv(class_3_mle_covariance)
        class_7_cov_det = np.linalg.det(class_7_mle_covariance)
        class_7_cov_inverse = np.linalg.inv(class_7_mle_covariance)
        # calculating error rate on train data
        train_error_rate = calculate_error_rate(normalized_image_data, class_3_mle_mean, class_7_mle_mean,
                                                class_3_cov_inverse, class_7_cov_inverse, class_3_cov_det,
                                                class_7_cov_det, class_3_prior, class_7_prior)
        # calculating error rate on test data
        test_error_rate = calculate_error_rate(normalized_image_test_data, class_3_mle_mean, class_7_mle_mean,
                                               class_3_cov_inverse, class_7_cov_inverse, class_3_cov_det,
                                               class_7_cov_det, class_3_prior, class_7_prior)
        print("training set error rate: ", train_error_rate)
        print("test set error rate: ", test_error_rate)

    print("")
