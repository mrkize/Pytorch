import numpy as np

path = "results/CIFAR100_training_size_2023_02_14_19_37_22/"
accuracy_path = path + "res_accuracy.npy"
accuracy = np.load(accuracy_path)
precision_path = path + "res_precision.npy"
precision = np.load(precision_path)
recall_path = path + "res_recall.npy"
recall = np.load(recall_path)
print("accuracy: {}".format(accuracy))
print("precision: {}".format(precision))
print("recall: {}".format(recall))