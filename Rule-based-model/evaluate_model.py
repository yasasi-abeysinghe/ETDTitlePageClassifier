import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


def get_true_y():
    path = "./Data/Input/"
    dir_list = os.listdir(path)

    y_true = []

    for filename in dir_list:
        file = open(path + filename, "r")
        content = file.read()
        first_3_pages = content.split("\n")[:3]

        for page in first_3_pages:
            if page.split(", ")[1] == "title-page":
                y_true.append(1)
            else:
                y_true.append(0)

    return y_true


def get_pred_y():
    path = "./Rule-based-model/Output/"
    dir_list = os.listdir(path)

    y_pred = []

    for filename in dir_list:
        file = open(path + filename, "r")
        content = file.read()
        first_3_pages = content.split("\n")

        for page in first_3_pages:
            if page.split(", ")[1] == "title-page":
                y_pred.append(1)
            else:
                y_pred.append(0)

    return y_pred


def get_confusion_matrix_values(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("True Positive  : " + str(tp))
    print("True Negative  : " + str(tn))
    print("False Positive : " + str(fp))
    print("False Negative : " + str(fn))


def get_precision_recall_accuracy_scores(y_true, y_pred):
    print("\nPrecision: " + str(precision_score(y_true, y_pred)))
    print("Recall: " + str(recall_score(y_true, y_pred)))
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))


if __name__ == "__main__":
    y_true = get_true_y()
    y_pred = get_pred_y()

    get_confusion_matrix_values(y_true, y_pred)
    get_precision_recall_accuracy_scores(y_true, y_pred)
