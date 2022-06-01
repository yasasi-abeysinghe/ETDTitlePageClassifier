import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


def get_true_y(filename):
    y = []
    file = open(filename, "r")
    content = file.read()
    first_3_pages = content.split("\n")

    for page in first_3_pages:
        if page.split(", ")[1] == "title-page":
            y.append(1)
        else:
            y.append(0)

    return y


def get_pred_y(filename):
    y = []

    file = open(filename, "r")
    content = file.read()
    first_3_pages = content.split("\n")

    for page in first_3_pages:
        if page.split(", ")[1] == "title-page":
            y.append(1)
        else:
            y.append(0)

    return y


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
    y_true = []
    y_pred = []
    for i in range(500):
        label_filename = "../Data/Labels/label_" + str(i+1) + ".txt"
        y1 = get_true_y(label_filename)

        output_filename = "./Output/label_" + str(i+1) + ".txt"
        y2 = get_pred_y(output_filename)

        diff = len(y1) - len(y2)
        if diff > 0:
            for j in range(diff):
                y2.append(0)
        elif diff < 0:
            y2 = y2[:diff]

        y_true.extend(y1)
        y_pred.extend(y2)

    get_confusion_matrix_values(y_true, y_pred)
    get_precision_recall_accuracy_scores(y_true, y_pred)
