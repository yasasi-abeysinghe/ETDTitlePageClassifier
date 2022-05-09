from sklearn.metrics import confusion_matrix, precision_score, recall_score


def get_true_y():
    file = open("./Data/Labels/label_1.txt", "r")
    content = file.read()
    first_3_pages = content.split("\n")[:3]

    y_true = []

    for page in first_3_pages:
        if page.split(", \t")[1] == "Label-TitlePage":
            y_true.append(1)
        else:
            y_true.append(0)

    return y_true


def get_pred_y():
    file = open("./Data/Output/label_1.txt", "r")
    content = file.read()
    first_3_pages = content.split("\n")

    y_pred = []

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


def get_precision_recall_scores(y_true, y_pred):
    print("\nPrecision: " + str(precision_score(y_true, y_pred)))
    print("Recall: " + str(recall_score(y_true, y_pred)))


if __name__ == "__main__":
    y_true = get_true_y()
    y_pred = get_pred_y()

    get_confusion_matrix_values(y_true, y_pred)
    get_precision_recall_scores(y_true, y_pred)
