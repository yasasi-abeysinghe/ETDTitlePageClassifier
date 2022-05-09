from read_ocr_text import get_first_n_pages
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def tokenize_text(page_text):
    page_text = page_text.lower()
    tokenized_text = word_tokenize(page_text)
    return tokenized_text


def text_vectorization(tokenized_text):
    features = ["partial", "fulfillment", "requirements"]
    vector = []
    for feature in features:
        if feature in tokenized_text:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def get_cosine_similarities(text_matrix):
    query_matrix = np.array([[1, 1, 1]])
    cosine_similarities = cosine_similarity(query_matrix, text_matrix)
    return cosine_similarities


def classify_pages(cosine_similarities):
    page_labels = []
    similarity_scores = []
    for i in range(n):
        similarity_scores.append((cosine_similarities[i], "etd1_page" + str(i + 1)))
    similarity_scores.sort(reverse=True)
    top = True
    for value in similarity_scores:
        if top:
            new_tuple = value[1:] + ("title-page",)
            top = False
        else:
            new_tuple = value[1:] + ("non-title-page",)
        page_labels.append(new_tuple)
    page_labels.sort()
    return page_labels


def write_labels(page_labels):
    file = open("Rule-based-model/Data/Output/label_1.txt", "w")
    print(page_labels)
    file.write('\n'.join("{}, {}".format(x[0], x[1]) for x in page_labels))


if __name__ == "__main__":
    n = 3
    first_n_pages = get_first_n_pages("Rule-based-model/Data/Input/ETD_1.txt", n)

    matrix = []
    for page in first_n_pages:
        tokenized_text = tokenize_text(page)
        matrix.append(text_vectorization(tokenized_text))

    arr = np.array(matrix)
    cosine_similarities = get_cosine_similarities(arr)[0]
    page_labels = classify_pages(cosine_similarities)
    write_labels(page_labels)
