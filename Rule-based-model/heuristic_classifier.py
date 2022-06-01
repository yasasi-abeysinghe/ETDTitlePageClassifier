import os
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from read_ocr_text import get_first_n_pages

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def tokenize_text(page_text):
    page_text = page_text.lower()
    tokenized_text = word_tokenize(page_text)
    return tokenized_text


def text_vectorization(tokenized_text):
    vector = []
    for feature in features:
        if feature in tokenized_text:
            vector.append(1)
        else:
            vector.append(0)
    return vector


def get_cosine_similarities(text_matrix):
    query_matrix = np.ones((1, len(features)), dtype=int)
    cosine_similarities = cosine_similarity(query_matrix, text_matrix)
    return cosine_similarities


def sort_similarity_score(cosine_similarities, no_of_pages):
    similarity_scores = []
    for i in range(no_of_pages):
        similarity_scores.append((cosine_similarities[i], "etd_page" + str(i + 1)))
    similarity_scores.sort(reverse=True)
    top = similarity_scores[0][0]
    if top > threshold:
        return similarity_scores
    else:
        if no_of_pages == 10:
            return similarity_scores
        else:
            return []


# check the cosine score and if that is less than 0.63.... then go for next page and calculate cosine similarity
# till that is above 0.64? till 10 pages. After 10, assign the title page label whatever page with the highest
# cosine score
def refactor_page_labels(page_labels):
    i = 0
    flag = False
    new_page_label_list = []
    for page in page_labels:
        page_label = page[1]
        if i < 3:
            new_page_label_list.append(page)
            if page_label == "title-page":
                flag = True
        elif not flag:
            new_page_label_list.append(page)
            if page_label == "title-page":
                break
        i += 1
    return new_page_label_list


def get_labels_for_pages(similarity_scores):
    page_labels = []
    top = True
    for value in similarity_scores:
        if top:
            new_tuple = value[1:] + ("title-page",)
            top = False
        else:
            new_tuple = value[1:] + ("non-title-page",)
        page_labels.append(new_tuple)
    page_labels.sort(key=lambda x: int(x[0][8:]))
    return refactor_page_labels(page_labels)


def write_labels(page_labels, output_file):
    file = open(output_file, "w")
    file.write('\n'.join("{}, {}".format(x[0], x[1]) for x in page_labels))


def classify_ETD(etd_text_file, output_file):
    for n in range(3, 11):
        print(etd_text_file)
        print(n)
        first_n_pages = get_first_n_pages(etd_text_file, n)

        matrix = []
        for page in first_n_pages:
            tokenized_text = tokenize_text(page)
            matrix.append(text_vectorization(tokenized_text))

        arr = np.array(matrix)
        cosine_similarities = get_cosine_similarities(arr)[0]
        print(cosine_similarities)
        similarity_scores = sort_similarity_score(cosine_similarities, n)
        if similarity_scores:
            page_labels = get_labels_for_pages(similarity_scores)
            write_labels(page_labels, output_file)
            break


def get_features():
    file = open("./Rule-based-model/features.txt", "r")
    content = file.read()
    return content.split(", ")


if __name__ == "__main__":

    features = get_features()
    threshold = 0.9  # Update the threshold here

    input_path = "./Data/Input/"
    output_path = "./Rule-based-model/Output/"
    dir_list = os.listdir(input_path)

    dir_list.sort(key=lambda x: int(x[4:-4]))

    for filename in dir_list:
        input_file = input_path + filename
        output_file = output_path + filename.replace("ETD", "label")
        classify_ETD(input_file, output_file)
