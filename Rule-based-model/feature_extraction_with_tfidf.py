import os
from read_ocr_text import get_first_n_pages
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# Return the list of page numbers which represent the title page of each ETD (length is 500)
def get_title_pages_of_etds():
    title_pages_of_etds = []

    label_path = "../Data/Labels/"
    dir_list = os.listdir(label_path)

    dir_list.sort(key=lambda x: int(x[6:-4]))

    for filename in dir_list:
        file = open(label_path + filename, "r")
        content = file.read()
        pages = content.split("\n")

        for page_string in pages:
            page = page_string.split(", ")
            if page[1] == "title-page":
                title_pages_of_etds.append(page[0].split("etd_page")[1])

    return title_pages_of_etds


def generate_train_set():
    title_pages_of_etds = get_title_pages_of_etds()
    train_document_set = []

    input_path = "../Data/Input/"
    dir_list = os.listdir(input_path)

    dir_list.sort(key=lambda x: int(x[4:-4]))

    i = 0

    for etd_text_file in dir_list:
        # If title page isn't within the first 10 pages, we will ignore that etd.
        # Hence, we will only consider the first 10 pages
        first_10_pages = get_first_n_pages(input_path + etd_text_file, 10)

        title_page_number = title_pages_of_etds[i]
        text_content_of_title_page = first_10_pages[int(title_page_number) - 1]

        train_document_set.append(text_content_of_title_page)

        i += 1

    return train_document_set


def feature_extraction(num_of_features):
    train_document_set = generate_train_set()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_of_features)
    X = vectorizer.fit_transform(train_document_set)
    terms = vectorizer.get_feature_names()

    df = pd.DataFrame(X.toarray(), columns=terms)
    df.to_csv("title-page-features.csv", sep='\t')

    return terms


if __name__ == "__main__":
    num_of_features = 20
    terms = feature_extraction(num_of_features)
    print("The top " + str(num_of_features) + " features: ")
    print(terms)
