# ETDTitlePageClassifier
In ETDs, sometimes the title page could not be the first page. This repository contains the code that checks the first n pages of ETDs and classify if each page is a title page or not.

## Rule-based-model 

### Feature Extraction with TF-IDF Vectorizer
To automatically pick the features to be used for the heuristic classifier, the `sklearn.feature_extraction.text.TfidfVectorizer` will be used. 

**Method:**
1. Extract the text content of title pages of 500 ETDs.
2. Get the top n features which can be used to identify the title pages.  

### Heuristic Classifier

**Assumption:** The title page is in the first 10 pages of an ETD. 

**Input:** The texts from ETDs are extracted using AWS Textract.  Those OCRed Text files are used as input into the heuristic classifier.

**Method:** 
1. Vectorize the text in first n pages using One-Hot Encoding word embedding technique. The features considered would be chosen from `feature_extraction_with_tfidf.py`. 
2. Then measure the cosine similarity between the query vector [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]. (Assume the number of features used is 20)
3. Iterate from page numbers 3 to 10, till the cosine similarity value of top ranked page is higher than a given threshold, and label it as 'title-page'. If none of the 10 pages has a cosine similarity value higher than the threshold, get the page with the highest cosine rank and label is as the "title-page".

**Output:** Text file for each ETD indicating the page name and the respective classification label. title-page or non-title-page

### Evaluate the model
To evaluate the heuristic model, the following labels are used.

**Labels**: A human judge labeled each page in ETDs.

Predicted values binary classifier are evaluated against the labels. And measured following evaluation metrics,
* True Positive, True Negative, False Positive, and False Negative
* Precision
* Recall
* Accuracy


## Instructions

### Setup
* **Python Version: 3.9**

### Install packages
* Install nltk:
  * `pip3 install nltk`

### Run feature extraction with tfidf
* This module will automatically pick the features to be used for the heuristic classifier using tfidf.
* Open the './Rule-based-model/feature_extraction_with_tfidf.py' file and add the number of features as num_of_features. (Default num_of_features = 20)
* Run `python ./Rule-based-model/feature_extraction_with_tfidf.py`
* The extracted features will be written into './Rule-based-model/features.txt' file.

### Run heuristic classifier
* This module will read the files which contains OCR extracted text of ETDs in the '/Data/Input' folder, get the features given in the './Rule-based-model/features.txt' file, generate the labels (title-page or non-title-page) for each ETD, and write the result to the 'Rule-based-model/Data/Output' folder.
* Update the `threshold` variable value with the minimum cosine similarity value you expect the title page should have (Default threshold = 0.9)
* `python ./Rule-based-model/heuristic_classifier.py`

### Run the evaluator
* This module will get the Labels of first 3 pages from '/Data/Labels' folder and measure evaluation metrics for the predicted labels.
* `python ./Rule-based-model/evaluate_model.py`