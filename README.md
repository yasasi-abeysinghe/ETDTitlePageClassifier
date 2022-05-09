# ETDTitlePageClassifier
In ETDs, sometimes the title page could not be the first page. This repository contains the code that checks the first n pages of ETDs and classify if each page is a title page or not.

## Rule-based-model 

### Heuristic Classifier

**Assumption:** The title page is in the first 3 pages of an ETD. Hence, n=3.

**Input:** The texts from ETDs are extracted using AWS Textract.  Those OCRed Text files are used as input into the heuristic classifier.

**Method:** 
1. Vectorize the text in first n pages using One-Hot Encoding word embedding technique. The features considered would be ["partial", "fulfillment", "requirements"]. 
2. Then measure the cosine similarity between the query vector [1, 1, 1].
3. Then, find the top ranked page and labeled it as 'title-page'

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

### Run heuristic classifier
* This module will read the files which contains OCR extracted text of ETDs in the 'Rule-based-model/Data/Input' folder, generate the labels (title-page or non-title-page) for first 3 pages in each ETD, and write the result to the 'Rule-based-model/Data/Output' folder 
* `python ./Rule-based-model/heuristic_classifier.py`

### Run the evaluator
* This module will get the Labels of first 3 pages from 'Rule-based-model/Data/Labels' folder and measure evaluation metrics for the predicted labels.
* `python ./Rule-based-model/evaluate_model.py`