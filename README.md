# TASK 1

Relevant for submission is trhe "two-step-transformer" of approach 3. Instructions for running in docker can be found below.

## Approach 1 - multipart majority vote
**steps:** 
* Find multipart spoiler with a majority vote of certain text features of the attributes postText and targetParagraphs.
* Run the transformer baseline on features that have not been classified as multipart. 

**goals:** 
* Reach a similar accuracy for multipart spoiler as the transformer baseline.
* Improve the efficiency due to less data that needs to be predicted by the transformer baseline.

**code:**
* [accuracy calculation of certain text features](statistical-model-multi-classification/baseline_calculations.ipynb)
* [majority voting](statistical-model-multi-classification/majority_vote.ipynb)

## Approach 2 - text simplification

**steps:** 

- Simplify the input text before predicting the spoiler type
- Either by just replacing difficult words ([MILES](https://github.com/Kvasirs/MILES))
- Or replacing words and reformulating ([MUSS](https://github.com/facebookresearch/muss))
- Predict spoiler type using the baseline transformer

**goals:**

- Better accuracy due to a smaller set of words / reduced complexity

**code:**
* [MILES transformer](miles-transformer-task-1)
* [MUSS transformer](muss-transformer-task-1)

**results:**

- Only replacing difficult words did not enhance the accuracy.
- Due to problems with the libraries, the approach has been discarded.

## Approach 3 - multipart classification model (extension of approach 1)
**steps:**
* Find multipart spoiler with a Gradient Boosting classification model trained on text features of the attributes postText and targetParagraphs.
* Run the transformer baseline on features that have not been classified as multipart.

**goals:**
* Reach a similar accuracy for multipart spoiler as the transformer baseline.
* Improve the efficiency due to less time for training and forward-pass of the Gradient Boosting classification model than the transformer baseline needs.

**code:**
* [feature engineering](statistical-model-multi-classification/multipart_spoiler_detection_model_features.ipynb)

* [feature engineering refactored](statistical-model-multi-classification/multipart_features.py)

* [feature engineering, training and evaulation of the Gradient Boosting Classifier](statistical-model-multi-classification/multipart_spoiler_detection_model.ipynb)

* [two-step pipeline](statistical-model-multi-classification/two-step-transformer.py)

**results:**

- Faster than baseline, as multipart spoilers are classified within seconds
- Probably similar or better accuracy, depending on the dataset

**Running in Docker:**

(Optional) Build:

1. `cd statistical-model-multi-classification`
2. `docker build -t ghcr.io/tudbs-clickbait/team-1-task-1:two-step .`
3. (optional) [Login to GitHub registry ](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry)
4. (optional, push image) `docker push ghcr.io/tudbs-clickbait/team-1-task-1:two-step`

Run:

1. `docker run -v ${PWD}/data:/data ghcr.io/tudbs-clickbait/team-1-task-1:two-step --input=/data/validation_short.jsonl --output=/data/out.jsonl`