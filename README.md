# TASK 1

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

## Approach 3 - multipart classification model (extension of approach 1)
**steps:**
* Find multipart spoiler with a Gradient Boosting classification model trained on text features of the attributes postText and targetParagraphs.
* Run the transformer baseline on features that have not been classified as multipart.

**goals:**
* Reach a similar accuracy for multipart spoiler as the transformer baseline.
* Improve the efficiency due to less time for training and forward-pass of the Gradient Boosting classification model than the transformer baseline needs.

**code:**
* [feature engineering](statistical-model-multi-classification/multipart_spoiler_detection_model_features.ipynb)

* [feature engineering refactored](statistical-model-multi-classification/multipart_detection.py)

* [feature engineering, training and evaulation of the Gradient Boosting Classifier](statistical-model-multi-classification/multipart_spoiler_detection_model.ipynb)
* [two-step pipeline](statistical-model-multi-classification/two-step-transformer.py)
