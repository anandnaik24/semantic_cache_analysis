# Semantic Cache Analysis

This repo is clone of [langcache](https://github.com/jiashenC/langcache) and certain features like dynamic distance threshold tuning and analysis of the [Quora Question Pairs dataset](https://www.kaggle.com/competitions/quora-question-pairs/data)
 have been added to it.

Dynamic distance threshold tuning gives us control over how we can tune the distance threshold i.e. set different sensitivity to false positives and false negatives, set a sensitivity rate to decrease the impact of false positives and false negatives to the distance threshold as training on the dataset progresses.

## How to use

Step 0: pip install all requirements mentioned in the requirements.txt file 

Step 1: Download the Quora Question Pairs dataset and add it in the test repo

[Quora Question Pairs dataset](https://www.kaggle.com/competitions/quora-question-pairs/data)

Step 2: Run the analyse.py script to get the histogram of distances between question pairs for duplicate and non-duplicate questions based on 4 different sentence feature extractor models (`all-MiniLM-L6-v2`, `all-MiniLM-L12-v2`, `paraphrase-MiniLM-L3-v2`, `paraphrase-albert-small-v2`)

Step 3: Run the `test/large_test.py` file

We can see how the distance threshold changes based on the number of questions that are sampled from the dataset. To change the behaviour the following parameters in the corresponding files can be changed

File - `test/large_test.py`
Parameters - `number_of_questions`, `tune_policy` (can be "dynamic" or "balance")

File - `test/core.py`
Parameters - `distance_threshold`, `sensitivity_fn`, `sensitivity_fp`, `sensitivity_rate`, `sensitivity_fn_min`, `sensitivity_fp_min`

A final plot which shows how distance threshold changes with training on each false positive or false negative classified question. The plot also has statistics like true positives, false negatives, false positives, true negatives in the title.
