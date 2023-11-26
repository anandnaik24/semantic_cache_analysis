# Semantic Cache Analysis

When we use any image generative AI application, we would input a query that sounds intuitive to us. However, the application would not generative the required image because we are not able to provide a descriptive prompt. This descriptive prompt is not intuitive to us. So, we can start with a prompt that is intuitive, search for similar prompts in a database of prompts and and keep refining our prompt till we are satisfied that it does describe the image that we want to generate and then only, input the prompt to the application as generation of images is quite expensive computationally.

## How to use

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
