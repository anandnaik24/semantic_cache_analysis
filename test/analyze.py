from collections import defaultdict
from dataset import read
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

model1 = "all-MiniLM-L6-v2"
model2 = "all-MiniLM-L12-v2"
model3 = "paraphrase-MiniLM-L3-v2"
model4 = "paraphrase-albert-small-v2"

models = [SentenceTransformer(model1), SentenceTransformer(model2), SentenceTransformer(model3), SentenceTransformer(model4)]


def calculate_histogram_for_model(model):

    id_to_question = {}
    similarity_question = defaultdict(lambda: set([]))
    encoded_questions = {}
    total = 0
    total_similar = 0
    duplicates_distances = []
    other_distances = []
    i = 0

    for data in read():
        if (i % 1000 == 0):
            print(i)
        i += 1
        if i == 10001:
            break
        total += 1
        qid1 = data["qid1"]
        qid2 = data["qid2"]
        q1 = data["q1"]
        q2 = data["q2"]
        duplicate = data["duplicate"]

        if qid1 not in id_to_question:
            id_to_question[qid1] = q1
        if qid2 not in id_to_question:
            id_to_question[qid2] = q2
        if duplicate:
            similarity_question[qid1].add(qid2)
            similarity_question[qid2].add(qid1)
            total_similar += 1

        if qid1 not in encoded_questions:
            encoded_questions[qid1] = model.encode(q1)

        if qid2 not in encoded_questions:
            encoded_questions[qid2] = model.encode(q2)

        present_distance = np.linalg.norm(encoded_questions[qid1] - encoded_questions[qid2])

        if duplicate:
            duplicates_distances.append(present_distance)
        else:
            other_distances.append(present_distance)

    print("Total pair", total)
    print("Total similar pair", total_similar)
    print("Num of question", len(id_to_question))

    max_len = 0
    len_dict = defaultdict(int)

    for key, value in similarity_question.items():
        len_dict[len(value)] += 1
        max_len = max(max_len, len(value))

    print("Max related questions ", max_len)
    print(len_dict)

    min_hist = np.floor(min(min(duplicates_distances), min(other_distances)))
    max_hist = np.ceil(max(max(duplicates_distances), max(other_distances)))

    num_bins = 21
    bin_edges = np.linspace(min_hist, max_hist, num_bins).tolist()
    print(bin_edges)

    print(min(duplicates_distances))
    print(max(duplicates_distances))
    print(min(other_distances))
    print(max(other_distances))

    duplicates_hist, bin_edges_1 = np.histogram(duplicates_distances, bin_edges)
    others_hist, _ = np.histogram(other_distances, bin_edges)

    return duplicates_hist, others_hist, bin_edges_1


duplicates_hist_all = []
others_hist_all = []
bin_edges_all = []

for model_name in models:
    duplicates_hist_, others_hist_, bin_edges_ = calculate_histogram_for_model(model_name)
    duplicates_hist_all.append(duplicates_hist_)
    others_hist_all.append(others_hist_)
    bin_edges_all.append(bin_edges_)


plt.subplot(2, 2, 1)
plt.plot(bin_edges_all[0][:-1], duplicates_hist_all[0], marker='o', linestyle='-', color='b', label='Duplicates')
plt.plot(bin_edges_all[0][:-1], others_hist_all[0], marker='o', linestyle='-', color='r', label='Non Duplicates')
plt.title(f'{model1}')
plt.xlabel('Distances')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(bin_edges_all[1][:-1], duplicates_hist_all[1], marker='o', linestyle='-', color='b', label='Duplicates')
plt.plot(bin_edges_all[1][:-1], others_hist_all[1], marker='o', linestyle='-', color='r', label='Non Duplicates')
plt.title(f'{model2}')
plt.xlabel('Distances')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(bin_edges_all[2][:-1], duplicates_hist_all[2], marker='o', linestyle='-', color='b', label='Duplicates')
plt.plot(bin_edges_all[2][:-1], others_hist_all[2], marker='o', linestyle='-', color='r', label='Non Duplicates')
plt.title(f'{model3}')
plt.xlabel('Distances')
plt.ylabel('Count')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(bin_edges_all[3][:-1], duplicates_hist_all[3], marker='o', linestyle='-', color='b', label='Duplicates')
plt.plot(bin_edges_all[3][:-1], others_hist_all[3], marker='o', linestyle='-', color='r', label='Non Duplicates')
plt.title(f'{model4}')
plt.xlabel('Distances')
plt.ylabel('Count')
plt.legend()

plt.suptitle('Histograms for Duplicate and Non Duplicate Meaning Quora Questions')

plt.tight_layout()
plt.show()
