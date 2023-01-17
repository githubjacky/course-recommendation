import sys
import numpy as np
import csv


def predict(result, path, user_id, domain='course'):
    output = []
    for i in range(len(result)):
        pred = ""
        for j in result[i]:
            pred += (str(j) + " ")
        pred = pred.strip()
            
        output.append([user_id[i], pred])
    
    if domain == 'course':
            head = ["user_id", "course_id"]
    else:
            head = ["user_id", "subgroup"]

    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(head)
        writer.writerows(output)


def predict_topic_from_course(result, user_idx, known_topic, course2subgroup): 
    pred_boost = []
    for idx, i in enumerate(result):
        weight = dict()
        for j in i:
            try:
                topic = course2subgroup[str(j)]
            except:
                topic = [0]
            for k in topic:
                if k in weight:
                    weight[k] += 1
                else:
                    weight[k] = 1
        secret = known_topic[user_idx[idx]]
        key  = list(weight.keys())
        val = list(weight.values())
        pred = [
            key[j]
            for j in np.argsort(np.array(val))[::-1]
            if key[j] not in secret
        ]
        pred_boost.append(secret+pred)
    return pred_boost


def predict_topic_from_course_no_secret(result, course2subgroup): 
    pred_boost = []
    for idx, i in enumerate(result):
        weight = dict()
        for j in i:
            try:
                topic = course2subgroup[str(j)]
            except:
                topic = [0]
            for k in topic:
                if k in weight:
                    weight[k] += 1
                else:
                    weight[k] = 1
        key  = list(weight.keys())
        val = list(weight.values())
        pred = [
            key[j]
            for j in np.argsort(np.array(val))[::-1]
            if key[j]
        ]
        pred_boost.append(pred)
    return pred_boost


def knn_predict_course(indices, user_idx, known_course):
    pred = []
    for i in user_idx:
        known = known_course[i]
        weight = dict()
        for j in indices[i][1:]:
            for k in known_course[j]:
                if k not in known and k in weight:
                    weight[k] += 1
                elif k not in known and k not in weight:
                    weight[k] = 1
                else:
                    continue
        key  = list(weight.keys())
        val = list(weight.values())
        pred.append([
            key[j]
            for j in np.argsort(np.array(val))[::-1]
        ])
    return pred

def knn_predict_topic(indices, user_idx, known_topic):
    pred_boost = []
    for i in user_idx:
        weight = dict()
        for j in indices[i][1:]:
            for k in known_topic[j]:
                if k in weight:
                    weight[k] += 1
                elif k not in weight:
                    weight[k] = 1
                else:
                    continue
        secret = known_topic[i]
        key  = list(weight.keys())
        val = list(weight.values())
        pred = [
            key[j]
            for j in np.argsort(np.array(val))[::-1]
            if key[j] not in secret
        ]
        pred_boost.append(secret+pred)
    return pred_boost

def mix2_rearrange(base, ref1):
    pred = []
    for (idx, i) in enumerate(base):
        weight = dict()
        ref = ref1[idx]
        for idx_j, j in enumerate(i):
            weight[j] = idx_j
            if j in ref:
                weight[j] += (np.where(np.array(ref) == j)[0][0])
            else:
                weight[j] += len(ref)
        
        key  = list(weight.keys())
        val = list(weight.values())
        pred.append([
            key[j]
            for j in np.argsort(np.array(val))
        ])
    return pred


def mix3_rearrange(base, ref1, ref2):
    pred = []
    for (idx, i) in enumerate(base):
        weight = dict()
        ref_1, ref_2 = ref1[idx], ref2[idx]
        for idx_j, j in enumerate(i):
            weight[j] = idx_j
            if j in ref_1 and j in ref_2:
                weight[j] += (np.where(np.array(ref_1) == j)[0][0] + np.where(np.array(ref_2) == j)[0][0])
            elif j in ref_1:
                weight[j] += (np.where(np.array(ref_1) == j)[0][0] + len(ref_2))
            elif j in ref_2:
                weight[j] += (len(ref_1) + np.where(np.array(ref_2) == j)[0][0])
            else:
                weight[j] += (len(ref_1) + len(ref_2))
        
        key  = list(weight.keys())
        val = list(weight.values())
        pred.append([
            key[j]
            for j in np.argsort(np.array(val))
        ])
    return pred