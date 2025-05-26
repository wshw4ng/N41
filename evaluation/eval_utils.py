import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from eTaPR_pkg import etapr
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm.notebook import trange, tqdm
import ray
from psutil import cpu_count

class Point_Adjust:
    def __init__(self):
        self._attack_scores = 0
        self._normal_scores = 0
        self._metric_threshold = 1e-10
        
        self._pred_labels = 0
        self._labels = 0

    def set(self, pred_labels, labels):
        assert len(pred_labels) == len(labels), "#prediction labels and #labels are different"
        # assert 0.0 <= metric_threshold <= 1.0, "The threshold of metric should be 0 to 1"

        self._pred_labels = pred_labels
        self._labels = labels

        self._normal_scores = pred_labels[labels == 0.0] #label이 normal인 영역에 부여한 점수 [ 1, 0, 0, 1, 1, 0, ... ]
        self._attack_scores = []               #label이 attack인 영역에 부여한 점수 [ [ 1, 0, 0, 1 ], [1, 0, 0], ... ]

        # self._metric_threshold = metric_threshold
        
        temp_range = []
        for i in range(len(labels)-1):            
            if labels[i] == 1:
                temp_range.append(pred_labels[i])

            if labels[i] == 1 and labels[i+1] == 0:
                self._attack_scores.append(temp_range)
                temp_range = []
            
        if len(temp_range) != 0:
            self._attack_scores.append(temp_range)

    def get_scores(self):        
        miss_counts = sum(self._normal_scores)
        
        hit_counts = 0
        for scores in self._attack_scores:                        
            if float(sum(scores)/len(scores)) >= self._metric_threshold:
                hit_counts += len(scores)                
        
        p = float(hit_counts)/(hit_counts + miss_counts)
        r = float(hit_counts)/sum(self._labels)        
        f1 = 2/(1/p+1/r)

        return { "precision": p, 
                 "recall": r, 
                 "f1": f1 }

    # def get_auc(self):
    #     thds = sorted(self._predictions)
    #     FPR_TPR = []
                
    #     for i, thd in enumerate(thds):
    #         if i % 1000 == 0:
    #             miss_counts = sum(s > thd for s in self._normal_scores)

    #             hit_counts = 0
    #             for scores in self._attack_scores:
    #                 temp_counts = sum(s > thd for s in scores)
    #                 if float(temp_counts/len(scores)) >= self._metric_threshold:
    #                     hit_counts += len(scores)

    #             FPR = float(miss_counts) / (len(self._labels) - sum(self._labels))
    #             TPR = float(hit_counts)/sum(self._labels)
    #             FPR_TPR.append([FPR, TPR])
            
    #         cnt += 1
        
    #     auc = 0.0
    #     FPR_TPR = sorted(FPR_TPR, key= lambda x: x[0])
    #     for i in range(len(FPR_TPR)):
    #         if i < len(FPR_TPR) - 1:
    #             auc += (FPR_TPR[i+1][0] - FPR_TPR[i][0]) * FPR_TPR[i][1]
    #         elif i == 0:
    #             auc += FPR_TPR[i][0] * FPR_TPR[i][1] + (FPR_TPR[i+1][0] - FPR_TPR[i][0]) * FPR_TPR[i][1]
    #         else:
    #             auc += (1.0 - FPR_TPR[i][0]) * FPR_TPR[i][1]

    #     return auc


def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

def my_auc(pred, label):
    assert(len(pred) == len(label))

    thds = sorted(pred)
    FPR_TPR = []
    FPRs = []
    TPRs = []

    cnt = 0
    for thd in thds:
        if cnt % 100 == 0:
            pred_label = put_labels(pred, thd)

            FPR = (sum(pred_label) - np.dot(pred_label, label)) / (len(label) - sum(label))
            TPR = np.dot(pred_label, label)/sum(label)
            FPRs.append(FPR)
            TPRs.append(TPR)
            FPR_TPR.append([FPR, TPR])
        
        cnt += 1
    
    auc = 0.0
    FPR_TPR = sorted(FPR_TPR, key= lambda x: x[0])
    for i in range(len(FPR_TPR)):
        if i < len(FPR_TPR) - 1:
            auc += (FPR_TPR[i+1][0] - FPR_TPR[i][0]) * FPR_TPR[i][1]
        elif i == 0:
            auc += FPR_TPR[i][0] * FPR_TPR[i][1] + (FPR_TPR[i+1][0] - FPR_TPR[i][0]) * FPR_TPR[i][1]
        else:
            auc += (1.0 - FPR_TPR[i][0]) * FPR_TPR[i][1]

    return auc

#pred: 예측 점수, label: 실제 정답 라벨, boundary: 임계값의 범위, n_bins: 검사할 갯수
def find_best_f1_10way(pred, label, min_thd, max_thd, n_bins):
    f1_scores = []
    term = (max_thd - min_thd)/(n_bins-1)
    for i in range(n_bins):
        pred_labels = put_labels(pred, min_thd + i*term)
        f1_scores.append(f1_score(label, pred_labels))
    
    max_id = f1_scores.index(max(f1_scores))

    if f1_scores[max(max_id-1, 0)] == f1_scores[max_id] == f1_scores[min(max_id+1, n_bins-1)]:
        return min_thd + max_id*term, f1_scores[max_id]
    else:
        return find_best_f1_10way(pred, label, max(min_thd + max_id*term - term/2, min_thd), min(min_thd + max_id*term + term/2, max_thd), n_bins)

#pred: 예측 점수, att: 실제 정답 라벨, sorted_err: 정렬된 예측 점수(임계값의 후보로 사용), min_idx, max_idx: 검사할 후보값의 범위, n_bins: 한번에 검사할 갯수
def find_best_f1(pred, att, sorted_err, min_idx, max_idx, n_bins = 1001):
    # 검사할 후보의 수가 1000개 이하인 경우, 모든 후보 검사 후 종료
    if max_idx - min_idx < 1000:
        pred_labels = put_labels(pred, sorted_err[min_idx])
        max_score = f1_score(att, pred_labels)
        # max_score = recall_score(att, pred_labels)
        maxscr_idx = min_idx
        for i in range(min_idx, max_idx+1):            
            pred_labels = put_labels(pred, sorted_err[i])
            temp_score = f1_score(att, pred_labels)
            # temp_score = recall_score(att, pred_labels)
            if max_score < temp_score:
                max_score = temp_score
                maxscr_idx = i
        return sorted_err[maxscr_idx], max_score

    n_buckets = n_bins
    bucket_boundaries = []
    bucket_size = int(np.floor((max_idx-min_idx)/n_buckets))

    for i in range(n_buckets):
        bucket_boundaries.append(min_idx + i*bucket_size)

    sample_f1_scores = []
    for i in range(0, len(bucket_boundaries)):
        mid_idx = bucket_boundaries[i] + int((bucket_size-1)/2)   
        pred_labels = put_labels(pred, sorted_err[mid_idx])
        temp_score = f1_score(att, pred_labels)  
        # temp_score = recall_score(att, pred_labels)
        sample_f1_scores.append(temp_score)
    # mid_idx = int((max_idx-bucket_boundaries[-1])/2)
    # sample_f1_scores.append(f1_score[mid_idx])

    max_idx_in_sample = np.where(max(sample_f1_scores)==sample_f1_scores)[0][0]

    min_bound_idx = max(0, max_idx_in_sample-1)
    max_bound_idx = max_idx_in_sample + 1
    
    if max_idx_in_sample == len(bucket_boundaries)-1:
        return find_best_f1(pred, att, sorted_err, bucket_boundaries[min_bound_idx], max_idx, n_bins)
    else:            
        return find_best_f1(pred, att, sorted_err, bucket_boundaries[min_bound_idx], bucket_boundaries[max_bound_idx], n_bins)


def find_min_boundary_by_fpr(pred, label, min_fpr):
    candidate_thds = sorted(pred[label == 0], reverse=True)
    print("There are {} candidates for finding minimum boundary of threshold".format(len(candidate_thds)))

    min_thd = 0.0
    for i, thd in enumerate(candidate_thds):
        pred_label = put_labels(pred, thd)
        FPR = (sum(pred_label) - np.dot(pred_label, label)) / (len(label) - sum(label))

        if FPR > min_fpr:
            min_thd = thd
            break

        if i % 1000 == 0:
            print('{}: FPR is {}'.format(i, FPR))
        
    return min_thd

def find_boundary_thd(pred, label, candidate_thds, min_prec):
    candidate_thds = sorted(pred[label[WINDOW_SIZE-1:] == 1])
    print("There are {} candidates".format(len(candidate_thds)))

    min_thd, max_thd = 0, 0
    for i in range(len(candidate_thds)):        
        pred_labels = put_labels(pred, candidate_thds[i])
        if min_prec <= precision_score(label, pred_labels):
            min_thd = candidate_thds[i]
            break

    for i in range(len(candidate_thds)-1, -1, -1):
        pred_labels = put_labels(pred, candidate_thds[i])
        if min_prec <= precision_score(label, pred_labels):
            max_thd = candidate_thds[i]
            break

    return min_thd, max_thd

def check_graph(xs, att, piece=2):
    l = xs.shape[0]
    peak = 1.1
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])
        axs[i].set_ylim([0.0, peak])
        if len(xs[L:R]) > 0:            
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
    plt.show()

from scipy.spatial.distance import jensenshannon

def js_dist(attack_label, predictions):
    normal_pred = predictions[attack_label == 0]
    anomal_pred = predictions[attack_label == 1]

    max_val = max(predictions)
    min_val = min(predictions)

    normal_dist = np.histogram(normal_pred, bins=np.arange(min_val, max_val, (max_val-min_val)*0.001))[0]
    anomal_dist = np.histogram(anomal_pred, bins=np.arange(min_val, max_val, (max_val-min_val)*0.001))[0]

    return jensenshannon(normal_dist, anomal_dist)

def evaluate_w_threshold(attack_label, predictions, threshold):   
    assert len(attack_label)==len(predictions), "Error:The number of predictions and attack labels are not identical"

    pred_label = put_labels(predictions, threshold)

    pa = Point_Adjust()
    pa.set(pred_label, attack_label)
    res = pa.get_scores()

    et = etapr.evaluate_w_streams(anomalies=attack_label, 
                                predictions=pred_label, 
                                theta_p = 1e-1, 
                                theta_r = 1e-5)

    # check_graph(pred_label, attack_label, 3)

    return { "precision": precision_score(attack_label, pred_label),
               "recall": recall_score(attack_label,pred_label),
               "f1": f1_score(attack_label, pred_label),
               "pa_precision": res['precision'],
               "pa_recall": res['recall'],
               "pa_f1": res['f1'],
               "eTaP": et['eTaP'],
               "eTaR": et['eTaR'],
               "eTaf1": et['f1'],
               "eTaRd": et['eTaRd']
            }

def print_result(result, is_brief=False):
    if is_brief == False:
        print("precision:", result['precision'])
        print("recall:", result['recall'])
        print("f1:", result['f1'])    
        print()

        print("point-adjust precision:", result['pa_precision'])
        print("point-adjust recall:", result['pa_recall'])
        print("point-adjust f1:", result['pa_f1'])    
        print()

        print("eTaP: ", result['eTaP'])
        print("eTaR: ", result['eTaR'])
        print("eTaF1: ", result['eTaf1'])
        print("eTaRd: ", result['eTaRd'] )
        print()
    else:
        print(result['precision'])
        print(result['recall'])
        print(result['f1'])    
        print(result['pa_precision'])
        print(result['pa_recall'])
        print(result['pa_f1'])    
        print(result['eTaP'])
        print(result['eTaR'])
        print(result['eTaf1'])
        print(result['eTaRd'] )
        print()


def evaluate_all(attack_label, predictions):
    print("Jensen-Shannon Dist.:", js_dist(attack_label, predictions))
    print()

    sorted_predictions = sorted(predictions)

    print("\nAccuracies without FPR ...")
    thd, _ = find_best_f1(predictions, attack_label, sorted_predictions, 0, len(predictions)-1)
    print("Threshold: ", thd)
    check_graph(predictions, attack_label, 3)
    print_result(evaluate_w_threshold(attack_label, predictions, thd))

    print("\nAccuracies with FPR (1e-3) ...")
    min_thd = find_min_boundary_by_fpr(predictions, attack_label, 1e-3)
    thd, _ = find_best_f1(predictions, attack_label, sorted_predictions, np.where(sorted_predictions == min_thd)[0][0], len(predictions)-1)
    print("Threshold: ", thd)
    check_graph(predictions, attack_label, 3)
    print_result(evaluate_w_threshold(attack_label, predictions, thd))

    print("\nAccuracies with FPR (1e-4) ...")
    min_thd = find_min_boundary_by_fpr(predictions, attack_label, 1e-4)
    thd, _ = find_best_f1(predictions, attack_label, sorted_predictions, np.where(sorted_predictions == min_thd)[0][0], len(predictions)-1)
    print("Threshold: ", thd)
    check_graph(predictions, attack_label, 3)
    print_result(evaluate_w_threshold(attack_label, predictions, thd))


# 최대 점수(임계값을 FPR 조건이 만족하는 최대 레벨까지 낮췄을 때, '제대로 분류된 갯수 - 잘못 분류된 갯수')와 인덱스 반환
# attack_label: 1은 이상, 0은 정상을 나타내는 리스트
# sorted_id_list: 점수를 기준으로 정렬된 index
# rest_miss_cnt: FPR 만족되는 경우까지 추가로 발생 가능한 false alarm의 수
def find_max_score_count(attack_label, sorted_id_list, rest_miss_cnt):
    score = 0
    miss_cnt = 0

    max_score = -len(sorted_id_list)    
    max_idx = 0 # index of sorted_id_list
    hit_cnt = 0
    hit_cnt_max = 0
    miss_cnt_max = 0

    assert (rest_miss_cnt >= 0), "find_max_score(): rest_miss_count must be larger than 0!"

    for i, eid in enumerate(sorted_id_list): # eid indicates entry id
        if attack_label[eid] == 0:
            miss_cnt += 1
            score -= 1
        elif attack_label[eid] == 1:
            score += 1
            hit_cnt += 1
        # skip the case (attack_label[idx] == 2) which means included by another group

        if rest_miss_cnt <= miss_cnt:
            return max_idx, max_score, hit_cnt_max, miss_cnt
        
        if max_score < score:
            max_score = score
            max_idx = i
            hit_cnt_max = hit_cnt
            miss_cnt_max = miss_cnt

@ray.remote
def find_max_score_ratio(attack_label, prediction, thd_candidate, detected_id_list, rest_miss_cnt):
    temp_score = 0
    hit_cnt = 0
    miss_cnt = 1

    result = {
        'score': 0,
        'thd_id': 0,
        'max_hit': 0,
        'max_miss': 0
    }

    assert (rest_miss_cnt >= 0), "find_max_score(): rest_miss_count must be larger than 0!"

    for i, thd in enumerate(thd_candidate):
        timeindex = np.where(prediction == thd)[0][0]
        if detected_id_list[timeindex] == False:
            if attack_label[timeindex] == 0:
                miss_cnt += 1
            elif attack_label[timeindex] == 1:
                hit_cnt += 1
        else:
            result['thd_id'] = i
        
        temp_score = hit_cnt/miss_cnt

        if rest_miss_cnt <= miss_cnt-1:            
            result['max_miss'] -= 1
            return result
        
        if result['score'] < temp_score: 
            result['score'] = temp_score
            result['thd_id'] = i
            result['max_hit'] = hit_cnt
            result['max_miss'] = miss_cnt            


def find_threshold_by_greedy_count(attack_label, predictions, fpr_threshold):
    N_TEST = len(predictions[0])
    N_GROUP = len(predictions)

    print("There are {} groups in the given results\n".format(N_GROUP))    

    # Sorting each list of scores
    SORTED_ID_LISTS = []
    for i in range(N_GROUP):
        SORTED_ID_LISTS.append(
            sorted(np.arange(N_TEST), reverse=True, key=predictions[i].__getitem__)
        )
    
    THD_IDS = [0 for i in range(N_GROUP)] # Initialize the thresholds for every group

    MISS_CNT = 0 # False alarm의 수
    MISS_THD = int((len(attack_label) - sum(attack_label)) * fpr_threshold) # FPR 기준으로 봤을 때 false alarm의 최대 수
    print("The maximum number of false alarm instances: ", MISS_THD)
    
    attack_label_dup = deepcopy(attack_label)
    while True:
        max_ids = []
        max_scores = []
        hit_counts = []
        miss_counts = []
        selected_gids = []
        
        for GID in range(N_GROUP):
            temp_idx, temp_score, temp_hit_cnt, temp_miss_cnt = find_max_score_count(attack_label_dup, SORTED_ID_LISTS[GID][THD_IDS[GID]:], MISS_THD - MISS_CNT)
            max_ids.append(temp_idx)
            max_scores.append(temp_score)
            hit_counts.append(temp_hit_cnt)
            miss_counts.append(temp_miss_cnt)
                    
        # print("Max_Scores in this time: ", max(max_scores))
        # print("Is tie scores: ", len(np.where(np.array(max_scores)==max(max_scores))[0]), max_scores)
        
        max_scores = np.array(max_scores)        
        selected_gids = np.where(np.array(max_scores) == max(max_scores))[0]
        if len(selected_gids) != 1: # 1등이 하나가 아닐 때
            hit_counts = np.array(hit_counts)
            selected_gids = np.where(np.array(hit_counts[selected_gids]) == max(hit_counts[selected_gids]))[0] # hitcount가 가장 높은 것을 선택
            if len(selected_gids) != 1: #1등이 하나가 아닐 때
                miss_counts = np.array(miss_counts)
                selected_gids = np.where(np.array(miss_counts[selected_gids]) == min(miss_counts[selected_gids]))[0] #miss count가 가장 낮은 것을 선택
        
        temp_miss_cnt = 0
        for gid in selected_gids:
            for i in range(THD_IDS[gid], THD_IDS[gid] + max_ids[gid] + 1):
                entry_id = SORTED_ID_LISTS[gid][i]
                if attack_label_dup[entry_id] == 0:
                    attack_label_dup[entry_id] = 2
                    temp_miss_cnt += 1

        if MISS_THD < MISS_CNT + temp_miss_cnt:
            break

        for gid in selected_gids:
            THD_IDS[gid] += max_ids[gid] + 1
            MISS_CNT += temp_miss_cnt
        
        if MISS_THD <= MISS_CNT:
            print("miss count:", MISS_CNT)
            break
    
    thresholds = []
    for i, s in enumerate(predictions):                
        min_id = len(s) - THD_IDS[i] - 1
        thresholds.append(sorted(s)[min_id])
        
    return thresholds

# 앙상블 모델에서 임계값 결정
def evaluate_ensemble3(attack_label, predictions, fpr_threshold, is_brief=False):
    thresholds = find_threshold_by_greedy_count(attack_label, predictions, fpr_threshold)
    voting_result = voting(predictions, thresholds)
    if is_brief == False:
        check_graph(voting_result, attack_label, 3)
    print_result(evaluate_w_threshold(attack_label, voting_result, 0.5), is_brief)
    

def find_threshold_by_greedy_ratio(attack_label, predictions, thresholds, fpr_threshold):
    N_TEST = len(predictions[0])
    N_GROUP = len(predictions)
    thresholds = deepcopy(thresholds)

    print("There are {} groups in the given results\n".format(N_GROUP))    

    # Sorting each list of scores
    thd_candidates = []
    for i in range(N_GROUP):
        thd_candidates.append(sorted(predictions[i], reverse=True))
    # thd_candidates = np.array(temp_thd_candidates)
    
    for i in range(N_GROUP):        
        thd_candidates[i] = np.array(thd_candidates[i])[thd_candidates[i] < thresholds[i]] #초기값보다 높은 임계값 후보 삭제
    # thd_candidates  = np.array(thd_candidates)
    # print(thd_candidates[0][10:], len(thd_candidates[0]))

    detected_id_list = np.array([ False for i in range(len(attack_label)) ])
    for i in range(N_GROUP):
        detected_id_list[predictions[i] >= thresholds[i]] = True

    MISS_CNT = sum(detected_id_list & (attack_label == 0)) # False alarm의 수
    # print("aa", MISS_CNT)
    MISS_THD = int((len(attack_label) - sum(attack_label)) * fpr_threshold) # FPR 기준으로 봤을 때 false alarm의 최대 수
    # print("The maximum number of false alarm instances: ", MISS_THD)
    
    pbar = tqdm(total=MISS_THD)
    pbar.update(MISS_CNT) 
    prev_miss = MISS_CNT

    ray.init(num_cpus = cpu_count()-1)    
    
    if MISS_THD - MISS_CNT <= 0:
        is_loop = False
    else:
        is_loop = True
    while is_loop:
        max_ids = []
        max_scores = []
        hit_counts = []
        miss_counts = []
        selected_gids = []
        
        # print(MISS_THD - MISS_CNT)
        results_remote = [ find_max_score_ratio.remote(attack_label, predictions[GID], thd_candidates[GID], detected_id_list, MISS_THD - MISS_CNT) for GID in range(N_GROUP) ]
        results = ray.get(results_remote)

        for GID in range(N_GROUP):            
            max_ids.append(results[GID]['thd_id'])
            max_scores.append(results[GID]['score'])
            hit_counts.append(results[GID]['max_hit'])
            miss_counts.append(results[GID]['max_miss'])

                    
        # print("Max_Scores in this time: ", max(max_scores))
        # print("Tie scores: ", len(np.where(np.array(max_scores)==max(max_scores))[0]), max_scores)
        
        #임계값을 조정할 group id (복수 가능) 찾아서 selected_gid에 저장
        max_scores = np.array(max_scores)
        selected_gids = np.where(np.array(max_scores) == max(max_scores))[0]
        # if len(selected_gids) != 1: # 1등이 하나가 아닐 때
        #     hit_counts = np.array(hit_counts)
        #     selected_gids = np.where(np.array(hit_counts[selected_gids]) == max(hit_counts[selected_gids]))[0] # hitcount가 가장 높은 것을 선택
        #     if len(selected_gids) != 1: #1등이 하나가 아닐 때
        #         miss_counts = np.array(miss_counts)
        #         selected_gids = np.where(np.array(miss_counts[selected_gids]) == min(miss_counts[selected_gids]))[0] #miss count가 가장 낮은 것을 선택


        # selected_gids를 miss_count가 적은 순으로 정렬
        # selected_miss_counts = miss_counts[selected_gids]
        miss_counts = np.array(miss_counts)        
        sorted_ids = sorted(np.arange(len(selected_gids)), reverse=False, key=miss_counts[selected_gids].__getitem__)
        # print(selected_gids, sorted_ids, miss_counts[selected_gids])

        for i in sorted_ids:
            gid = selected_gids[i]
            # print(gid, i, miss_counts[selected_gids][i])
            detected_id_list[predictions[gid] >= thd_candidates[gid][max_ids[gid]]] = True
            MISS_CNT = sum(detected_id_list & (attack_label == 0))
            pbar.update(MISS_CNT - prev_miss) 
            if MISS_THD <= MISS_CNT:   
                is_loop = False         
                break
            else:
                # print('aa', thresholds[gid], thd_candidates[gid][max_ids[gid]])                
                thresholds[gid] = thd_candidates[gid][max_ids[gid]]
                thd_candidates[gid] = thd_candidates[gid][max_ids[gid]+1:]

            prev_miss = MISS_CNT
        
    ray.shutdown()
    return thresholds


def voting(predictions, thresholds):
    new_scores = []
    for i, s in enumerate(predictions):                        
        new_scores.append(put_labels(s, thresholds[i]))
    new_scores = np.sum(new_scores, axis=0)
    new_scores[new_scores >= 1] = 1

    return new_scores
     

# def show_result(attack_label, predictions, thresholds, is_brief):
#     new_scores = []
#     for i, s in enumerate(predictions):                        
#         new_scores.append(put_labels(s, thresholds[i]))
        
#     summarize_score = np.sum(new_scores, axis=0)
#     fa_count = 0
#     for i in range(len(summarize_score)):
#         if summarize_score[i] != 0 and attack_label[i] == 0:
#             fa_count += 1
#     print("The number of false alarm instances: ", fa_count)

#     if is_brief:
#         evaluate_w_threshold_simple(attack_label, summarize_score, 0.5)
#     else:
#         evaluate_w_threshold(attack_label, summarize_score, 0.5)


def false_positive_list(attack_label, prediction, threshold):
   detection = prediction >= threshold
   return (attack_label == 0) & detection

def false_positive_count(attack_label, prediction, detected_id_list, threshold):
   detection = prediction >= threshold
   return sum((attack_label == 0) & detection & (detected_id_list == 0))


@ray.remote
def count_least_false_positive_count(attack_range, predictions, detected_id_list, attack_label):
    #각 공격에 대해, 가장 적은 FP 카운트로 탐지할 수 있는 그룹, 임계값, 얼마가 필요한지를 체크해 둠
    N_PRED = len(predictions)
    temp_thd = max(predictions[0][attack_range[0]:attack_range[1]+1])
    
    result = {
        "gid": 0,
        "thd": temp_thd,
        "fpc": false_positive_count(attack_label, predictions[0], detected_id_list, temp_thd)
    }

    for gid in range(1, N_PRED):
        temp_thd = max(predictions[gid][attack_range[0]:attack_range[1]+1])
        temp_fpc = false_positive_count(attack_label, predictions[gid], detected_id_list, temp_thd)
        if result["fpc"] > temp_fpc:
            result['gid'] = gid
            result['thd'] = temp_thd
            result['fpc'] = temp_fpc

    return result

def find_threshold_by_more_detection(attack_label, predictions, fpr_threshold):
    # FPR_CNT_THD = int(sum(attack_label == 0) * fpr_threshold)
    
    #각 결과(모델 or 그룹) 공격별 비용(오탐 수) -> 제일 낮은 것 부터 업데이트(prediction 수정)
    FPR_CNT_THD = int(sum(attack_label == 0) * fpr_threshold)
    pred = deepcopy(predictions)

    #attack label을 공격별로 구분하자
    print("Build attack ranges...")
    attack_ranges = []
    start_idx = 0
    for i in range(1, len(attack_label)):
        if attack_label[i-1] == 1 and attack_label[i] == 0: #end of attack
            attack_ranges.append((start_idx, i-1))
        elif attack_label[i-1] == 0 and attack_label[i] == 1: #start of attack
            start_idx = i

    N_ATT = len(attack_ranges)
    N_PRED = len(pred)

    thresholds = np.zeros(N_PRED, dtype=float)
    for i, a_pred in enumerate(pred):
        thresholds[i] = max(a_pred) + 1e-5

    detected_id_list = np.array([ False for i in range(len(attack_label)) ])

    print("Finding the attack range with the minimum false positive count...")
    # min_fp, gid, its_score

    pbar = tqdm(total=FPR_CNT_THD)
    prev_fpc = 0
    
    ray.init(num_cpus = cpu_count()-1)

    
    is_loop = True
    while is_loop and N_ATT != 0:

        fpcs = np.zeros(N_ATT, dtype=int)
        gids = np.zeros(N_ATT, dtype=int)
        thds = np.zeros(N_ATT, dtype=float)

        results_remote = [ count_least_false_positive_count.remote(attack_ranges[i], predictions, detected_id_list, attack_label) for i in range(N_ATT) ]
        results = ray.get(results_remote)        
        for i, a_result in enumerate(results):
            fpcs[i] = a_result['fpc']
            gids[i] = a_result['gid']
            thds[i] = a_result['thd']

        ids = np.where(fpcs == min(fpcs))[0]
        # print(ids, fpcs[ids])
        for a_id in ids:
            gid = gids[a_id]
            detected_id_list[pred[gid] >= thds[a_id]] = True
            cur_fpc = sum(detected_id_list & (attack_label == 0))
            if  cur_fpc > FPR_CNT_THD:
                is_loop = False
                break
            thresholds[gid] = thds[a_id]
        
        for i in range(len(ids)-1, -1, -1):
            attack_ranges.pop(ids[i])
        N_ATT = len(attack_ranges)
                # detected_id_list[fps[a_id]] = True
        
        pbar.update(cur_fpc - prev_fpc)            
        prev_fpc = cur_fpc
        # print(sum(detected_id_list & (attack_label == 0)), FPR_CNT_THD)
        
    # print(thresholds)        

    ray.shutdown()
    return thresholds

def evaluate_ensemble4(attack_label, predictions, fpr_threshold, is_brief =False):
    threshold = find_threshold_by_more_detection(attack_label, predictions, fpr_threshold)
    threshold = find_threshold_by_greedy_ratio(attack_label, predictions, threshold, fpr_threshold)    
    voting_result = voting(predictions, threshold)
    if is_brief==False:
        check_graph(voting_result, attack_label, 3)
    print_result(voting_result, is_brief)

def check_2darray(array1, array2):
    if len(array1) != len(array2):
        print('array size is not same')
    for i in range(len(array1)):
        if len(array1[i]) != len(array2[i]):
            print('array #{} size is not same'.format(i))
        for j in range(len(array1[i])):
            if array1[i][j] != array2[i][j]:
                print('not same value', i, j)

def check_1darray(array1, array2):
    if len(array1) != len(array2):
        print('array size is not same')
    for i in range(len(array1)):
        if array1[i] != array2[i]:
            print('not same value', i, j)


def evaluate_ensemble_best(attack_label, predictions, fpr_threshold, is_brief=False):
    evaluate_results = []
    voting_results = []
    
    org_predictions = deepcopy(predictions)
    org_attack_label = deepcopy(attack_label)

    threshold = find_threshold_by_greedy_count(attack_label, predictions, fpr_threshold)
    voting_results.append(voting(predictions, threshold))
    evaluate_results.append(evaluate_w_threshold(attack_label, voting_results[-1], 0.5))

    check_2darray(org_predictions, predictions)
    check_1darray(org_attack_label, attack_label)
    
    threshold = []
    for i in range(len(predictions)):        
        threshold.append(max(predictions[i]) + 1e-5)
    threshold = find_threshold_by_greedy_ratio(attack_label, predictions, threshold, fpr_threshold)
    voting_results.append(voting(predictions, threshold))
    print(sum(voting_results[-1]))
    evaluate_results.append(evaluate_w_threshold(attack_label, voting_results[-1], 0.5))
    
    check_2darray(org_predictions, predictions)
    check_1darray(org_attack_label, attack_label)

    threshold = find_threshold_by_more_detection(attack_label, predictions, fpr_threshold)
    threshold = find_threshold_by_greedy_ratio(attack_label, predictions, threshold, fpr_threshold)
    voting_results.append(voting(predictions, threshold))
    evaluate_results.append(evaluate_w_threshold(attack_label, voting_results[-1], 0.5))

    best_idx = 0
    print(evaluate_results[0]['eTaf1'])
    for i in range(1, len(evaluate_results)):
        print(evaluate_results[i]['eTaf1'])
        if evaluate_results[best_idx]['eTaf1'] < evaluate_results[i]['eTaf1']:
            best_idx = i
    print()

    result = evaluate_results[best_idx]
    if is_brief:
        print_result(result, is_brief)        
    else:
        check_graph(voting_results[best_idx], attack_label, 3)
        print_result(result, is_brief)


