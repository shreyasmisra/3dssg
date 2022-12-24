import torch
import numpy as np
import sys
import os

def get_eval(data_dict):
    """ Evaluation of the model
    Parameters
    ----------
    data_dict: dict includes objects/relationship predict and triplets

    Returns data_dict: dict includes eval metrics
    -------
    """
    batch_size = len(data_dict["objects_predict"])
    top5_ratio_o = []
    top10_ratio_o = []
    top3_ratio_r = []
    top5_ratio_r = []
    top50_predicate = []
    top100_predicate = []
    pred_relations_ret = []
    
    NUM_CLUSTERS = 60
    for i in range(batch_size):
        object_pred = data_dict["objects_predict"][i].cpu().numpy()
        object_cat = data_dict["objects_cat"][i].cpu().numpy().copy()

        # here, need to notice the relationships between keys 'edges', 'pairs' and 'triples'
        # 'pairs' is the none duplicate list of the first two column of 'triples', while 'edges' corresponds to the index
        # the 'predicate_predicate' takes the mapping relation with 'pairs'
        # predicate_count = np.count_nonzero((data_dict['triples'][i] != 0).sum(1))
        predicate_pred = data_dict["predicate_predict"][i].cpu().numpy() #[:predicate_count]
        edges = np.concatenate((np.ones((NUM_CLUSTERS, 1))*60, np.arange(0, 60).reshape(-1,1)), axis=1).astype(int)

        triples = data_dict["triples"][i].cpu().numpy().astype(np.int64)
        triples = triples[~np.all(triples == 0, axis=1)]
        cluster_triples = []
        for index, cat in enumerate(object_cat):
            rel_idx = np.argwhere(triples[:, 1] == cat)
            if len(rel_idx)==1:
                cluster_triples.append([60, index, triples[rel_idx, 2].item()])
            elif len(rel_idx)>=1:
                cluster_triples.append([60, index, triples[rel_idx[0], 2].item()])
            else:
                cluster_triples.append([60, index, 0])
        cluster_triples = np.asarray(cluster_triples)

        zero_rows = np.zeros(cluster_triples.shape).astype(np.uint8)
        mask = (cluster_triples == zero_rows)[:, :2]
        mask = ~ (mask[:, 0] & mask[:, 1])
        cluster_triples = cluster_triples[mask]

        pairs = cluster_triples[:, :2]
        predicate_cat = cluster_triples[:, 2]
        predicate_pred_expand = np.zeros([cluster_triples.shape[0], predicate_pred.shape[1]])


        # import pdb; pdb.set_trace()
        for index, tpl in enumerate(cluster_triples):
            tmp = np.repeat(tpl[:2].reshape(1, -1), len(pairs), axis=0)
            mask = pairs == tmp
            mask = mask[:, 0] & mask[:, 1]
            # try:
            predicate_pred_expand[index] = predicate_pred[mask]
            # except:
            #     predicate_pred_expand[mask] = predicate_pred[mask]
        top5_ratio_o.append(topk_ratio(object_pred[:-1, :], object_cat, 5))
        top10_ratio_o.append(topk_ratio(object_pred[:-1, :], object_cat, 10))
        top3_ratio_r.append(topk_ratio(predicate_pred_expand, predicate_cat, 3))
        top5_ratio_r.append(topk_ratio(predicate_pred_expand, predicate_cat, 5))

         # store the index
        object_logits = np.max(object_pred, axis=1)
        obj_scores_per_rel = object_logits[edges].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * predicate_pred
        score_inds = argsort_desc(overall_scores)[:100]

        pred_rels = np.column_stack((pairs[score_inds[:, 0]], score_inds[:, 1]))

        top50_predicate.append(topk_triplet(pred_rels, cluster_triples, 50))
        top100_predicate.append(topk_triplet(pred_rels, cluster_triples, 100))
        pred_relations_ret.append(pred_rels)

    data_dict["top5_ratio_o"] = np.mean(np.array(top5_ratio_o))
    data_dict["top10_ratio_o"] = np.mean(np.array(top10_ratio_o))
    data_dict["top3_ratio_r"] = np.mean(np.array(top3_ratio_r))
    data_dict["top5_ratio_r"] = np.mean(np.array(top5_ratio_r))
    data_dict["top50_predicate"] = np.mean(np.array(top50_predicate))
    data_dict["top100_predicate"] = np.mean(np.array(top100_predicate))

    return data_dict, pred_relations_ret

def topk_ratio(logits, category, k):
    """
    Parameters
    ----------
    logits: [N C] N objects/relationships with C categroy
    category: [N 1] N objects/relationships
    k:  top k

    Returns topk_ratio: recall of top k (R@k)
    -------
    """
    topk_pred = np.argsort(-logits, axis=1)[:, :k]  # descending order
    topk_ratio = 0
    for index, x in enumerate(topk_pred):
        if category[index] in x:
            topk_ratio += 1
    topk_ratio /= category.shape[0]
    return topk_ratio

def topk_triplet(pred_tri, gt_tri, k):
    """
    Parameters
    ----------
    pred_tri: multiplying predict scores results
    gt_tri: triplets exist in the scene
    k:  top k

    Returns ratio: recall of top k (R@k)
    -------
    """
    # assert len(tri)>=k
    ratio = 0
    gt = gt_tri.tolist()
    pred = pred_tri[:k]
    for item in pred:
        line = item.tolist()
        if line in gt:
            ratio += 1
    ratio /= len(gt_tri)
    return ratio

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))