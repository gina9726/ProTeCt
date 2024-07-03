### metrics.py
# Evaluation metrics.
###

import numpy as np
import torch
import pdb


class hierConsistency():
    def __init__(self, indices, n_samples=None):
        self.indices = indices
        self.n_samples = n_samples if n_samples is not None else len(indices)
        self.reset()

    def reset(self):
        self.is_incorrect = torch.zeros((self.n_samples)).bool().cuda()
        self.avg = 0

    def update(self, index, iscorrect):
        self.is_incorrect[index] |= (~iscorrect)
        self.avg = float((~self.is_incorrect[self.indices]).float().mean()) * 100.0

class Instance(object):
    def __init__(self, index):
        self.index = index
        self.labels = {}
        self.predictions = {}
        self.confidence = {}
        self.iscorrect = {}

    def update(self, node, pred, label, conf=None):
        self.predictions[node] = pred
        self.labels[node] = label
        if conf is not None:
            self.confidence[node] = conf
        self.iscorrect[node] = (pred == label)

    def isconsistent(self):
        return sum(self.iscorrect.values()) == len(self.iscorrect)

class HierClsMetrics(object):
    def __init__(self, index_list, n_intnl_nodes):
        self.instances = {i: Instance(i) for i in index_list}
        self.n_samples = len(index_list)
        self.n_intnl_nodes = n_intnl_nodes
        self.leaf_acc = None
        self.node_acc = None
        self.consistent_score = None

    def get_leaf_acc(self):
        if self.leaf_acc is None:
            self.leaf_acc = np.mean([x.iscorrect[-1] for x in self.instances.values()])
        return self.leaf_acc

    def get_node_acc(self):
        if self.node_acc is None:
            self.node_acc = {}
            for i in range(self.n_intnl_nodes):
                res = [x.iscorrect.get(i, -1) for x in self.instances.values()]
                self.node_acc[i] = np.mean([x for x in res if x >= 0])
        return self.node_acc

    def get_consistent_score(self):
        if self.consistent_score is None:
            self.consistent_score = np.mean([x.isconsistent() for x in self.instances.values()])
        return self.consistent_score


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
def multilabel_accuracy(all_pred, all_label):
    all_pred = np.asarray(all_pred)
    all_label = np.asarray(all_label)
    # for each sample, compare all_pred and all_label for each class label
    # 1.0 is the same and 0.0 is different
    # multiply all values for each row 
    correct_per_instance = np.prod(np.equal(all_pred, all_label)*1, axis=1)
    acc = np.mean(correct_per_instance)
    return acc


def percls_accuracy(all_pred, all_label, num_class=0):
    """Computes per class accuracy"""
    num_class = len(set(all_label)) if num_class == 0 else num_class
    all_pred = np.asarray(all_pred)
    all_label = np.asarray(all_label)

    cls_acc = -np.ones([num_class])
    for i in range(num_class):
        idx = (all_label == i)
        if idx.sum() > 0:
            cls_acc[i] = (all_pred[idx] == all_label[idx]).mean() * 100.0

    return cls_acc


def bin_accuracy(output, target):
    """Computes the binary classification accuracy"""
    pred = (output > 0.5).long()
    acc = (pred == target).float().mean() * 100.0
    return acc


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

