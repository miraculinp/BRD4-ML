import os
import pickle
import argparse
import pandas as pd
import shap
import matplotlib.pyplot as plt
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", default="./demo/demo.csv", help="Input table in .csv format. Please refer to README for more requirements.", type=str)
parser.add_argument("--output", "-o", default="./demo/", help="Output folder.", type=str)
parser.add_argument("--core", "-c", default="", help="Scarffold for decomposition, in SMILES format.", type=str)
args = parser.parse_args()
prefix = args.input.split('/')[-1].replace('.csv', '')

if not os.path.exists(args.output+"/{}_decomp.pk".format(prefix)):
    print("Please run decompose.py for side-chain decomposition first!")
    exit()

# Load table
print("MolSHAP analysis for {}".format(args.input))
table = pd.read_csv(args.input)
assert 'ID' in table.columns
assert table['ID'].unique().shape[0] == table.shape[0], 'Compound ID conflicts!'
assert 'SMILES' in table.columns
assert 'Activity' in table.columns

# Load decomposition results
[core, list_success, success_idx] = pickle.load(open(args.output+"/{}_decomp.pk".format(prefix), 'rb'))
list_smiles = table['SMILES'].values
list_affinity = table['Activity'].values
list_mol = [get_mol_from_smiles(smiles) for smiles in list_smiles]


list_smiles = [list_smiles[i] for i in success_idx]
list_affinity = [list_affinity[i] for i in success_idx]
list_mol = [list_mol[i] for i in success_idx]
print("Using {} compounds for analysis.".format(len(list_success)))

# Define fragments
list_res = [{r:Chem.MolToSmiles(item[r]) for r in item} for item in list_success]
frag_smi2mol = {"":None}
for res, decomp in zip(list_res, list_success):
    for item in res:
        if res[item] not in frag_smi2mol:
            frag_smi2mol[res[item]] = decomp[item]
list_keys = list(list_success[0].keys())
list_info = np.array([[item[key] for key in list_keys] for item in list_res])
num = len(list_res)
dict_core_res = defaultdict(lambda:set())
for i in list_res:
    for j in i:
        dict_core_res[j].add(i[j])
dict_candidate = {}
print('-'*60)
print("Side-chain candidates:")
for key in list_keys:
    dict_candidate[key] = np.unique([item[key] for item in list_res])
    print(key, len(dict_candidate[key]), sep=': ')


# Side-chain ranking task
threshold = 2
list_rule = []
dict_rule_pair = {}
dict_count_pair = {}
list_train = []
for i in range(num):
    infoi = list_info[i]
    for j in range(i+1, num):
        infoj = list_info[j]
        if sum(infoi != infoj) != 1:
            continue
        r = np.where(infoi != infoj)[0][0]
        if list_affinity[i] > list_affinity[j]:
            list_rule.append([r, i, j, infoi[r], infoj[r], list_affinity[i], list_affinity[j]])
        else:
            list_rule.append([r, j, i, infoj[r], infoi[r], list_affinity[j], list_affinity[i]])
        if np.abs(list_affinity[i] - list_affinity[j]) < np.log10(threshold):
            continue
        if r not in dict_rule_pair:
            dict_rule_pair[r] = defaultdict(lambda:0)
            dict_count_pair[r] = defaultdict(lambda:0)
        if list_affinity[i] > list_affinity[j]:
            dict_rule_pair[r][(infoi[r], infoj[r])] += 1
        else:
            dict_rule_pair[r][(infoj[r], infoi[r])] += 1
        dict_count_pair[r][(infoi[r], infoj[r])] += 1
        dict_count_pair[r][(infoj[r], infoi[r])] += 1
        list_train.extend([i, j])
list_rule = np.array(list_rule, dtype=object)
list_train = np.unique(list_train)
print('-'*60)
print("Side-chain ranking pairs:")
for r in dict_rule_pair:
    for item in dict_rule_pair[r]:
        if item[0] < item[1]:
            continue
        dict_rule_pair[r][item] = dict_rule_pair[r][item] / dict_count_pair[r][item] * 100
    print(list_keys[r], len(dict_rule_pair[r]), sep=': ')


# Background side chains
list_frag = np.unique(np.concatenate([[smi for smi in item.values()] for item in list_res]))
frag2idx = defaultdict(lambda:len(frag2idx))
frag2idx[""] # take zero indices
dict_empty = {}
for i in list_keys:
    list_frag_r = np.unique([item[i] for item in list_res])
    list_count_ring = np.array([frag.count(":[*:{}]".format(i[1:])) for frag in list_frag_r])
    list_count_nonring = np.array([frag.count("[*:{}]".format(i[1:])) for frag in list_frag_r])
    ring_counts = pd.value_counts(list_count_ring)
    nonring_counts = pd.value_counts(list_count_nonring)
    flag = False
    if i == 'Core':
        empty = list_frag_r[0]
    elif ring_counts.index[0] == 2:
        empty = "c(:[*:{}]):[*:{}]".format(i[1:], i[1:])
    elif ring_counts.index[0] == 0:
        empty = '.'.join(["[H][*:{}]".format(i[1:])] * nonring_counts.index[0])
    else:
        flag = True
    if flag or empty not in dict_candidate[i]:
        # print('Warning:', i, empty)
        length = np.array([len(item) for item in dict_candidate[i]])
        empty = np.random.choice(dict_candidate[i][length == length.min()])
    frag2idx[empty]
    dict_empty[i] = empty
    for frag in list_frag_r:
        frag2idx[frag]
# new fragment
frag2idx = dict(frag2idx)
num_frag = len(frag2idx)
list_frag = ["" for _ in range(num_frag)]
for frag in frag2idx:
    list_frag[frag2idx[frag]] = frag
list_frag = np.array(list_frag)
list_frag_mol = np.array([frag_smi2mol[frag] if frag in frag_smi2mol else Chem.MolFromSmiles(frag, sanitize=False) for frag in list_frag])
print('-'*60)
for i in list_keys:
    print(i, "background:", dict_empty[i])

# Side-chain representations
ori_idx = np.zeros((len(list_mol), len(list_keys)), dtype=int)
for i, res in enumerate(list_res):
    ori_idx[i] = [frag2idx[res[r]] for r in list_keys]

# Model training
print("-"*60)
print("Model training")
use_onehot = True
np.random.seed(1234)
kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
for train_idx, test_idx in kfold.split(list_smiles):
    break
molshap = MolSHAP()
molshap.init(list_keys, list_frag, list_frag_mol, frag2idx)
molshap.preload(ori_idx)
X_ori = np.array(get_fps(list_mol))
y_ori = np.array(list_affinity)
if use_onehot:
    molshap.train_onehot(ori_idx[train_idx], y_ori[train_idx])
else:
    molshap.train(X_ori[train_idx], y_ori[train_idx])

# Evaluate regression
if use_onehot:
    y_qsar = molshap.predict_onehot(ori_idx[test_idx])
else:
    y_qsar = molshap.predict(X_ori[test_idx])

dict_results = {}
dict_results['Regression'] = {
    'PCC': pearsonr(y_ori[test_idx], y_qsar)[0],
    'R2': r2_score(y_ori[test_idx], y_qsar),
}
print(dict_results)


# Side-chain sampling
print("-"*60)
print("Model analysing")    
def give_me_some(total_num=1000):
    candidate_idx = []
    for i in list_keys:
        list_temp = []
        parent = [frag2idx[frag] for frag in dict_candidate[i]]
        while len(list_temp) < total_num:
            child = parent.copy()
            np.random.shuffle(child)
            list_temp.extend(child)
        candidate_idx.append(list_temp[:total_num])
    candidate_idx = np.array(candidate_idx).T
    return candidate_idx

new_idx = set()
total_num = 1000
while len(new_idx) < total_num:
    print(len(new_idx), '\r', end='')
    idxs = give_me_some()
    for idx in idxs:
        new_idx.add(tuple(idx))
new_idx = np.array([list(item) for item in new_idx])
new_idx = new_idx[:total_num]
y_new_qsar, y_new_qsar_std = molshap.predict_onehot(new_idx, True)
# Shapley values
if os.path.exists(args.output+"/{}_contrib.pk".format(prefix)):
    dict_shap_rule = pickle.load(open(args.output+"/{}_contrib.pk".format(prefix), 'rb'))
else:
    explainer = shap.KernelExplainer(molshap.predict_onehot, 
                                    np.array([[frag2idx[dict_empty[i]] for i in list_keys]]))
    new_shap_values = explainer.shap_values(new_idx, nsamples=100)
    dict_shap_rule = defaultdict(lambda:[])
    for i in range(len(new_idx)):
        for j, r in enumerate(list_keys):
            dict_shap_rule[new_idx[i,j]].append(new_shap_values[i][j]) 
    dict_shap_rule = dict(dict_shap_rule)
    for i in dict_shap_rule:
        score = np.array(dict_shap_rule[i])
        dict_shap_rule[i] = np.nanmean(score)
    pickle.dump(frag2idx, open(args.output+"/{}_frag2idx.pk".format(prefix), 'wb'))
    pickle.dump(dict_shap_rule, open(args.output+"/{}_contrib.pk".format(prefix), 'wb'))
    dict_shap_rule_by_smiles = {}
    for i in dict_shap_rule:
        dict_shap_rule_by_smiles[list_frag[i]] = dict_shap_rule[i]
    pickle.dump(dict_shap_rule_by_smiles, open(args.output+"/{}_contrib_by_smiles.pk".format(prefix), 'wb'))


list_success_label = []
list_success_shap = []
for rule in list_rule:
    if rule[1] not in train_idx:
        continue
    if rule[2] not in train_idx:
        continue
    label_diff = int(np.abs(rule[5] - rule[6]) > np.log10(threshold))
    shap_diff = dict_shap_rule[frag2idx[rule[3]]] - dict_shap_rule[frag2idx[rule[4]]]

    list_success_label.append(label_diff)
    list_success_shap.append(shap_diff)
dict_results["Seen"] = score_cls(list_success_label, list_success_shap)
print("Seen:", dict_results["Seen"])

list_success_label = []
list_success_shap = []
for rule in list_rule:
    if (rule[1] in train_idx) and (rule[2] in train_idx):
        continue
    label_diff = int(np.abs(rule[5] - rule[6]) > np.log10(threshold))
    shap_diff = dict_shap_rule[frag2idx[rule[3]]] - dict_shap_rule[frag2idx[rule[4]]]

    list_success_label.append(label_diff)
    list_success_shap.append(shap_diff)
dict_results["Unseen"] = score_cls(list_success_label, list_success_shap)
print("Unseen:", dict_results["Unseen"])


