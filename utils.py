import copy
import numpy as np
import os
import pandas as pd
import scipy.stats as st
import seaborn as sns
import shap
import sklearn.decomposition as sk_decomposition
import xgboost
import xgboost as xgb
from collections import Counter
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from mytookit import utils
from scipy.special import softmax
from scipy.stats import pearsonr
import config as cfg, beeswarm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, \
    roc_curve
from xgboost import plot_importance


def get_xgboost_sklearn_predict_auc(params, X_train, X_test, y_train, y_test, mode="fusion", args=None, y_pred=None,
                                    k=0):
    i = args.instance_index

    if y_pred is None:

        clf = xgboost.XGBClassifier(**params, verbosity=0)
        # print(params)
        clf.fit(X_train, y_train)  # eval_set=[(X_train, y_train)]
        # print(clf.evals_result())
        y_pred_test_logits = clf.predict_proba(X_test)
        if args.classification_method == 'four_classification':
            y_pred = y_pred_test_logits
        else:
            y_pred = y_pred_test_logits[:, 1]

        y_label = y_test.iloc[i]
        print('y_label', y_label)
        pred_result = clf.predict_proba(X_test.iloc[i][None, :])
        print(mode, pred_result)

        if args.need_explain is True:
            # show_explain_lime(X_train, X_test, clf, y_test, mode, i, args)
            show_explain_shap(X_train, X_test, clf, y_test, mode, i, args)

    else:
        y_pred = y_pred
        clf = None
        pred_result = y_pred[i]

    # res_train = get_metric(y_train, y_pred_train, type, subset='train', plot=args.plot_fig)
    res_test = get_metric(y_test.values, y_pred, mode, subset='test', plot=args.plot_fig, args=args)

    if clf is not None:
        plot_importance(clf, max_num_features=15)
        plt.title(mode)
        plt.show()

    if args.save_feature_importance:
        save_feature_importance(clf, X_train, args, k)

    return res_test, y_pred, pred_result


def show_explain_shap(X_train, X_test, clf, y_test, mode, i, args):
    y_label = y_test.iloc[i]
    labels = int(y_label)
    index_dir = os.path.join(args.case_dir, 'plot', f'shap_explain')
    os.makedirs(index_dir, exist_ok=True)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    os.makedirs(os.path.join(index_dir, 'save_shap_values'), exist_ok=True)
    for index, data in enumerate(shap_values):
        save_shap_df = pd.DataFrame(data, columns=X_train.columns)
        save_shap_df.to_csv(os.path.join(index_dir, 'save_shap_values', f'shap_values_{mode}_{index}.csv'), index=False)

    # 1. total summary_plot
    if args.total_summary_plot:
        from matplotlib import colors as plt_colors
        class_inds = [0, 1, 2, 3]
        plt = beeswarm.summary_legacy(shap_values, X_train, show=False, class_names=['IA', 'MIA', 'AIS', "Benign"],
                                      title=f'{mode}', class_inds=class_inds, color=plt_colors.ListedColormap(
                np.array(['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2'])), max_display=len(X_train.columns.values))
        plt.savefig(os.path.join(index_dir, 'total_summary', f'summary_plot_{mode}.png'), bbox_inches='tight', dpi=600)
        plt.show()

    # 2. single class summary plot
    if args.single_summary_plot:
        dic = {0: 'IA', 1: 'MIA', 2: 'AIS', 3: "Benign"}
        for which_class in range(4):
            plt = beeswarm.summary_legacy_origin(shap_values[which_class], X_train)
            plt.title(f'{dic[which_class]}', fontsize=18)
            plt.savefig(os.path.join(index_dir, 'single_summary', f'single_summary_plot_{mode}_{which_class}.png'),
                        bbox_inches='tight', dpi=300)
            plt.show()

    # 3. instance force plot
    if args.instance_force_plot:
        shap_values = explainer.shap_values(X_test)
        plot = shap.force_plot(explainer.expected_value[labels], shap_values[labels][i, :], X_test.iloc[i, :],
                               link='logit')  # link='logit'
        shap.save_html(os.path.join(index_dir, 'force_plot', f"force_plot_{mode}_label_{labels}_index_{i}.htm"),
                       plot)  #
        force_plot_df = pd.DataFrame(shap_values[labels][i, :][None, :], columns=X_train.columns)
        force_plot_df.to_csv(os.path.join(index_dir, 'force_plot', f'shap_values_force_plot.csv'), index=False)
        x_test = pd.DataFrame(X_test.iloc[i, :][None, :], columns=X_train.columns)
        x_test.to_csv(os.path.join(index_dir, 'force_plot', f'x_test.csv'), index=False)


def get_common_metrics(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    Youden_index = tpr - fpr  # Sensitivity + Specificity − 1 = tpr + 1-fpr − 1
    index = np.argmax(Youden_index)
    thresh = thresholds[index]

    y_pred = np.where(y_pred_prob >= thresh, 1, 0)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / float(tn + fp)

    return sensitivity, specificity, acc, precision


def get_auc_p_value(y_true, y_pred_prob, original_auc):
    # random,repeat 1000
    n_permutations = 1000
    random_aucs = []
    for _ in range(n_permutations):
        y_true_permuted = np.random.permutation(y_true)
        random_auc = roc_auc_score(y_true_permuted, y_pred_prob, multi_class='ovr', average='micro')
        random_aucs.append(random_auc)

    p_value = (np.sum(random_aucs >= original_auc) + 1) / (n_permutations + 1)
    print("AUC:", original_auc)
    print("p值:", p_value)

    return p_value


def get_metric(y_true, y_pred, mode, subset, plot=False, args=None, thresh='Youden_index'):
    # thresh = 'default'
    if args.only_calculate_auc is True:
        auc_ovr = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')

        res = dict(auc=round(auc_ovr, 3))

    elif args.classification_method == 'four_classification':
        # y = y.map({'IA': 0, 'MIA': 1, 'AIS': 2, "Benign": 3})
        y_pred_argmax = np.argmax(y_pred, axis=1)

        # plot_confusion_matrix(y_true, y_pred_argmax, type)

        F1_score_micro = f1_score(y_true, y_pred_argmax, average="micro")
        auc_ovo = roc_auc_score(y_true, y_pred, multi_class='ovo', average='macro')
        auc_ovr = roc_auc_score(y_true, y_pred, multi_class='ovr', average='micro')
        # F1_score_macro = f1_score(y_true, y_pred_argmax, average="macro")
        auc_p_value = get_auc_p_value(y_true, y_pred, auc_ovr)

        # Distinguish malignancy
        # 0,1,2 and 3
        bridge_index = 3
        y_true_AIS_malignant = np.where(y_true < bridge_index, 1, 0)
        # print("AIS_malignant", y_true_AIS_malignant)
        y_pred_AIS_malignant_prob = y_pred[:, :bridge_index].sum(axis=1)
        AIS_malignant_auc = roc_auc_score(y_true_AIS_malignant, y_pred_AIS_malignant_prob)

        if thresh == 'Youden_index':
            AIS_malignant_sensitivity, AIS_malignant_specificity, AIS_malignant_acc, AIS_malignant_precision = get_common_metrics(
                y_true=y_true_AIS_malignant, y_pred_prob=y_pred_AIS_malignant_prob)
        else:
            y_pred_AIS_malignant = np.where(y_pred_AIS_malignant_prob > 0.5, 1, 0)
            AIS_malignant_precision = precision_score(y_true_AIS_malignant, y_pred_AIS_malignant)  # 需要指定阳性
            AIS_malignant_sensitivity = recall_score(y_true_AIS_malignant, y_pred_AIS_malignant)
            AIS_malignant_acc = accuracy_score(y_true_AIS_malignant, y_pred_AIS_malignant)
            conf_matrix = confusion_matrix(y_true_AIS_malignant, y_pred_AIS_malignant)
            tn, fp, fn, tp = conf_matrix.ravel()
            AIS_malignant_specificity = tn / float(tn + fp)

        # 0,1 and 2,3
        bridge_index = 2
        # print(y_true)
        y_true_AIS_benign = np.where(y_true < bridge_index, 1, 0)
        # print("AIS_benign", y_true_AIS_benign)
        y_pred_AIS_benign_prob = y_pred[:, :bridge_index].sum(axis=1)
        AIS_benign_auc = roc_auc_score(y_true_AIS_benign, y_pred_AIS_benign_prob)

        if thresh == 'Youden_index':
            AIS_benign_sensitivity, AIS_benign_specificity, AIS_benign_acc, AIS_benign_precision = get_common_metrics(
                y_true=y_true_AIS_benign, y_pred_prob=y_pred_AIS_benign_prob)
        else:
            y_pred_AIS_benign = np.where(y_pred_AIS_benign_prob > 0.5, 1, 0)
            AIS_benign_precision = precision_score(y_true_AIS_benign, y_pred_AIS_benign)  # 需要指定阳性
            AIS_benign_sensitivity = recall_score(y_true_AIS_benign, y_pred_AIS_benign)
            AIS_benign_acc = accuracy_score(y_true_AIS_benign, y_pred_AIS_benign)
            conf_matrix = confusion_matrix(y_true_AIS_benign, y_pred_AIS_benign)
            tn, fp, fn, tp = conf_matrix.ravel()
            AIS_benign_specificity = tn / float(tn + fp)

        # Distinguish between IA and non-IA
        # 0 and 1,2 3
        bridge_index = 1
        y_true_IA = np.where(y_true < bridge_index, 1, 0)
        # print("IA", y_true_IA)
        y_pred_IA_prob = y_pred[:, :bridge_index]
        IA_auc = roc_auc_score(y_true_IA, y_pred_IA_prob)

        if thresh == 'Youden_index':
            IA_sensitivity, IA_specificity, IA_acc, IA_precision = get_common_metrics(y_true=y_true_IA,
                                                                                      y_pred_prob=y_pred_IA_prob)
        else:
            y_pred_IA = np.where(y_pred_IA_prob > 0.5, 1, 0)
            IA_precision = precision_score(y_true_IA, y_pred_IA)
            IA_sensitivity = recall_score(y_true_IA, y_pred_IA)
            IA_acc = accuracy_score(y_true_IA, y_pred_IA)
            conf_matrix = confusion_matrix(y_true_IA, y_pred_IA)
            tn, fp, fn, tp = conf_matrix.ravel()
            IA_specificity = tn / float(tn + fp)

        # Distinguish between IA and MIA
        # y = y.map({'IA': 0, 'MIA': 1, 'AIS': 2, "Benign": 3})
        index_0_1 = np.where(y_true <= 1)
        y_true_MIA_IA = np.where(y_true == 0, 1, 0)[index_0_1]
        # print("IA", y_true_IA)
        y_pred_MIA_IA_prob = softmax(y_pred[:, :2], axis=1)[:, 0][index_0_1]

        MIA_IA_auc = roc_auc_score(y_true_MIA_IA, y_pred_MIA_IA_prob)

        if thresh == 'Youden_index':
            MIA_IA_sensitivity, MIA_IA_specificity, MIA_IA_acc, MIA_IA_precision = get_common_metrics(
                y_true=y_true_MIA_IA, y_pred_prob=y_pred_MIA_IA_prob)

        else:
            y_pred_MIA_IA = np.argmax(y_pred[:, :2][index_0_1], axis=1)
            y_pred_MIA_IA = np.where(y_pred_MIA_IA == 0, 1, 0)
            MIA_IA_precision = precision_score(y_true_MIA_IA, y_pred_MIA_IA)
            MIA_IA_sensitivity = recall_score(y_true_MIA_IA, y_pred_MIA_IA)
            MIA_IA_acc = accuracy_score(y_true_MIA_IA, y_pred_MIA_IA)
            conf_matrix = confusion_matrix(y_true_MIA_IA, y_pred_MIA_IA)
            tn, fp, fn, tp = conf_matrix.ravel()
            MIA_IA_specificity = tn / float(tn + fp)

        # # Distinguish between AIS and MIA,IA
        # y = y.map({'IA': 0, 'MIA': 1, 'AIS': 2, "Benign": 3})
        index_0_1_2 = np.where(y_true <= 2)
        y_true_AIS_MIA_IA = np.where(y_true == 2, 0, 1)[index_0_1_2]
        # print("IA", y_true_IA)
        y_pred_AIS_MIA_IA_prob = np.sum(y_pred[:, :1], axis=1)[index_0_1_2]
        y_pred_AIS_MIA_IA_neg_prob = y_pred[:, 2][index_0_1_2]
        y_pred_AIS_MIA_IA_total_prob = np.concatenate(
            (y_pred_AIS_MIA_IA_neg_prob[:, None], y_pred_AIS_MIA_IA_prob[:, None]), axis=1)
        y_pred_AIS_MIA_IA_prob = softmax(y_pred_AIS_MIA_IA_total_prob[:, :2], axis=1)[:, 1]

        AIS_MIA_IA_auc = roc_auc_score(y_true_AIS_MIA_IA, y_pred_AIS_MIA_IA_prob)

        if thresh == 'Youden_index':
            AIS_MIA_IA_sensitivity, AIS_MIA_IA_specificity, AIS_MIA_IA_acc, AIS_MIA_IA_precision = get_common_metrics(
                y_true=y_true_AIS_MIA_IA, y_pred_prob=y_pred_AIS_MIA_IA_prob)
        else:
            y_pred_AIS_MIA_IA = np.argmax(y_pred_AIS_MIA_IA_total_prob, axis=1)
            AIS_MIA_IA_precision = precision_score(y_true_AIS_MIA_IA, y_pred_AIS_MIA_IA)
            AIS_MIA_IA_sensitivity = recall_score(y_true_AIS_MIA_IA, y_pred_AIS_MIA_IA)
            AIS_MIA_IA_acc = accuracy_score(y_true_AIS_MIA_IA, y_pred_AIS_MIA_IA)
            conf_matrix = confusion_matrix(y_true_AIS_MIA_IA, y_pred_AIS_MIA_IA)
            tn, fp, fn, tp = conf_matrix.ravel()
            AIS_MIA_IA_specificity = tn / float(tn + fp)

        y_pred = y_pred_argmax

        print('y_true', Counter(y_true))
        res = dict(auc_ovr=round(auc_ovr, 3), auc_p_value=auc_p_value,  # auc_ovo=round(auc_ovo, 3),
                   F1_score_micro=round(F1_score_micro, 3),

                   AIS_malignant_auc=round(AIS_malignant_auc, 3),
                   # AIS_malignant_precision=round(AIS_malignant_precision, 3),
                   AIS_malignant_sensitivity=round(AIS_malignant_sensitivity, 3),
                   AIS_malignant_specificity=round(AIS_malignant_specificity, 3),
                   # AIS_malignant_acc=round(AIS_malignant_acc, 3),

                   AIS_benign_auc=round(AIS_benign_auc, 3),  # AIS_benign_precision=round(AIS_benign_precision, 3),
                   AIS_benign_sensitivity=round(AIS_benign_sensitivity, 3),
                   AIS_benign_specificity=round(AIS_benign_specificity, 3),  # AIS_benign_acc=round(AIS_benign_acc, 3),

                   IA_auc=round(IA_auc, 3),  # IA_precision=round(IA_precision, 3),
                   IA_sensitivity=round(IA_sensitivity, 3), IA_specificity=round(IA_specificity, 3),
                   # IA_acc=round(IA_acc, 3),

                   MIA_IA_auc=round(MIA_IA_auc, 3),  # MIA_IA_precision=round(MIA_IA_precision, 3),
                   MIA_IA_sensitivity=round(MIA_IA_sensitivity, 3), MIA_IA_specificity=round(MIA_IA_specificity, 3),
                   # MIA_IA_acc=round(MIA_IA_acc, 3),

                   AIS_MIA_IA_auc=round(AIS_MIA_IA_auc, 3),  # AIS_MIA_IA_precision=round(AIS_MIA_IA_precision, 3),
                   AIS_MIA_IA_sensitivity=round(AIS_MIA_IA_sensitivity, 3),
                   AIS_MIA_IA_specificity=round(AIS_MIA_IA_specificity, 3),  # AIS_MIA_IA_acc=round(AIS_MIA_IA_acc, 3),
                   y_pred=y_pred,
                   y_true=y_true)  # F1_score_macro=round(F1_score_macro, 3), )  # from pprint import pprint  # pprint(res)

    else:
        auc = roc_auc_score(y_true, y_pred)
        res = round(auc, 3)
    return res
