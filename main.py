import numpy as np
import os
import pandas as pd
import random
import sklearn.metrics as skl_metrics
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from mytookit import utils
import config as cfg
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils import get_xgboost_sklearn_predict_auc, select_dict_num, plot_fig_setting, select_features, IC_95

random.seed(42)
np.random.seed(42)


def _parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--case_name", default='case', type=str, help="experiment name")
    arg_parser.add_argument("--model_name", default='xgboost', type=str, help="xgboost or svm")
    arg_parser.add_argument("--boostrap_num", default=1, type=int, help="boostrap_num")
    arg_parser.add_argument("--plot_fig", default=False, type=bool, help="whether plot_fig")
    arg_parser.add_argument("--test_features", default="['CT', 'IVD', 'CT+IVD']", type=str, help="which feature to "
                                                                                                 "calculate")
    arg_parser.add_argument("--cross_validation", default=1, type=int, help="whether use cross validation")
    arg_parser.add_argument("--need_explain", default=False, type=bool, help="whether need explain")
    arg_parser.add_argument("--total_summary_plot", default=False, type=bool, help="whether need total summary_plot")
    arg_parser.add_argument("--single_summary_plot", default=False, type=bool, help="whether need single summary plot")
    arg_parser.add_argument("--instance_force_plot", default=False, type=bool, help="whether need instance force plot")
    arg_parser.add_argument("--instance_index", default=0, type=int, help="instance index")
    arg_parser.add_argument("--classification_method", default='', type=str, help="classification method，AIS_benign， "
                                                                                  "AIS_malignant, "
                                                                                  "four_classification")
    arg_parser.add_argument("--n_splits", default=10, type=int, help="n split")
    arg_parser.add_argument("--start_random_state", default=42, type=int, help="start_random_state")

    args = arg_parser.parse_args()
    # read test features
    args.test_features = eval(args.test_features)

    # read environment variables
    args.case_dir = os.path.join(cfg.log_dir, args.case_name)
    os.makedirs(args.case_dir, exist_ok=True)

    args.df_path = cfg.CSV_PATH
    for k, v in vars(args).items():
        print(k, '=', v)

    return args


def get_params(mode):
    if '+' not in mode:
        if mode == 'evlRNA':
            params = cfg.GENE_XGBOOST_LOGISTIC_PARAMS
        else:
            params = cfg.IMAGES_XGBOOST_LOGISTIC_PARAMS
    else:
        params = cfg.FUSION_XGBOOST_LOGISTIC_PARAMS

    return params


def get_gene_image_fusion_res(args):
    df = utils.read_csv(args.df_path)
    print(args.df_path, "shape", df.shape)
    # print(df.columns.values.tolist())

    # creating X and y
    print(df.pathology.value_counts())
    y = df['pathology']
    # print(y.map(lambda x: 0 if x in ["Benign", 'AIS', 'AAH'] else 1).value_counts())
    X = df.drop(["study_id", 'pathology'], axis=1)

    if args.use_argmax:
        cfg.image_features = cfg.argmax_features

    res_dict = {f: [] for f in args.test_features}
    pred_gt_dict = {f: [] for f in args.test_features}

    if args.classification_method == 'four_classification':
        y = y.map({'ADC': 0, 'MIA': 1, 'AIS': 2, "Benign": 3})
        objective_update = {'objective': 'multi:softprob'}  # 'num_class':4
        cfg.GENE_XGBOOST_LOGISTIC_PARAMS.update(objective_update)
        cfg.IMAGES_XGBOOST_LOGISTIC_PARAMS.update(objective_update)
        cfg.FUSION_XGBOOST_LOGISTIC_PARAMS.update(objective_update)
    elif args.classification_method == "AIS_benign":
        y = y.map(lambda x: 0 if x in ["Benign", 'AIS', 'AAH'] else 1)
    elif args.classification_method == "AIS_malignant":
        y = y.map(lambda x: 0 if x == "Benign" else 1)
    elif args.classification_method == "IA":
        y = y.map(lambda x: 1 if x == "IA" else 0)
    elif args.classification_method == "Benign":
        y = y.map(lambda x: 1 if x == "Benign" else 0)

    annatated_ct_features = cfg.ANNAOTATED_CT_FEATURES
    solid_ct_features = cfg.CTR_FEATURES

    gene_features_list = cfg.WITH_WEIGHT_GENE_FEATURES[:args.selected_IVD_nums]
    ai_features_list = cfg.CT_PATHOLOGY + cfg.SOLID_SELECTED_FEATURES
    observer1_features_list = cfg.OBSERVER1_PATHOLOGY
    observer2_features_list = cfg.OBSERVER2_PATHOLOGY
    X_copy = X

    pred_prob_total = {}
    for i in range(args.boostrap_num):
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.start_random_state + i)
        print(skf)
        for k, (train_index, test_index) in enumerate(skf.split(X, y)):
            if args.cross_validation == 0 and k != 1:
                continue

            # get feature list
            for modes in args.test_features:
                print("\nmodes", modes)
                params = get_params(modes)
                features_list = []
                all_modes = modes.split(' + ')
                for mode in all_modes:
                    if cfg.evlRNA == mode:
                        features_list += gene_features_list
                        print(cfg.evlRNA, gene_features_list)
                    elif cfg.Rad == mode:
                        features_list += ai_features_list
                        print(cfg.Rad, ai_features_list)
                    elif cfg.Senior == mode:
                        features_list += observer1_features_list
                        print(cfg.Senior, observer1_features_list)
                    elif cfg.Junior == mode:
                        features_list += observer2_features_list
                        print(cfg.Junior, observer2_features_list)
                    elif cfg.CLINICAL == mode:
                        features_list += annatated_ct_features
                        print(cfg.CLINICAL, annatated_ct_features)
                    elif cfg.vCTR == mode:
                        features_list += solid_ct_features
                        print(cfg.vCTR, solid_ct_features)
                    elif cfg.CTR == mode:
                        X_copy['CTR'] = X_copy.apply(
                            lambda x: x['neg300_solid_2d_diameter_clinical(mm)'] / x['nodule_2d_diameter_clinical(mm)'],
                            axis=1)
                        features_list += ['CTR']
                        print(cfg.CTR, solid_ct_features)

                features_list = sorted(list(set(features_list)))
                print(modes, features_list)

                # generate train, test
                X = X_copy[features_list]

                # rename features
                X.rename(columns={'observer1_invasiveness': 'Senior_invas', 'observer2_invasiveness': 'Junior_invas',
                                  'malignancy_pathology': 'malig_prob', 'invasiveness': 'rad_invas', 'IA': 'IA_prob',
                                  'attenuation': 'attenuation', 'neg300_volume_ratio': 'vCTR',
                                  'nodule_2d_diameter_clinical(mm)': 'diameter'}, inplace=True)

                X_train, X_test, y_train, y_test = X[X.index.isin(train_index)], X[X.index.isin(test_index)], y[
                    y.index.isin(train_index)], y[y.index.isin(test_index)]

                X_train_IVD_fs = X_train
                X_test_IVD_fs = X_test
                if modes in ['Senior', 'Junior']:
                    doctor_judge = X_test_IVD_fs.reset_index(drop=True).astype(int).values
                    num = doctor_judge.shape[0]
                    y_pred = np.zeros((num, 4))
                    for i in range(num):
                        y_pred[i, doctor_judge[i]] = 1
                else:
                    y_pred = None

                res_test, y_pred, pred_result = get_xgboost_sklearn_predict_auc(params, X_train_IVD_fs, X_test_IVD_fs,
                                                                                y_train, y_test, mode=modes, args=args,
                                                                                y_pred=y_pred, k=k)

                print(modes, "auc", res_test)
                res_dict[modes].append(res_test)
                y_pred_gt = np.concatenate((y_pred, y_test.values[:, None]), axis=1)
                y_pred_gt_df = pd.DataFrame(y_pred_gt, columns=['IA', 'MIA', 'AIS', 'Benign', 'gt'])
                y_pred_gt_df['k_fold'] = k
                pred_gt_dict[modes].append(y_pred_gt_df)
                pred_prob_total[modes] = pred_result

    mean_df_dict = {}
    predict_argmax_df = []
    for key, res in res_dict.items():
        # for one_fold_res in res:
        y_pred = res[0].pop('y_pred')
        y_true = res[0].pop('y_true')
        df_explain = pd.DataFrame(y_pred, columns=[key])
        predict_argmax_df.append(df_explain)

        res_df = pd.DataFrame(res)
        print(res_df)
        if args.add_95IC is True:
            mean_df_dict[key] = IC_95(res_df)
        else:
            mean_df_dict[key] = res_df.mean()
        df = pd.concat(pred_gt_dict[key])
        df.to_csv(os.path.join(args.case_dir, f"pred_gt_{key}.csv"))

    if args.need_explain is True:
        predict_explain_df = pd.concat(predict_argmax_df, axis=1)
        predict_explain_df.insert(0, 'gt', y_true)
        predict_explain_df.to_csv(os.path.join(args.case_dir, f"all_instance_expalin.csv"))

    total_mean_df = pd.DataFrame(mean_df_dict).T
    total_mean_df.to_csv(os.path.join(args.case_dir, f"mean_bs{args.boostrap_num}_rs{args.start_random_state}.csv"))
    print("total mean df\n", total_mean_df)


if __name__ == "__main__":
    args = _parse_args()
    get_gene_image_fusion_res(args)
