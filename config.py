import os

root_dir = ""
csv_path = ""
log_dir = ""

GENE_XGBOOST_LOGISTIC_PARAMS = {'n_estimators': 200, 'learning_rate': 0.1, 'booster': 'gbtree',
                                'objective': 'binary:logistic', 'gamma': 0.1, 'max_depth': 5, 'reg_alpha': 0,
                                'reg_lambda': 10, "scale_pos_weight": 1, 'subsample': 0.7, 'colsample_bytree': 0.7,
                                'min_child_weight': 1, 'random_state': 42, 'n_jobs': 4, "importance_type": "weight"}

IMAGES_XGBOOST_LOGISTIC_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'booster': 'gbtree',
                                  'objective': 'binary:logistic', 'gamma': 0.1, 'max_depth': 4, 'reg_alpha': 1,
                                  'reg_lambda': 2, "scale_pos_weight": 1, 'subsample': 0.6, 'colsample_bytree': 0.5,
                                  'min_child_weight': 1, 'random_state': 42, 'n_jobs': 4, }

FUSION_XGBOOST_LOGISTIC_PARAMS = {'n_estimators': 300, 'learning_rate': 0.1, 'booster': 'gbtree',
                                  'objective': 'binary:logistic', 'gamma': 0.1, 'max_depth': 4, 'reg_alpha': 0,
                                  'reg_lambda': 2, "scale_pos_weight": 1, 'subsample': 0.6, 'colsample_bytree': 0.5,
                                  'min_child_weight': 1, 'random_state': 42, 'n_jobs': 4, }

evlRNA = "evlRNA"
Rad = "Rad"
Senior = "Senior"
Junior = "Junior"
CLINICAL = 'size_density'
CTR = 'CTR'
vCTR = 'vCTR'

features = ['Rad_invasiveness', 'attenuation', 'IA_prob', 'malignancy_prob', 'CTR', 'diameter_mm']
categorical_features_names = ['Senior_invasiveness', 'Junior_invasiveness', 'Rad_invasiveness', 'attenuation']
categorical_features_dict = {'Senior_invasiveness': ['IA', 'MIA', 'AIS', 'Benign'],
                             'Junior_invasiveness': ['IA', 'MIA', 'AIS', 'Benign'],
                             'Rad_invasiveness': ['IA', 'MIA', 'AAH_AIS'],  # 0.0    61
                             'attenuation': ['SN', 'PSN', 'GGO'], }

WITH_WEIGHT_GENE_FEATURES = ['HLA-E', 'BIN2', 'Z97192.1', 'KAZN', 'CCDC9B', 'PLEKHO1', 'PTGS1', 'ANXA4', 'SNX29',
                             'CEP164', 'GFRA2', 'TBC1D24', 'NPC2', 'CCND1', 'KIAA1217', 'DMD', 'SEZ6L', 'NET1', 'TMTC1',
                             'FAM163A', 'GALNT12', 'VASH1', 'GNG11', 'SLC37A1', 'ITM2B', 'SEC62', 'RPH3A', 'EIF4E',
                             'GKAP1', 'MYL6', 'PDIA3', 'CD55', 'SP1', 'ARMCX4', 'NDRG4', 'LINC00632', 'TFDP2',
                             'ARHGAP30', 'PF4', 'ADRB1']

SELECTED_17_GENE_FEATURES = ['HLA-E', 'BIN2', 'Z97192.1', 'KAZN', 'CCDC9B', 'PLEKHO1', 'PTGS1', 'ANXA4', 'SNX29',
                             'CEP164', 'GFRA2', 'TBC1D24', 'NPC2', 'CCND1', 'KIAA1217', 'DMD', 'SEZ6L']

OBSERVER1_PATHOLOGY = [  # 'observer1_pathology'
    # 'observer1_malignancy_pathology',
    'observer1_invasiveness',  # 0,1,2,3 'IA':0.0, 'MIA':1.0, 'AIS':2.0, 'Benign':3.0
    # 'observer1_IA',
]
OBSERVER2_PATHOLOGY = [  # 'observer2_pathology',
    # 'observer2_malignancy_pathology',
    'observer2_invasiveness',  # 'observer2_IA',
]

CT_PATHOLOGY = ['malignancy_pathology', 'invasiveness', 'IA', 'attenuation']
ANNAOTATED_CT_FEATURES = [  # 'diameter',
    'diameter_mm', 'density',  # 0,1,2
]

PREDICTED_CT_FEATURES = [  # 'common_diameter',
    'attenuation',  # 'calcification',
    # 'lobulation',
    # 'burr',
    # 'malignancy_pathology',
    # 'EGFR_CT',
    # 'HALO_halo',
    # 'invasiveness',
    # 'IA',
    # 'CHAR_cavity',
    # 'vacuole',
    # 'CHAR_vessel',
    # 'CHAR_pleural_indent',
    # 'pleural_tract',
    # 'DRB_vessel_convergence',
]

CTR_FEATURES = ['neg300_volume_ratio']
# CTR_FEATURES = ['neg300_solid_volume(mm3)']

SOLID_SELECTED_FEATURES = [  # 'nodule_2d_diameter(mm)',
    'nodule_2d_diameter_clinical(mm)',  # 'nodule_3d_diameter(mm)',
    # 'nodule_volume(mm3)',
    # 'nodule_mean_hu',
    # 'nodule_mass(mg)',
    # 'nodule_entropy',
    # 'neg300_solid_2d_diameter(mm)',
    # 'neg300_solid_2d_diameter_clinical(mm)',
    # 'neg300_solid_3d_diameter(mm)',
    # 'neg300_solid_volume(mm3)',
    'neg300_volume_ratio',  # 'neg300_solid_entropy',
    # 'neg300_solid_mean_hu',
    # 'neg300_solid_mass(mg)',
    # 'neg200_solid_2d_diameter(mm)',
    # 'neg200_solid_2d_diameter_clinical(mm)',
    # 'neg200_solid_3d_diameter(mm)',
    # 'neg200_solid_volume(mm3)',
    # 'neg200_volume_ratio',
    # 'neg200_solid_entropy',
    # 'neg200_solid_mean_hu',
    # 'neg200_solid_mass(mg)',
    # 'neg100_solid_2d_diameter(mm)',
    # 'neg100_solid_2d_diameter_clinical(mm)',
    # 'neg100_solid_3d_diameter(mm)',
    # 'neg100_solid_volume(mm3)',
    # 'neg100_volume_ratio',
    # 'neg100_solid_entropy',
    # 'neg100_solid_mean_hu',
    # 'neg100_solid_mass(mg)',
    # '0_solid_2d_diameter(mm)',
    # '0_solid_2d_diameter_clinical(mm)',
    # '0_solid_3d_diameter(mm)',
    # '0_solid_volume(mm3)',
    # '0_volume_ratio',
    # '0_solid_entropy',
    # '0_solid_mean_hu',
    # '0_solid_mass(mg)',
]
