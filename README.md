# Multimodal Fusion

[`Paper`]: Multimodal Fusion of Liquid Biopsy and CT Enhances Differential Diagnosis of Early-stage Lung Adenocarcinoma

This research explores the potential of multimodal fusion for the differential diagnosis of early-stage lung adenocarcinoma (LUAD) (tumor sizes < 2 cm). It combines liquid biopsy biomarkers, specifically extracellular vesicle long RNA (evlRNA) and the computed tomography (CT) attributes. The fusion model achieves an impressive cross-validated four-category (Benign, AIS, MIA and IA) AUC of 91.9%, along with a benign-malignant AUC of 94.8% (sensitivity: 89.1%, specificity: 94.3%). These outcomes outperform the diagnostic capabilities of the single-modal models and human experts. A comprehensive SHAP explanation is provided to offer deep insights into model predictions. Our findings reveal the complementary interplay between evlRNA and image-based characteristics, underscoring the significance of integrating diverse modalities in diagnosing early-stage LUAD.


![pipeline](pipeline.png?raw=true)


## Installation

The code requires `python>=3.8`

Install the requirements:
```
pip install -r requirments.txt
```

## Get Started

The Dataset is available at https://doi.org/10.5281/zenodo.10025009:

1. Four category classification of all models.
```shell script
python main.py --case_name=test_classification --model_name=xgboost \
--classification_method=four_classification --boostrap_num=1 --start_random_state=42 --n_splits=5 \
--test_features="['evlRNA', 'CTR', 'vCTR', 'Rad',  'Junior', 'Senior', 'evlRNA + CTR', 'evlRNA + vCTR','evlRNA + Rad',  'evlRNA + Junior','evlRNA + Senior', 'evlRNA + Rad + Junior',  'evlRNA + Rad + Senior']" \
```
2. SHAP explanation
```shell script
# total summary plot
python main.py --case_name=test_classification --model_name=xgboost \
--classification_method=four_classification --boostrap_num=1 --start_random_state=42 \
--cross_validation=0 --n_splits=5 --test_features="['evlRNA', 'Rad', 'evlRNA + Rad']" \
--need_explain=True --total_summary_plot=True  

# single summary plot
python main.py --case_name=test_classification --model_name=xgboost \
--classification_method=four_classification --boostrap_num=1 --start_random_state=42 \
--cross_validation=0 --n_splits=5 --test_features="['evlRNA', 'Rad', 'evlRNA + Rad']" \
--need_explain=True --single_summary_plot=True

# force plot IA index 2
python main.py --case_name=test_classification --model_name=xgboost \
--classification_method=four_classification --boostrap_num=1 --start_random_state=42 \
--cross_validation=0 --n_splits=5 --test_features="['evlRNA', 'Rad', 'evlRNA + Rad']" \
--need_explain=True --instance_force_plot=True --instance_index=2 

```
## License
The model is licensed under the Apache 2.0 license.