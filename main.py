import matplotlib.pyplot as plt
from data_loader import Dataprocess
from eda import EDA
from model import Model


''''
This script does the following:
1. Runs data_loader.py (data import and processing)
2. Runs eda.py (exploratory data analysis)
3. Runs model.py (feature engineering & logistic regression for multi-class)

'''



def run_dataloader(meal_path, cgm_path, db_path, clinical_metricslist):
    load= Dataprocess(meal_path, cgm_path, db_path, clinical_metricslist)
    load.db_data()
    load.read_tsv()
    load.check_missing(load.meal_df, 'missingdata_meals.png')
    load.check_missing(load.cgm_df, 'missingdata_cgm.png')
    load.check_missing(load.clinical_df, 'missingdata_clinical.png')
    load.drop_missing()
    clinical_clean= load.drop_missing_clinical()
    cgm_formatted= load.format_cgmdata()
    meal_formatted= load.format_mealdata()
    return clinical_clean, cgm_formatted, meal_formatted


def run_eda(clinical_clean, meal_formatted, cgm_formatted, metrics_list):
    eda=EDA(clinical_clean, meal_formatted, cgm_formatted, metrics_list)
    eda.corr_matrix(eda.clinical_df,eda.metrics_list, 'clnical_corrmatrix.png')
    df_diab = eda.diabetic(eda.clinical_df)   
    eda.anova_summary(df_diab, 'clinical_anova.png') 
    eda.skew_kurtosis(eda.meal_df,'meal_skewkurt')
    eda.skew_kurtosis(eda.cgm_df, 'cgm_skewkurt')  
    eda.check_imbalance_diabetic(df_diab, 'diabetes_classimbalance.png')
    eda.meal_classes('meal_classes.png')
    for metric in eda.metrics_list:
        eda.histograms(clinical_clean, metric, str(metric) + '.png') 
    eda.histograms(eda.meal_df,['GlucoseValue'], 'meal_hist.png')
    eda.histograms(eda.cgm_df, ['GlucoseValue'], 'cgm_hist.png')

def run_model(meal_formatted, cgm_formatted,clinical_clean, metrics_list ):
    mod= Model(meal_formatted, cgm_formatted,clinical_clean, metrics_list)
    mod.clinical_input()
    mod.nutrients()
    mod.userlevel_cgm()
    mod.userlevel_meal()
    df_inputs_ohe, df_targets= mod.inputs_targets()
    mod.model(df_inputs_ohe, df_targets)




if __name__=='__main__':

    cgm_path= 'cgm_data.tsv'
    meal_path= 'meals_cgm.tsv'
    db_path= 'maintable.db'
    clinical_metrics_list=['Age', 'BMI','A1C','FBG', 'ogtt.2hr','insulin', 'Trg', 'HDL'  ]

    clinical_clean, cgm_formatted, meal_formatted= run_dataloader(meal_path, cgm_path, db_path, clinical_metrics_list)
    run_eda(clinical_clean, meal_formatted, cgm_formatted, clinical_metrics_list)
    run_model(meal_formatted, cgm_formatted,clinical_clean, clinical_metrics_list + ['userID'])













