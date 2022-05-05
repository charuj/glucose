import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import dataframe_image as dfi

'''
This class does the following:
1. Plots correlation matrix of clinical data to identify multicollinearity
2. Checks for class imbalance in terms of patient type (diabetic, normal, pre-diabetic), 
and meal types (PB, Corn flakes, energy bar) 
3. Plots distributions of certain features 
4. Identifies skew & kurtosis
5. Performs ANOVA test of clinical data to see if means of data are significantly different based on diabetes-status

'''

class EDA:
    def __init__(self, clinical_df, meal_df, cgm_df, metrics_list) -> None:
        self.clinical_df= clinical_df
        self.meal_df= meal_df
        self.cgm_df= cgm_df 
        self.metrics_list= metrics_list

    def corr_matrix(self, df,feat_list, save_name) -> None:
        df_subset= df[feat_list]
        sns.heatmap(df_subset.corr(), annot=True, cmap="YlGnBu")
        plt.savefig(save_name)
    
    def diabetic(self, df_clinical) -> pd.DataFrame:
        '''
        HbA1c ≥ 6.5%, fasting blood glucose ≥ 126 mg/dL, or 2-hour glucose during 75 gram OGTT ≥ 200 mg/dL; 
        14 had prediabetes, defined as HbA1c > 5.7% and < 6.5%, fasting blood glucose 100–125 mg/dL, or 2-hour glucose during OGTT 140–199 mg/dL; 
        the remainder were normoglycemic, defined as fasting and 2-hour OGTT plasma glucose and HbA1c below the diagnostic thresholds for prediabetes and diabetes
        '''
        df_diab= df_clinical
        df_diab['Diabetes_status'] = 'Normal'
        df_diab['Diabetes_status'].loc[((df_diab['A1C']>=6.5) & (df_diab['FBG']>.126)) | (df_diab['ogtt.2hr']>=200)] = 'Diabetic'
        df_diab['Diabetes_status'].loc[((df_diab['A1C']>=5.7) & (df_diab['A1C'] < 6.5) & (df_diab['FBG']>=100)  & (df_diab['FBG']<=125)) | ((df_diab['ogtt.2hr']>= 140) & (df_diab['ogtt.2hr'] <=199)) ] = 'Pre'
        return df_diab
    

    def anova_summary(self, df_diab, savename) -> None:
    
        condition_diabetic = df_diab['Diabetes_status'] == 'Diabetic'
        condition_pre = df_diab['Diabetes_status'] == 'Pre'
        condition_norm = df_diab['Diabetes_status'] == 'Normal'

        diab_outcome = df_diab.loc[condition_diabetic, self.metrics_list]
        pre_outcome = df_diab.loc[condition_pre, self.metrics_list]
        norm_outcome = df_diab.loc[condition_norm, self.metrics_list]

        results_dict={}

        for metric in self.metrics_list:
            F, p = f_oneway(diab_outcome[metric], pre_outcome[metric], norm_outcome[metric])
            if p<0.05:
                sig= 'Significant'
            else:
                sig= 'Insignificant'
            results_dict[metric]= [F,p,sig]
        
        anova_df= pd.DataFrame.from_dict(results_dict, orient='index').rename(columns={0: 'F-statistic', 1: "p value", 2:' Difference Significance'})
        dfi.export(anova_df, savename)

    
    def skew_kurtosis(self, df,  savename) -> None:
        results=[]
        skew_val= df['GlucoseValue'].skew()
        kurt_val= df['GlucoseValue'].kurt()
        results.append(['GlucoseValue', skew_val, kurt_val])   
        results_df= (pd.DataFrame(results)).rename(columns={0: savename , 1: 'Skewness', 2: 'Kurtosis'})
        dfi.export(results_df, savename + '.png')


    def check_imbalance_diabetic(self, df_diab, savename):
        mask_diabetic = df_diab['Diabetes_status'] == 'Diabetic'
        mask_normal = df_diab['Diabetes_status'] == 'Normal'
        mask_pre = df_diab['Diabetes_status'] == 'Pre'

        df_diabetic= df_diab[mask_diabetic]
        df_normal= df_diab[mask_normal]
        df_pre= df_diab[mask_pre]

        num_diabetic= df_diabetic.shape[0]
        num_normal= df_normal.shape[0]
        num_pre= df_pre.shape[0]

        perc_diabetic= num_diabetic/(num_diabetic + num_normal + num_pre)
        perc_normal= num_normal/(num_diabetic + num_normal + num_pre)
        perc_pre= num_pre/(num_diabetic + num_normal + num_pre)

        results_dict={'Percentage Diabetic': [perc_diabetic], 'Percentage Normal': [perc_normal], 'Percentage Pre': [perc_pre]}
        results_df= pd.DataFrame.from_dict(results_dict)
        dfi.export(results_df, savename)


    def meal_classes(self, savename):
        df_mealevents= self.meal_df.groupby([self.meal_df.time.dt.date, self.meal_df.Meal, self.meal_df.userID], as_index=False).count()
        tmp= df_mealevents.groupby([self.meal_df.Meal]).count()
        tmp= (tmp.drop(['userID', 'time', 'GlucoseValue'], axis= 'columns')).rename(columns={'Meal': 'Count'})
        dfi.export(tmp, savename)


    def histograms(self, df, feat_list, save_name) -> None:
        df_subset=df[feat_list]
        plt.figure()
        sns.histplot(data=df_subset, kde=True)
        plt.savefig(save_name, bbox_inches='tight')




