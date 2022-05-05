
from scipy.stats import variation
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics





class Model:
    def __init__(self, meal_df, cgm_df, clinical_df, metrics_list) -> None:
        self.meal_df= meal_df 
        self.cgm_df= cgm_df
        self.clinical_df= clinical_df
        self.metrics_list= metrics_list
    
    def clinical_input(self) -> None:
        self.clinical_input_df= self.clinical_df[self.metrics_list]

    def nutrients(self):
        pb1_dict= {'Meal':['PB 1'],'calories':[430], 'fat': [20], 'carb': [51], 'sugar': [12], 'fiber': [12], 'protein': [18]}
        pb2_dict= {'Meal': ['PB 2'],'calories': [430], 'fat': [20], 'carb':[51], 'sugar': [12], 'fiber': [12], 'protein': [18]}
        bar1_dict= {'Meal':['Bar 1'],'calories': [370], 'fat': [18], 'carb': [48], 'sugar': [19], 'fiber': [6], 'protein': [9]}
        bar2_dict= {'Meal':['Bar 2'],'calories': [370], 'fat': [18], 'carb': [48], 'sugar': [19], 'fiber': [6], 'protein': [9]}
        cf1_dict= {'Meal':['CF 1'],'calories': [280], 'fat': [2.5], 'carb':[54], 'sugar': [35.2], 'fiber': [3.3], 'protein': [1.1]}
        cf2_dict= {'Meal':['CF 2'],'calories':[280], 'fat': [2.5], 'carb':[54], 'sugar': [35.2], 'fiber': [3.3], 'protein': [1.1]}

        pb1= pd.DataFrame.from_dict(pb1_dict)
        pb2= pd.DataFrame.from_dict(pb2_dict)
        bar1= pd.DataFrame.from_dict(bar1_dict)
        bar2= pd.DataFrame.from_dict(bar2_dict)
        cf1= pd.DataFrame.from_dict(cf1_dict)
        cf2= pd.DataFrame.from_dict(cf2_dict)

        frames= [pb1, pb2, bar1, bar2, cf1, cf2]
        self.nutrients_df= pd.concat(frames)
        
    
    def coeff_variability(self, df):
        coeff_var= variation(df)
        return coeff_var
    
    def time_inrange(self, df):
        time_in= len(df.loc[(df>= 72) & (df<= 110)])
        total_rows= df.shape[0]
        inrange= time_in/total_rows 
        return inrange 
    
    def userlevel_cgm(self):
        df_grouped= self.cgm_df.groupby([self.cgm_df.subjectId ])
        self.df_var_tir= (df_grouped.agg({'GlucoseValue':{self.coeff_variability, self.time_inrange}}).reset_index()).fillna(0).rename(columns={'subjectId': 'userID' })
        self.df_var_tir.columns = self.df_var_tir.columns.map(''.join)

    def glucose_peak(self, df):
        peak= np.amax(df)
        return peak
        
    def userlevel_meal(self):
        df_grouped= self.meal_df.groupby([self.meal_df.userID, self.meal_df.time.dt.date, self.meal_df.Meal])
        self.df_peak= df_grouped.GlucoseValue.agg(self.glucose_peak).reset_index()
        self.df_peak['Quartile']= pd.qcut(self.df_peak['GlucoseValue'], q=4, labels=[1,2,3,4]) 

    def ohe(self, df):
        df_ohe= pd.get_dummies(df)
        return df_ohe
    

    def inputs_targets(self):
        merged_1= pd.merge(self.clinical_input_df, self.df_peak, how="inner", on='userID')
        merged_2= pd.merge(merged_1, self.df_var_tir, how="inner", on='userID')
        merged_all= (pd.merge(merged_2, self.nutrients_df, how="left", on='Meal')) ##ONE HOT ENCODE MEAL
        target_list=['Quartile']
        input_list= ['Age', 'BMI', 'A1C', 'FBG', 'ogtt.2hr', 'insulin', 'Trg', 'HDL','Meal', 'GlucoseValue', 
       'GlucoseValuetime_inrange', 'GlucoseValuecoeff_variability', 'calories','fat', 'carb', 'sugar', 'fiber', 'protein']

        df_targets= merged_all[target_list]
        df_inputs= merged_all[input_list]
        df_inputs_ohe= self.ohe(df_inputs)
        return df_inputs_ohe, df_targets
    
    def model(self, input,target):
        
        X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.25, random_state=42)

        clf = LogisticRegression(random_state=0, multi_class='multinomial').fit(X_train, y_train)
        y_pred= clf.predict(X_test)

        #training accuracy 
        print ("TRAINING ACCURACY: ", clf.score(X_train, y_train))

        # Multi-class precision and recall, among other metrics
        print(metrics.classification_report(y_test, y_pred, digits=3))


        


        

        

    

        
    

    
