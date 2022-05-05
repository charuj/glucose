from ast import Mod
import unittest
import pandas as pd 
import numpy as np 
from data_loader import * 
from model import Model 
from pandas.util.testing import assert_frame_equal # <-- for testing dataframes




class TestModel(unittest.TestCase):
    def test_OHE(self):
        data = [['CF', 40], ['BAR', 15], ['PB1', 10], ['PB2', 10],['CF', 40] ]
        df = pd.DataFrame(data)
        data_ohe= [[40,0,1,0, 0], [15, 1, 0 , 0 , 0], [10,0,0 ,1 ,0], [10,0 , 0, 0 , 1],[40,0,1,0, 0]]
        df_ohe= pd.DataFrame(data_ohe, columns=[1, '0_BAR', '0_CF', '0_PB1', '0_PB2'])
        
        meal_df= pd.DataFrame(np.ones((5, 5)))
        cgm_df= pd.DataFrame(np.ones((5, 5)))
        clinical_df = pd.DataFrame(np.ones((5, 5)))
        metrics_list=['Age', 'BMI','A1C','FBG', 'ogtt.2hr','insulin', 'Trg', 'HDL'  ]

        model= Model(meal_df, cgm_df, clinical_df, metrics_list)
        model_ohe= model.ohe(df)

        assert_frame_equal(model_ohe.astype('int64'), df_ohe.astype('int64'))



    def test_glucosepeak(self):
        meal_df= pd.DataFrame(np.ones((5, 5)))
        cgm_df= pd.DataFrame(np.ones((5, 5)))
        clinical_df = pd.DataFrame(np.ones((5, 5)))
        metrics_list=['Age', 'BMI','A1C','FBG', 'ogtt.2hr','insulin', 'Trg', 'HDL'  ]

        model= Model(meal_df, cgm_df, clinical_df, metrics_list)

        peak_df= pd.DataFrame([0,4,7,27,9,9,13])
        peak= model.glucose_peak(peak_df)
        self.assertEqual(peak.to_list(), [27])




if __name__ == '__main__':
    unittest.main()
