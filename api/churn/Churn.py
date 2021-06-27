import pickle
import inflection
import numpy as np
import pandas as pd


class Churn( object ):
    def __init__( self ):
        self.home_path = r'C:\Users\Cliente\repos\pa003_churn_predict\churn_predict\parameter'
        self.credit_score_scaler     = pickle.load( open( self.home_path + '/credit_score_scaler.pkl', 'rb' ) )
        self.ternure_scaler          = pickle.load( open( self.home_path + '/tenure_scaler.pkl', 'rb' ) )
        self.balance_scaler          = pickle.load( open( self.home_path + '/balance_scaler.pkl', 'rb' ) )
        self.estimated_salary_scaler = pickle.load( open( self.home_path + '/estimated_salary_scaler.pkl', 'rb' ) )
        self.amount_scaler           = pickle.load( open( self.home_path + '/amount_scaler.pkl', 'rb' ) )
        self.credit_score_product    = pickle.load( open( self.home_path + '/credit_score_product_scaler.pkl', 'rb' ) )
        self.balance_product_scaler  = pickle.load( open( self.home_path + '/balance_product_scaler.pkl', 'rb' ) )
        self.age_scaler              = pickle.load( open( self.home_path + '/age_scaler.pkl', 'rb' ) )
        self.credit_score_age_scaler = pickle.load( open( self.home_path + '/credit_score_age_scaler.pkl', 'rb' ) )
        self.balance_age_scaler      = pickle.load( open( self.home_path + '/balance_age_scaler.pkl', 'rb' ) )


    
    def data_cleaning( self, df1 ):    

        ## 1.1 Rename columns

        cols_old = df1.columns

        snakecase = lambda x: inflection.underscore( x )

        cols_new = list( map ( snakecase, cols_old ) )
        # rename
        df1.columns = cols_new
        
        return df1


    def feature_engineering( self, df2 ):
    
        # Credit score per age
        df2['credit_score_age'] = df2['credit_score'] / df2['age']

        # amount = estimated_salary + balance
        df2['amount'] = df2['estimated_salary'] + df2['balance']

        # Credit score per product
        df2['credit_score_product'] = df2['credit_score'] / df2['num_of_products']

        # balance per age
        df2['balance_age'] = df2['balance'] / df2['age']

        # balance per number of products
        df2['balance_product'] = df2['balance'] / df2['num_of_products']

        # Mean Salary
        mean_salary = np.mean( df2['estimated_salary'].values)
        df2['mean_salary'] = df2['estimated_salary'].apply( lambda x: 'below average' if x < mean_salary else 'average' if x==mean_salary else 'above average'  )

        # Balance scale
        df2['balance_scale'] = df2['balance'].apply( lambda x: 'zero' if x==0 else 'normal_balance' if (x > 0) & (x <= 76485.89) else 'high_balance' )

        # 3.0 VARIABLE SELECTION

        df2 = df2.drop( ['row_number','surname'], axis=1 )
        
        return df2


    def data_preparation( self, df5 ):

        ## 5.1 Rescalling
        # Credit Score
        df5['credit_score'] = self.credit_score_scaler.fit_transform( df5[['credit_score']].values )

        # tenure
        df5['tenure'] = self.ternure_scaler.fit_transform( df5[['tenure']].values )

        # balance
        df5['balance'] = self.balance_scaler.fit_transform( df5[['balance']].values )

        # estimated_salary
        df5['estimated_salary'] = self.estimated_salary_scaler.fit_transform( df5[['estimated_salary']].values )

        # amount - minmax
        df5['amount'] = self.amount_scaler.fit_transform( df5[['amount']].values )

        # credit_score_product
        df5['credit_score_product'] = self.credit_score_product.fit_transform( df5[['credit_score_product']].values )

        # balance_product
        df5['balance_product'] = self.balance_product_scaler.fit_transform( df5[['balance_product']].values )

        # age 
        df5['age'] = self.age_scaler.fit_transform( df5[['age']].values )

        # credit_score_age
        df5['credit_score_age'] = self.credit_score_age_scaler.fit_transform( df5[['credit_score_age']].values )

        # balance_age
        df5['balance_age'] = self.balance_age_scaler.fit_transform( df5[['balance_age']].values )

        ## 5.3 Transformation

        ### 5.3.1 Encoding

        # Label enconding mean_salary
        mean_salary_dict = {'above average': 1, 'below average': 2}
        df5['mean_salary'] = df5['mean_salary'].map( mean_salary_dict )

        # Label encoding balance_scale
        balance_scale_dict = { 'zero': 0, 'high_balance': 1, 'normal_balance': 2 }
        df5['balance_scale'] = df5['balance_scale'].map( balance_scale_dict )

        # label encoding gender 
        gender_dict = { 'Male':1 , 'Female':2 }
        df5['gender'] = df5['gender'].map( gender_dict ) 

        # One hot encoding encoding geography
        df5 = pd.get_dummies( df5, prefix=['geo'], columns=['geography'] ) 
        
        # Selected colunms
        cols_selected = ['age', 'num_of_products', 'is_active_member', 'estimated_salary', 'amount', 'balance', 'credit_score', 'geo_Germany', 'geo_France', 'geo_Spain']
        
        return df5[cols_selected]
    
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction

        pred = model.predict_proba( test_data )
        #pred = model.predict( test_data )

        
        # join pred into the original data

        original_data['prediction'] = pred[:,1]
        #original_data['prediction'] = pred
        
        return original_data.to_json( orient='records', date_format='iso' )
      
