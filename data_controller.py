import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

class YourDataAnalyzer:


    def __init__(self, input_array):
        
        self.input_array = input_array


    def data(self):

        try:

            df = pd.DataFrame(self.input_array)

            return df
        
        except Exception as e:

            raise Exception(f"An error occurred at data: {str(e)}")


    def to_type(self, df):

        try:

            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df['PJME_MW']  = pd.to_numeric(df['PJME_MW'])

            return df
        
        except Exception as e:

            raise Exception(f"An error occurred at to_type: {str(e)}")


    def train_test_size(self, df):

        try:
            
            df_sorted  = df.sort_values(by='Datetime')
            train_size = int(len(df_sorted) * 0.80)
            train_df   = df_sorted[:train_size]
            test_df    = df_sorted[train_size:]

            return train_df, test_df
        
        except Exception as e:

            raise Exception(f"An error occurred at train_test_size: {str(e)}")


    def rename(self, train_df, test_df):

        try:

            train_df = train_df.rename(columns={'Datetime': 'ds', 'PJME_MW': 'y'})
            test_df  = test_df.rename(columns={'Datetime': 'ds', 'PJME_MW': 'y'})

            return train_df, test_df
        
        except Exception as e:

            raise Exception(f"An error occurred at rename: {str(e)}")


    def calculate_holidays(self, df_sorted):

        cal                        = pd.to_datetime(pd.Series(calendar().holidays(start=df_sorted['Datetime'].min(), end=df_sorted['Datetime'].max())))
        df_holiday                 = df_sorted.copy()
        df_holiday['is_holiday']   = df_holiday['Datetime'].dt.date.isin(cal.dt.date)
        holiday_df                 = df_holiday.loc[df_holiday['is_holiday']].reset_index(drop=True).rename(columns={'Datetime': 'ds', 'PJME_MW': 'y'}).drop(['is_holiday'], axis=1)
        holiday_df['holiday']      = 'USFederalHoliday'

        return holiday_df[['ds', 'holiday']]

    
    def clean_data(self, train_df, test_df):

        # Clean Data
        train_df['y_clean']                              = train_df['y']
        train_df.loc[train_df['y'] < 19000, 'y_clean']   = np.nan
        train_df                                         = train_df.drop(columns="y")

        train_df.dropna(subset=['y_clean'], inplace=True)
        train_df.rename(columns={'y_clean': 'y'}, inplace=True)

        test_df['y_clean']                               = test_df['y']
        test_df.loc[test_df['y'] < 19000, 'y_clean']     = np.nan
       
        test_df = test_df.drop(columns="y")
        test_df.rename(columns={'y_clean': 'y'}, inplace=True)

        return train_df, test_df


    def create_model(self, train_df, holiday_df=None):

        if holiday_df is None:
        
            model = Prophet()
        
        else:
        
            model = Prophet(holidays=holiday_df)
        
        model.fit(train_df)
        
        return model

    def evaluate_model(self, model, test_df):

        forecast  = model.predict(test_df)
        r2        = r2_score(test_df['y'], forecast['yhat'])
        mape      = mean_absolute_percentage_error(test_df['y'], forecast['yhat'])
        
        return r2, mape

    def process_analysis(self, df):

        try:

            df_sorted          = df.sort_values(by='Datetime')

            train_df, test_df  = self.train_test_size(df_sorted)

            # Rename columns

            train_df, test_df  = self.rename(train_df, test_df)

            # Model without holidays

            model_v2           = self.create_model(train_df)
            r2_v2, mape_v2     = self.evaluate_model(model_v2, test_df)

            # Calculate holidays

            holiday_df         = self.calculate_holidays(df_sorted)

            # Model with holidays

            model_with_holidays                 = self.create_model(train_df, holiday_df)
            r2_with_holiday, mape_with_holiday  = self.evaluate_model(model_with_holidays, test_df)

            # Clean Data

            train_df_cleaned, test_df_cleaned   = self.clean_data(train_df, test_df)

            # Model with fixed data

            fixed_model                                 = self.create_model(train_df_cleaned)
            r2_with_fixed_model, mape_with_fixed_model  = self.evaluate_model(fixed_model, test_df_cleaned)

            # Model with fixed data and holidays

            fixed_model_with_holiday                                    = self.create_model(train_df_cleaned, holiday_df)
            r2_with_fixed_model_holiday, mape_with_fixed_model_holiday  = self.evaluate_model(fixed_model_with_holiday, test_df_cleaned)

            # Final Model

            final_df              = pd.concat([train_df_cleaned, test_df_cleaned], axis=0)
            final_model           = self.create_model(final_df, holiday_df)

            future                = final_model.make_future_dataframe(periods=100, freq='H')
            forecast_final        = final_model.predict(future)
            selected_columns      = ['ds', 'yhat']
            last_100_rows         = forecast_final[selected_columns].tail(100)
            last_100_rows['ds']   = last_100_rows['ds'].astype(str)

            # Convert DataFrames to JSON-serializable format

            last_100_rows_json    = last_100_rows.to_dict('records')

            return {

                "model_v2_r2"                   : r2_v2,
                "model_v2_mape"                 : mape_v2,
                "model_with_holidays_r2"        : r2_with_holiday,
                "model_with_holidays_mape"      : mape_with_holiday,
                "fixed_model_r2"                : r2_with_fixed_model,
                "fixed_model_mape"              : mape_with_fixed_model,
                "fixed_model_with_holiday_r2"   : r2_with_fixed_model_holiday,
                "fixed_model_with_holiday_mape" : mape_with_fixed_model_holiday,
                "last_100_rows"                 : last_100_rows_json
            }

        except Exception as e:

            return f"An error occurred during analysis: {str(e)}"

    def analyze(self):

        try:

            df = self.data()
            df = self.to_type(df)

            return self.process_analysis(df)
        
        except Exception as e:

            return f"An error occurred during analysis: {str(e)}"



