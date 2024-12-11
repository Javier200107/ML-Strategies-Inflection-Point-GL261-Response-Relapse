import pandas as pd
import logging

import config
from src.preprocessing import col_standardisation,col_minmax_scaling,\
                                            imputate_row_missing_values_with_median,\
                                            df_to_numeric\
                                            
class Processing:
    """
    Class to ingest and process data for clustering algorithms.
    """

    def __init__(self, 
                 source_path=config.RADIOMICS_FINAL_FEATURES_NO_FILTERS,
                 df=None,
                 all_features=False):
        
        self.source_path = source_path
        self.df = df
        self.all_features = all_features

    def read(self):
        """
        Read the dataset from the source path.
        
        Returns:
            df: DataFrame containing dataset data.
        """

        df_features = pd.read_csv(self.source_path)

        logging.info(f"\nData read successfully ({df_features.shape[0]} rows, {df_features.shape[1]} columns)")
        
        if self.all_features:
            self.check_features_in_df(df_features, config.RADIOMICS_FINAL_FEATURES)
        else:
            self.check_features_in_df(df_features, config.RADIOMICS_FINAL_FEATURES_NO_FILTERS)

        self.df = df_features
        return df_features
    
    def preprocess(self, start_col=2, remove_mice_id=True):
        '''
        General Preprocessor

        Args:
            start_col: Index of the first column to preprocess
            remove_mice_id: Boolean to remove mice id from the dataset
        Preprocess:
            Numeric columns:
                - Convert to float
                - Imputate missing values with median
                - Scale data
        
        Returns:
            prep_df: Preprocessed DataFrame
        '''

        prep_df = self.df.copy()

        # Numeric preprocessing
        
        if remove_mice_id:
            # Drop mice id from the dataset
            prep_df.drop(columns=["m_id"], inplace=True)
        
        columns = prep_df.columns[start_col:] 

        # Ensure the values are Real numbers
        prep_df = df_to_numeric(prep_df, columns)

        # Impute Missing Values
        prep_df = imputate_row_missing_values_with_median(prep_df, columns)

        # Scale data
        prep_df = col_minmax_scaling(prep_df, columns)

        self.df = prep_df
        return prep_df
    
    def drop_late_days(self, days):
        '''
        Drop the rows with day_of_study greater than the specified days

        Args:
            days: Maximum day_of_study to keep in the dataset
        '''
        self.df = self.df[self.df["day_of_study"] <= days]
        return self.df
    
    def drop_mice_group(self, m_group):
        '''
        Drop the rows with the specified mice group

        Args:
            m_group: Mice group to drop from the dataset
        '''
        self.df = self.df[self.df["group_name"] != m_group]
        return self.df
    
    @staticmethod
    def check_features_in_df(df, cols):
        """
        Check if the features are in the dataset

        Args:
            df: DataFrame containing the dataset
            cols: List of features to check
        """
        out_cols = [col for col in cols if col not in df.columns]
        if len(out_cols) > 0:

            logging.warning(f"Columns {out_cols} are not in the dataset")

if __name__ == "__main__":
    import numpy as np

    dataclass = Processing()
    df = dataclass.read()
    df = df.iloc[:, 1:]
    print(df.head())

    print(df.loc[0, "10Percentile"])
    df.loc[0, "10Percentile"] = np.nan
    print(df.loc[0, "10Percentile"])
    df = dataclass.preprocess()
    print(df.loc[0, "10Percentile"])

    print(df["90Percentile"].mean(), df["10Percentile"].mean())

    print(df.head())