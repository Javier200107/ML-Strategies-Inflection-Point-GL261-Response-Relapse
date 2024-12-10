from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from itertools import combinations
import random

N_SPLITS = 4
TEST_SIZE = 0.6
SEED = 42
SAMPLE_SIZE = 10

class DataSplitting:
    """Class to split data into training and testing sets."""
    @staticmethod
    def expanding_window_split(df: pd.DataFrame, n_splits: int = N_SPLITS) -> dict:
        """
        Perform Expanding Window Split

        Args:
            df (pd.DataFrame): DataFrame containing all instances, sorted by time.
            n_splits (int): Number of splits for expanding windows.

        Returns:
            splits (dict): Dictionary with train-test splits for each fold.
        """
        splits = {}

        # Sort the DataFrame by 'day_of_study' to ensure chronological order
        df = df.sort_values(by='day_of_study').reset_index(drop=True)

        # TimeSeriesSplit for expanding window
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (train_indices, test_indices) in enumerate(tscv.split(df)):
            
            # Prepare training, and testing sets
            X_train = df.iloc[train_indices].drop(columns=['group_name', 'm_id']).reset_index(drop=True)
            y_train = df.iloc[train_indices]['group_name'].reset_index(drop=True)

            # Test data
            X_test = df.iloc[test_indices].drop(columns=['group_name', 'm_id']).reset_index(drop=True)
            y_test = df.iloc[test_indices]['group_name'].reset_index(drop=True)

            # Store splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
        
        return splits
    
    @staticmethod
    def split_data_population_informed(df: pd.DataFrame, n_splits=N_SPLITS, test_size=TEST_SIZE, seed=SEED):
        """
        Perform population-informed nested cross-validation.

        Args:
            df (pd.DataFrame): The dataset with instances and metadata.
            n_splits (int): Number of splits for the outer loop.
            test_size (float): Proportion of groups used for testing.

        Returns:
            splits (dict): Dictionary containing training, and test sets.
        """
        np.random.seed(seed=seed)
        splits = {}
        group_ids = df['m_id'].unique()
        total_groups = len(group_ids)
        test_groups_count = max(1, int(total_groups * test_size))

        print(f"Total groups: {total_groups}, Test groups: {test_groups_count}")

        # Outer loop to vary test groups
        for i in range(n_splits):
            # Randomly select test groups
            test_ids = np.random.choice(group_ids, test_groups_count, replace=False)

            # Separate data into test, and training groups
            test_group = df[df['m_id'].isin(test_ids)].sort_values(by='day_of_study').reset_index(drop=True)
            train_group = df[~df['m_id'].isin(test_ids)].sort_values(by='day_of_study').reset_index(drop=True)

            # Define train and test indices for the test group
            train_indices = list(range(0, int(len(test_group) * 0.60)))  # First 60% for training
            test_indices = list(range(int(len(test_group) * 0.60), len(test_group)))  # Remaining 40% for testing

            # Prepare training, and testing sets
            X_train = pd.concat([train_group.drop(columns=['m_id', 'group_name']),
                                test_group.iloc[train_indices].drop(columns=['m_id', 'group_name'])]).reset_index(drop=True)
            y_train = pd.concat([train_group['group_name'], test_group.iloc[train_indices]['group_name']]).reset_index(drop=True)

            X_test = test_group.iloc[test_indices].drop(columns=['m_id', 'group_name']).reset_index(drop=True)
            y_test = test_group.iloc[test_indices]['group_name'].reset_index(drop=True)

            # Store splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }

        return splits
    
    @staticmethod
    def expanding_window_split_by_mice_group(df: pd.DataFrame, n_splits: int = N_SPLITS) -> dict:
        """
        Perform Expanding Window Split by Mice Group

        Args:
            df (pd.DataFrame): DataFrame containing all instances, sorted by time.
            n_splits (int): Number of splits for expanding windows.

        Returns:
            splits (dict): Dictionary with train-test splits for each fold.
        """
        
        splits = {}
        for _, group in df.groupby('m_id'):
            group = group.sort_values(by='day_of_study').reset_index(drop=True)

            if len(group) < n_splits + 1:
                split_idx = 0
                if split_idx not in splits:
                    X_train, y_train = group.drop(columns=['m_id', 'group_name']), group['group_name']
                    splits[split_idx] = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': None,
                        'y_test': None
                    }
                else:
                    if splits[split_idx]['X_train'] is None and splits[split_idx]['y_train'] is None:
                        X_train, y_train = group.drop(columns=['m_id', 'group_name']), group['group_name']
                        splits[split_idx]['X_train'] = X_train
                        splits[split_idx]['y_train'] = y_train
                    else:
                        X_train, y_train = group.drop(columns=['m_id', 'group_name']), group['group_name']
                        splits[split_idx]['X_train'] = pd.concat([splits[split_idx]['X_train'], X_train], axis=0, ignore_index=True)
                        splits[split_idx]['y_train'] = pd.concat([splits[split_idx]['y_train'], y_train], axis=0, ignore_index=True)
                continue

            tscv = TimeSeriesSplit(n_splits=n_splits + 1).split(group) 
            next(tscv)  # Skip the first split

            for split_idx, (train_cv_indices, test_indices) in enumerate(tscv):
                X_train, y_train = group.iloc[train_cv_indices].drop(columns=['m_id', 'group_name']), group.iloc[train_cv_indices]['group_name']
                X_test, y_test = group.iloc[test_indices].drop(columns=['m_id', 'group_name']), group.iloc[test_indices]['group_name']
                if split_idx not in splits:
                    splits[split_idx] = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test
                    }
                else:
                    splits[split_idx]['X_train'] = pd.concat([splits[split_idx]['X_train'], X_train], axis=0, ignore_index=True)
                    splits[split_idx]['y_train'] = pd.concat([splits[split_idx]['y_train'], y_train], axis=0, ignore_index=True)
                    splits[split_idx]['X_test'] = pd.concat([splits[split_idx]['X_test'], X_test], axis=0, ignore_index=True)
                    splits[split_idx]['y_test'] = pd.concat([splits[split_idx]['y_test'], y_test], axis=0, ignore_index=True)

        return splits
    
    @staticmethod
    def grouped_combinations_split_with_sampling(df, n_splits=N_SPLITS, sample_size=SAMPLE_SIZE, seed=SEED):
        """
        Generate a random sample of combinations of 'm_id' groups for train-test splits.

        Args:
            df (pd.DataFrame): DataFrame containing all instances, including 'm_id' and 'group_name'.
            n_splits (int): Number of splits (used to calculate test size proportion).
            sample_size (int): Number of random combinations to sample.
            seed (int): Random seed for reproducibility.

        Returns:
            splits (list): List of dictionaries, each representing one train-test split combination.
        """
        random.seed(seed)  # Ensure reproducibility
        splits = {}
        
        # Get all unique mouse IDs
        unique_mice = df['m_id'].unique()

        # Compute the size of the test group based on n_splits
        test_size = len(unique_mice) // n_splits

        # Generate all possible combinations of 'm_id' for the test set
        test_combinations = list(combinations(unique_mice, test_size))
        print(f"Total possible combinations: {len(test_combinations)}")

        # Randomly sample a subset of the combinations
        sampled_combinations = random.sample(test_combinations, k=sample_size)

        # Create splits for the sampled combinations
        for i, test_group in enumerate(sampled_combinations):
            # Test mice are those in the current combination
            test_mice = set(test_group)
            
            # Train mice are the remaining mice
            train_mice = set(unique_mice) - test_mice

            # Filter the DataFrame for train and test data
            train_data = df[df['m_id'].isin(train_mice)].reset_index(drop=True)
            test_data = df[df['m_id'].isin(test_mice)].reset_index(drop=True)

            # Prepare X (features) and y (labels) for train and test sets
            X_train = train_data.drop(columns=['group_name', 'm_id']).reset_index(drop=True)
            y_train = train_data['group_name'].reset_index(drop=True)

            X_test = test_data.drop(columns=['group_name', 'm_id']).reset_index(drop=True)
            y_test = test_data['group_name'].reset_index(drop=True)

            # Append this combination to the splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }

        return splits
