from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from itertools import combinations
import random

N_SPLITS = 4
TEST_SIZE = 0.6
SEED = 42
SAMPLE_SIZE = 10
CURED_GROUPS = [1276, 1285, 1281, 1284, 1382]

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
            X_train = df.iloc[train_indices].drop(columns=['group_name']).reset_index(drop=True)
            y_train = df.iloc[train_indices]['group_name'].reset_index(drop=True)

            # Test data
            X_test = df.iloc[test_indices].drop(columns=['group_name']).reset_index(drop=True)
            y_test = df.iloc[test_indices]['group_name'].reset_index(drop=True)

            # Store splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
        
        return 'expanding_window_split', splits
    
    """Class to split data into training and testing sets."""
    @staticmethod
    def expanding_window_split_test(df: pd.DataFrame, n_splits: int = N_SPLITS) -> dict:
        """
        Perform Expanding Window Split, where test is all instances after training.

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
        for i, (train_indices, _) in enumerate(tscv.split(df)):
            
            # Prepare training, and testing sets
            X_train = df.iloc[train_indices].drop(columns=['group_name']).reset_index(drop=True)
            y_train = df.iloc[train_indices]['group_name'].reset_index(drop=True)

            # Test data
            X_test = df.iloc[train_indices[-1]:].drop(columns=['group_name']).reset_index(drop=True)
            y_test = df.iloc[train_indices[-1]:]['group_name'].reset_index(drop=True)

            # Store splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }            
        
        return 'expanding_window_split', splits
    
    @staticmethod
    def rolling_window_split(df: pd.DataFrame, n_splits: int, train_pct: float, test_pct: float) -> dict:
        """
        Perform Rolling Window Split using percentages for train and test sizes.

        Args:
            df (pd.DataFrame): DataFrame containing all instances, sorted by time.
            n_splits (int): Number of splits for rolling windows.
            train_pct (float): Percentage of the dataset to use for training (e.g., 0.7 for 70%).
            test_pct (float): Percentage of the dataset to use for testing (e.g., 0.3 for 30%).

        Returns:
            splits (dict): Dictionary with train-test splits for each fold.
        """
        splits = {}

        # Sort the DataFrame by 'day_of_study' to ensure chronological order
        df = df.sort_values(by='day_of_study').reset_index(drop=True)

        total_samples = len(df)
        train_size = int(total_samples * train_pct)
        test_size = int(total_samples * test_pct)
        step_size = (total_samples - train_size - test_size) // (n_splits - 1)

        if train_size + test_size > total_samples:
            raise ValueError("Train and test percentages exceed total data size.")

        if step_size <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for the dataset size={total_samples}. "
                f"Reduce n_splits or adjust train_pct and test_pct."
            )

        for i in range(n_splits):
            start_index = i * step_size
            train_indices = list(range(start_index, start_index + train_size))
            test_indices = list(range(start_index + train_size, start_index + train_size + test_size))

            # Ensure indices do not exceed dataset bounds
            train_indices = [idx for idx in train_indices if idx < total_samples]
            test_indices = [idx for idx in test_indices if idx < total_samples]

            # Prepare training and testing sets
            X_train = df.iloc[train_indices].drop(columns=['group_name']).reset_index(drop=True)
            y_train = df.iloc[train_indices]['group_name'].reset_index(drop=True)

            X_test = df.iloc[test_indices].drop(columns=['group_name']).reset_index(drop=True)
            y_test = df.iloc[test_indices]['group_name'].reset_index(drop=True)

            # Print the indices for each split
            print(f"Split {i}: Train={train_indices}, Test={test_indices}")

            # Store splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }

        return 'rolling_window_split', splits

    
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
            # Randomly select test groups, ensuring at least one cured group is included
            test_ids = np.random.choice(group_ids, size=test_groups_count, replace=False)
            while not any([group in test_ids for group in CURED_GROUPS]):
                test_ids = np.random.choice(group_ids, size=test_groups_count, replace=False)

            # Separate data into test, and training groups
            test_group = df[df['m_id'].isin(test_ids)].sort_values(by='day_of_study').reset_index(drop=True)
            train_group = df[~df['m_id'].isin(test_ids)].sort_values(by='day_of_study').reset_index(drop=True)

            # Define train and test indices for the test group
            train_indices = list(range(0, int(len(test_group) * 0.60)))  # First 60% for training
            test_indices = list(range(int(len(test_group) * 0.60), len(test_group)))  # Remaining 40% for testing

            # Prepare training, and testing sets
            X_train = pd.concat([train_group.drop(columns=['group_name']),
                                test_group.iloc[train_indices].drop(columns=['group_name'])]).reset_index(drop=True)
            y_train = pd.concat([train_group['group_name'], test_group.iloc[train_indices]['group_name']]).reset_index(drop=True)

            X_test = test_group.iloc[test_indices].drop(columns=['group_name']).reset_index(drop=True)
            y_test = test_group.iloc[test_indices]['group_name'].reset_index(drop=True)

            # Store splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }

        return 'split_data_population_informed', splits
    
    def split_population_informed_combinations(df: pd.DataFrame, test_size=0.2, sample_size=None, seed=42):
        """
        Create train-test splits based on population-informed combinations of unique `m_id`s.

        Args:
            df (pd.DataFrame): DataFrame containing all instances, including 'm_id' and 'group_name'.
            test_size (float): Proportion of unique `m_id`s to include in the test set (e.g., 0.2).
            sample_size (int or None): Number of random combinations to sample. If None, use all combinations.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple: ('split_population_informed_combinations', dict)
                dict: A dictionary where each key is a fold index and value is a dict containing:
                    - X_train, y_train: Training data and labels.
                    - X_test, y_test: Testing data and labels.
        """
        random.seed(seed)  # Ensure reproducibility
        splits = {}

        # Get all unique mouse IDs
        unique_mice = df['m_id'].unique()

        # Calculate the number of mice in the test set
        test_groups_count = max(1, int(len(unique_mice) * test_size))

        # Generate all possible test set combinations
        test_combinations = list(combinations(unique_mice, test_groups_count))
        print(f"Total possible combinations: {len(test_combinations)}")

        # If sample_size is specified, randomly sample combinations; otherwise, use all
        if sample_size is not None:
            sampled_combinations = random.sample(test_combinations, k=min(sample_size, len(test_combinations)))
        else:
            sampled_combinations = test_combinations

        # Remove combinations without at least one cured group
        sampled_combinations = [comb for comb in sampled_combinations if any([group in comb for group in CURED_GROUPS])]

        print(f"Number of sampled combinations: {len(sampled_combinations)}")

        # Iterate over sampled combinations to create splits
        for i, test_group in enumerate(sampled_combinations):

            test_mice = set(test_group)

            # Train mice are the remaining mice
            train_mice = set(unique_mice) - test_mice

            # Filter the DataFrame for train and test data
            train_data = df[df['m_id'].isin(train_mice)].reset_index(drop=True)
            test_data = df[df['m_id'].isin(test_mice)].reset_index(drop=True)

            # Prepare X (features) and y (labels) for train and test sets
            X_train = train_data.drop(columns=['group_name']).reset_index(drop=True)
            y_train = train_data['group_name'].reset_index(drop=True)

            X_test = test_data.drop(columns=['group_name']).reset_index(drop=True)
            y_test = test_data['group_name'].reset_index(drop=True)

            # Append this combination to the splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }

        # Return split method name, and the splits
        return 'split_population_informed_combinations', splits

    
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
                    X_train, y_train = group.drop(columns=['group_name']), group['group_name']
                    splits[split_idx] = {
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': None,
                        'y_test': None
                    }
                else:
                    if splits[split_idx]['X_train'] is None and splits[split_idx]['y_train'] is None:
                        X_train, y_train = group.drop(columns=['group_name']), group['group_name']
                        splits[split_idx]['X_train'] = X_train
                        splits[split_idx]['y_train'] = y_train
                    else:
                        X_train, y_train = group.drop(columns=['group_name']), group['group_name']
                        splits[split_idx]['X_train'] = pd.concat([splits[split_idx]['X_train'], X_train], axis=0, ignore_index=True)
                        splits[split_idx]['y_train'] = pd.concat([splits[split_idx]['y_train'], y_train], axis=0, ignore_index=True)
                continue

            tscv = TimeSeriesSplit(n_splits=n_splits + 1).split(group) 
            next(tscv)  # Skip the first split

            for split_idx, (train_cv_indices, test_indices) in enumerate(tscv):
                X_train, y_train = group.iloc[train_cv_indices].drop(columns=['group_name']), group.iloc[train_cv_indices]['group_name']
                X_test, y_test = group.iloc[test_indices].drop(columns=['group_name']), group.iloc[test_indices]['group_name']
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

        return 'expanding_window_split_by_mice_group', splits
    
    @staticmethod
    def grouped_combinations_split_with_sampling(df, n_splits=N_SPLITS, sample_size=None, seed=SEED):
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

        if sample_size is not None:
            # Randomly sample a subset of the combinations
            sampled_combinations = random.sample(test_combinations, k=min(sample_size, len(test_combinations)))
        else:
            sampled_combinations = test_combinations
        
        # Remove combinations without at least one cured group
        sampled_combinations = [comb for comb in sampled_combinations if any([group in comb for group in CURED_GROUPS])]

        # Create splits for the sampled combinations
        for i, test_group in enumerate(sampled_combinations):
            # Test mice are those in the current combination
            test_mice = set(test_group)
            print(test_mice)
            
            # Train mice are the remaining mice
            train_mice = set(unique_mice) - test_mice

            # Filter the DataFrame for train and test data
            train_data = df[df['m_id'].isin(train_mice)].reset_index(drop=True)
            test_data = df[df['m_id'].isin(test_mice)].reset_index(drop=True)

            # Prepare X (features) and y (labels) for train and test sets
            X_train = train_data.drop(columns=['group_name']).reset_index(drop=True)
            y_train = train_data['group_name'].reset_index(drop=True)

            X_test = test_data.drop(columns=['group_name']).reset_index(drop=True)
            y_test = test_data['group_name'].reset_index(drop=True)

            # Append this combination to the splits
            splits[i] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }

        # Return split method name, and the splits
        return 'grouped_combinations_split_with_sampling', splits
        
