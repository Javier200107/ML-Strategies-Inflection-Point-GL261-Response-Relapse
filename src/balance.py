import pandas as pd
from sklearn.utils import resample

def split_data(df, test_size=0.2):
    # Create lists to store the rows for training and test sets
    train_rows = []
    test_rows = []

    # Iterate over each mouse (m_id) and perform time-based splitting
    for _, group in df.groupby('m_id'):
        # Sort by 'day_of_study' within each mouse to ensure temporal order
        group = group.sort_values(by='day_of_study')
        
        # Split the mouse data into training and test based on test_size
        split_point = int(len(group) * (1 - test_size))
        train_rows.append(group.iloc[:split_point])  # Initial part for training
        test_rows.append(group.iloc[split_point:])   # Final part for testing

    # Concatenate the results into DataFrames
    train_df = pd.concat(train_rows).drop(columns='m_id')
    test_df = pd.concat(test_rows).drop(columns='m_id')

    # Return X_train, X_test, y_train, y_test
    X_train, X_test = train_df.drop(columns='group_name'), test_df.drop(columns='group_name')
    y_train, y_test = train_df['group_name'], test_df['group_name']

    return X_train, X_test, y_train, y_test

from sklearn.model_selection import TimeSeriesSplit

def split_data_nested_cv_ts(df, n_splits=3, test_size=0.2):
    '''
    Splits data into training, validation (cross-validation), and testing sets for each mouse (m_id) in temporal order.

    Args:
        df: DataFrame containing the data.
        n_splits: Number of train/validation folds for TimeSeriesSplit.
        test_size: Fraction of data to be used for the testing set.

    Returns:
        splits: A list of dictionaries with train, validation, and test splits for each mouse.
    '''
    # Initialize a list to store training, validation, and test sets for each mouse
    splits = []
    
    # Iterate over each mouse group based on `m_id`
    for mouse_id, group in df.groupby('m_id'):
        # Sort each mouse's data by `day_of_study` to ensure temporal order
        group = group.sort_values(by='day_of_study').reset_index(drop=True)
        
        # Define the size of the test set
        test_len = int(len(group) * test_size)
        
        # Initialize TimeSeriesSplit to create indices for training and validation
        tscv = TimeSeriesSplit(n_splits=n_splits + 1)  # Adding 1 split since we'll skip the first

        # Skip the first split. This is done to ensure the first split isn’t too small, which can lead to unstable training
        train_cv_splits = list(tscv.split(group.iloc[:-test_len])) 
        train_cv_splits = train_cv_splits[1:]  # Remove the first split to avoid very small train set

        # Define the test set as the last `test_len` samples
        X_test, y_test = group.iloc[-test_len:].drop(columns=['m_id', 'group_name']), group.iloc[-test_len:]['group_name']

        # Save the splits in the final list for each mouse
        for train_indices, cv_indices in train_cv_splits:
            # Create training and validation sets based on the indices from TimeSeriesSplit
            X_train, y_train = group.iloc[train_indices].drop(columns=['m_id', 'group_name']), group.iloc[train_indices]['group_name']
            X_cv, y_cv = group.iloc[cv_indices].drop(columns=['m_id', 'group_name']), group.iloc[cv_indices]['group_name']
            
            # Store the split as a dictionary for this mouse
            splits.append({
                'mouse_id': mouse_id,
                'X_train': X_train,
                'y_train': y_train,
                'X_cv': X_cv,
                'y_cv': y_cv,
                'X_test': X_test,
                'y_test': y_test
            })

            # Print date ranges for each split as a verification step
            print(f"Mouse {mouse_id}:")
            print(f"  Train: {X_train['day_of_study'].min()} -- {X_train['day_of_study'].max()}")
            print(f"Train number of samples: {len(X_train)}")
            print(f"  CV: {X_cv['day_of_study'].min()} -- {X_cv['day_of_study'].max()}")
            print(f"CV number of samples: {len(X_cv)}")
            print(f"  Test: {X_test['day_of_study'].min()} -- {X_test['day_of_study'].max()}")
            print(f"Test number of samples: {len(X_test)}")
            print("=" * 50)

    return splits


def drop_mice_group(df, m_group):
    '''
    Drop the rows with the specified mice group

    Args:
        m_group: Mice group to drop from the dataset
    '''
    return df[df["group_name"] != m_group]

def balance_classes_by_undersampling_features_and_target(X, Y, target_col='group_name'):
    '''
    Balance the classes in the dataset by undersampling the majority class
    
    Args:
        X: Features DataFrame
        Y: Target DataFrame
        target_col: Name of the target column
        
    Returns:
        X_balanced: Balanced Features DataFrame
        Y_balanced: Balanced Target DataFrame
    '''
    # Concatenate the features and target
    df = pd.concat([X, Y], axis=1)
    print("Original class distribution:\n", df[target_col].value_counts())
    
    # Find the class with the minimum number of samples
    min_class_count = df[target_col].value_counts().min()

    # Create a balanced DataFrame by undersampling
    df_balanced = df.groupby(target_col).apply(lambda x: x.sample(min_class_count, random_state=42)).reset_index(drop=True)

    # Print the results
    print(f"Balanced classes by undersampling. Number of samples per class: {min_class_count}")
    print("Balanced class distribution:\n", df_balanced[target_col].value_counts())

    # Split the features and target
    X_balanced = df_balanced.drop(columns=[target_col])
    Y_balanced = df_balanced[target_col]

    return X_balanced, Y_balanced

def balance_classes_by_undersampling(df, target_col='group_name'):
    '''
    Balance the classes in the dataset by undersampling the majority class
    
    Args:
        df: DataFrame containing the dataset
        target_col: Name of the target column
        
    Returns:
        df_balanced: Balanced DataFrame
    '''
    print("Original class distribution:\n", df[target_col].value_counts())
    
    # Find the class with the minimum number of samples
    min_class_count = df[target_col].value_counts().min()

    # Create a balanced DataFrame by undersampling
    df_balanced = df.groupby(target_col).apply(lambda x: x.sample(min_class_count, random_state=42)).reset_index(drop=True)

    # Print the results
    print(f"Balanced classes by undersampling. Number of samples per class: {min_class_count}")
    print("Balanced class distribution:\n", df_balanced[target_col].value_counts())

    return df_balanced


def balance_before_merge(X, y, target_group=0, group_1=1, group_2=2):
    '''
    Balance each class individually before merging two groups (relapse and control) into one.
    
    Args:
        X: Features DataFrame
        y: Target Series/DataFrame (containing the group labels)
        target_group: The group that remains unchanged (e.g., cured mice, default: 0)
        group_1: First group to merge (default: 1, relapse)
        group_2: Second group to merge (default: 2, control)
        
    Returns:
        X_balanced: Balanced features DataFrame
        y_balanced: Balanced target Series/DataFrame
    '''

    # Separate the data by group
    X_target = X[y == target_group]  # Group 0 (cured)
    X_group1 = X[y == group_1]       # Group 1 (relapse)
    X_group2 = X[y == group_2]       # Group 2 (control)
    y_target = y[y == target_group]
    y_group1 = y[y == group_1]
    y_group2 = y[y == group_2]

    # Find the minimum class count between the three groups
    min_class_count = min(len(X_target), len(X_group1), len(X_group2))

    # Undersample each group to balance the dataset
    X_target_resampled = resample(X_target, replace=False, n_samples=min_class_count, random_state=42)
    y_target_resampled = resample(y_target, replace=False, n_samples=min_class_count, random_state=42)

    X_group1_resampled = resample(X_group1, replace=False, n_samples=min_class_count, random_state=42)
    y_group1_resampled = resample(y_group1, replace=False, n_samples=min_class_count, random_state=42)

    X_group2_resampled = resample(X_group2, replace=False, n_samples=min_class_count, random_state=42)
    y_group2_resampled = resample(y_group2, replace=False, n_samples=min_class_count, random_state=42)

    # Merge group 1 and group 2 into a single class (death)
    X_merged = pd.concat([X_group1_resampled, X_group2_resampled])
    y_merged = pd.concat([y_group1_resampled, y_group2_resampled]).replace({group_1: 1, group_2: 1})  # Merge into class 1 (death)

    # Concatenate the resampled data
    X_balanced = pd.concat([X_target_resampled, X_merged])
    y_balanced = pd.concat([y_target_resampled, y_merged])

    # Shuffle the data to mix the classes
    X_balanced, y_balanced = resample(X_balanced, y_balanced, random_state=42)

    print(f"Balanced dataset with {min_class_count} samples per class.")
    
    return X_balanced, y_balanced

def oversample_and_merge(X, y, target_group=0, group_1=1, group_2=2):
    '''
    Oversample the minority class (target_group) and merge two groups (group_1 and group_2) into one.
    
    Args:
        X: Features DataFrame
        y: Target Series/DataFrame (containing the group labels)
        target_group: The minority group to oversample (default: 0)
        group_1: First group to merge (default: 1, relapse)
        group_2: Second group to merge (default: 2, control)
        
    Returns:
        X_balanced: Features DataFrame after oversampling and merging
        y_balanced: Target Series/DataFrame after oversampling and merging
    '''
    # Alinear los índices de X e y
    X, y = X.align(y, join='inner', axis=0)

    # Separar los datos por grupo
    X_target = X[y == target_group]  # Grupo 0 (curado)
    X_group1 = X[y == group_1]       # Grupo 1 (recaída)
    X_group2 = X[y == group_2]       # Grupo 2 (control)
    y_target = y[y == target_group]
    y_group1 = y[y == group_1]
    y_group2 = y[y == group_2]

    # Encontrar el tamaño máximo de la clase para hacer oversampling
    max_class_count = max(len(X_target), len(X_group1), len(X_group2))
    second_max_class_count = sorted([len(X_target), len(X_group1), len(X_group2)])[-2]

    # Hacer oversampling de la clase minoritaria (grupo 0)
    X_target_oversampled = resample(X_target, 
                                     replace=True,          # Muestreo con reemplazo
                                     n_samples=max_class_count + second_max_class_count,  # Emparejar con el máximo tamaño
                                     random_state=42)       # Reproducibilidad
    y_target_oversampled = resample(y_target, 
                                     replace=True, 
                                     n_samples=max_class_count + second_max_class_count, 
                                     random_state=42)

    # Combinar grupo 1 y grupo 2 en una sola clase (muerte)
    X_merged = pd.concat([X_group1, X_group2])
    y_merged = pd.concat([y_group1, y_group2]).replace({group_1: 1, group_2: 1})  # Fusionar en clase 1 (muerte)

    # Concatenar los datos re-muestreados
    X_balanced = pd.concat([X_target_oversampled, X_merged])
    y_balanced = pd.concat([y_target_oversampled, y_merged])

    # Mezclar los datos para barajar las clases
    X_balanced, y_balanced = resample(X_balanced, y_balanced, n_samples=len(X_balanced), replace=False, random_state=42)
    
    return X_balanced, y_balanced


def ensemble_sample_and_merge(X, y, target_group=0, group_1=1, group_2=2):
    '''
    Use an ensemble of samplers to balance the classes and merge two groups.
    
    Args:
        X: Features DataFrame
        y: Target Series/DataFrame (containing the group labels)
        target_group: The minority group to oversample (default: 0)
        group_1: First group to merge (default: 1, relapse)
        group_2: Second group to merge (default: 2, control)
        
    Returns:
        X_balanced: Features DataFrame after sampling and merging
        y_balanced: Target Series/DataFrame after sampling and merging
    '''
    # Separate the data by group
    X_target = X[y == target_group]  # Group 0 (cured)
    X_group1 = X[y == group_1]       # Group 1 (relapse)
    X_group2 = X[y == group_2]       # Group 2 (control)
    y_target = y[y == target_group]
    y_group1 = y[y == group_1]
    y_group2 = y[y == group_2]

    # Merge group 1 and group 2 into a single class (death)
    X_merged = pd.concat([X_group1, X_group2])
    y_merged = pd.concat([y_group1, y_group2]).replace({group_1: 1, group_2: 1})  # Merge into class 1 (death)

    # Create the ensemble sampler
    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)

    # Fit and transform the data
    X_combined = pd.concat([X_target, X_merged])
    y_combined = pd.concat([y_target, y_merged])
    
    X_balanced, y_balanced = smote_enn.fit_resample(X_combined, y_combined)

    # Shuffle the data to mix the classes
    X_balanced, y_balanced = resample(X_balanced, y_balanced, n_samples=len(X_balanced), replace=False, random_state=42)
    
    return X_balanced, y_balanced

import pandas as pd

def oversample_and_merge_from_array(X, y, target_group=0, group_1=1, group_2=2):
    # Convert X (PCA-transformed data, NumPy array) to a Pandas DataFrame
    X = pd.DataFrame(X)
    
    # If y is a NumPy array, convert it to a Pandas Series with the same index as X
    if not isinstance(y, pd.Series):
        y = pd.Series(y).reset_index(drop=True)
    
    # Ensure the index of X matches the index of y
    X.index = y.index
    
    # Oversample the target group (e.g., group_name == 0)
    X_target = X[y == target_group]
    y_target = y[y == target_group]
    
    max_class_count = y.value_counts().max()
    second_max_class_count = y.value_counts().sort_values(ascending=False).iloc[1]

    # Oversample target group
    X_target_oversampled = resample(X_target, 
                                    replace=True, 
                                    n_samples=max_class_count + second_max_class_count, 
                                    random_state=42)
    y_target_oversampled = resample(y_target, 
                                    replace=True, 
                                    n_samples=max_class_count + second_max_class_count, 
                                    random_state=42)

    # Select the instances for groups 1 and 2
    X_group1 = X[y == group_1]
    X_group2 = X[y == group_2]
    y_group1 = y[y == group_1]
    y_group2 = y[y == group_2]

    # Merge group 1 and group 2 into a single class (e.g., class 1)
    X_merged = pd.concat([X_group1, X_group2])
    y_merged = pd.concat([y_group1, y_group2]).replace({group_1: 1, group_2: 1})  # Merge into class 1

    # Concatenate the oversampled target group with the merged group
    X_balanced = pd.concat([X_merged, X_target_oversampled])
    y_balanced = pd.concat([y_merged, y_target_oversampled])

    return X_balanced, y_balanced


from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def oversample_with_smote_and_merge(X, y, target_group=0, group_1=1, group_2=2, smote_strategy='auto'):
    '''
    Apply SMOTE to oversample the minority class (target_group) and merge two groups (group_1 and group_2) into one.
    
    Args:
        X: Features DataFrame
        y: Target Series/DataFrame (containing the group labels)
        target_group: The minority group to oversample (default: 0)
        group_1: First group to merge (default: 1, relapse)
        group_2: Second group to merge (default: 2, control)
        smote_strategy: SMOTE oversampling strategy. Default is 'auto' (all classes are balanced).
        
    Returns:
        X_balanced: Features DataFrame after oversampling and merging
        y_balanced: Target Series/DataFrame after oversampling and merging
    '''
    # Merge group_1 and group_2 into a single class (death)
    y_merged = y.replace({group_1: 1, group_2: 1})  # Merge into class 1 (death)

    # Apply SMOTE to balance the data
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y_merged)

    # Split data back into the balanced classes
    return X_balanced, y_balanced

