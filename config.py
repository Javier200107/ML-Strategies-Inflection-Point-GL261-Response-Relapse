import os

# Base Input Directory
BASE_INPUT_DIR = "input"

# Feature-related paths
FEATURES_DIR = os.path.join(BASE_INPUT_DIR, "features")
FEATURES_FILE_PATH = os.path.join(FEATURES_DIR, "radiomics_features.csv")

# Dataset-related paths
DATASET_DIR = os.path.join(BASE_INPUT_DIR, "dataset")

# TFrecord-related paths
TFRECORD_DIR = os.path.join(BASE_INPUT_DIR, "full_ds.tfrecord")

RADIOMICS_FEATURES = {
    "group_name", "day_of_study", "10Percentile", "90Percentile",
    "Energy", "Entropy", "InterquartileRange", "Kurtosis", "Maximum",
    "MeanAbsoluteDeviation", "Mean", "Median", "Minimum", "Range",
    "RobustMeanAbsoluteDeviation", "RootMeanSquared", "Skewness",
    "TotalEnergy", "Uniformity", "Variance", "Elongation",
    "MajorAxisLength", "MaximumDiameter", "MeshSurface", "MinorAxisLength",
    "Perimeter", "PerimeterSurfaceRatio", "PixelSurface", "Sphericity",
    "Autocorrelation", "ClusterProminence", "ClusterShade", "ClusterTendency",
    "Contrast", "Correlation", "DifferenceAverage", "DifferenceEntropy",
    "DifferenceVariance", "Id", "Idm", "Idmn", "Idn", "Imc1", "Imc2",
    "InverseVariance", "JointAverage", "JointEnergy", "JointEntropy", "MCC",
    "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares",
    "GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized",
    "GrayLevelVariance", "HighGrayLevelRunEmphasis", "LongRunEmphasis",
    "LongRunHighGrayLevelEmphasis", "LongRunLowGrayLevelEmphasis",
    "LowGrayLevelRunEmphasis", "RunEntropy", "RunLengthNonUniformity",
    "RunLengthNonUniformityNormalized", "RunPercentage", "RunVariance",
    "ShortRunEmphasis", "ShortRunHighGrayLevelEmphasis",
    "ShortRunLowGrayLevelEmphasis", "HighGrayLevelZoneEmphasis",
    "LargeAreaEmphasis", "LargeAreaHighGrayLevelEmphasis",
    "LargeAreaLowGrayLevelEmphasis", "LowGrayLevelZoneEmphasis",
    "SizeZoneNonUniformity", "SizeZoneNonUniformityNormalized",
    "SmallAreaEmphasis", "SmallAreaHighGrayLevelEmphasis",
    "SmallAreaLowGrayLevelEmphasis", "ZoneEntropy", "ZonePercentage",
    "ZoneVariance", "Busyness", "Coarseness", "Complexity", "Strength",
    "DependenceEntropy", "DependenceNonUniformity",
    "DependenceNonUniformityNormalized", "DependenceVariance",
    "HighGrayLevelEmphasis", "LargeDependenceEmphasis",
    "LargeDependenceHighGrayLevelEmphasis", "LargeDependenceLowGrayLevelEmphasis",
    "LowGrayLevelEmphasis", "SmallDependenceEmphasis",
    "SmallDependenceHighGrayLevelEmphasis", "SmallDependenceLowGrayLevelEmphasis"
}