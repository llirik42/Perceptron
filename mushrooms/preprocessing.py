import pandas as pd
from sklearn.preprocessing import LabelEncoder

from common import calculate_gain_ratio


__all__ = ['preprocess']


def preprocess(df: pd.DataFrame,
               original_target: str,
               destination_target: str,
               features_count: int = 10) -> pd.DataFrame:
    features: list[str] = []
    features_to_drop: list[str] = []
    for c in df.columns:
        unique_values_count: int = df[c].nunique()
        if unique_values_count == 1:
            features_to_drop.append(c)
        elif c != original_target:
            features.append(c)

    result: pd.DataFrame = pd.get_dummies(df, columns=features, dtype=float)

    target_not_encoded_values = result[original_target].values
    result.drop(original_target, axis=1, inplace=True)
    result[destination_target] = LabelEncoder().fit(target_not_encoded_values).transform(target_not_encoded_values)

    for f in features_to_drop:
        result.drop(f, axis=1, inplace=True)

    final_features: list[str] = []
    # +1 -- because we want "FEATURES_COUNT" features + target column
    for column, gain_ratio in calculate_gain_ratio(df=result, target=destination_target)[:features_count + 1]:
        final_features.append(column)

    for f in result.columns:
        if f not in final_features:
            result.drop(f, axis=1, inplace=True)

    return result
