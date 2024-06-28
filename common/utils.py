import pandas as pd


def get_train_validation_test(
        df: pd.DataFrame, train: float, validation: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled_df: pd.DataFrame = df.sample(frac=1, random_state=1)

    size: int = len(shuffled_df)
    train_size: int = int(train * size)
    validation_size: int = int(validation * size)
    train: pd.DataFrame = shuffled_df[0:train_size]
    validation: pd.DataFrame = shuffled_df[train_size: train_size + validation_size]
    test: pd.DataFrame = shuffled_df[train_size + validation_size:]

    return train, validation, test
