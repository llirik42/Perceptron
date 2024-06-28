import os
from pickle import dump, load

import matplotlib.pyplot as plt
import opendatasets as od
import pandas as pd
from tqdm import tqdm

from common import Layer, get_train_validation_test, LeakyReLu
from model import TemperatureModel
from preprocessing import preprocess

TARGET: str = "Temperature (C)"
TRAIN: float = 0.7
VALIDATION: float = 0.0
EPOCHS: int = 3
LEARNING_RATE: float = 0.0005
PREPROCESSED_DATA_PATH: str = "preprocessed.csv"
MODEL_PATH: str = "model.pkl"


def test(df: pd.DataFrame, test_df: pd.DataFrame, model: TemperatureModel) -> None:
    mean_target: float = df.mean()[TARGET]
    ss_tot: float = 0

    mse: float = 0
    indexes2 = range(len(test_df))
    print("Starting test ...")
    for i in tqdm(indexes2):
        current_row: pd.Series = test_df.iloc[i]
        real_target_value = current_row[TARGET]
        features: pd.Series = current_row.drop(TARGET)
        predicted_target_value: float = model.forward(features)
        mse += (real_target_value - predicted_target_value) ** 2
        ss_tot += (real_target_value - mean_target) ** 2
    print("Finished test!\n")
    ss_res = mse
    mse /= len(test_df)

    r2_score: float = 1 - ss_res / ss_tot

    print(f"MSE: {mse}")
    print(f"R2: {r2_score}")

    for i in range(20):
        current_row: pd.Series = test_df.iloc[i]
        predicted_value: float = model.forward(current_row.drop(TARGET))
        real_value: float = current_row[TARGET]
        print(f"Predicted: {round(predicted_value, 1)}      Real: {round(real_value, 1)}")


def main() -> None:
    if os.path.isfile(PREPROCESSED_DATA_PATH):
        print("Reading preprocessed data ...")
        df = pd.read_csv(PREPROCESSED_DATA_PATH)
        print("Preprocessed is read!\n")
    else:
        od.download("https://www.kaggle.com/datasets/budincsevity/szeged-weather")
        df: pd.DataFrame = pd.read_csv("szeged-weather/weatherHistory.csv")
        print("Starting preprocessing ...")
        preprocess(df)
        print("Preprocessing finished!\n")
        df.to_csv(PREPROCESSED_DATA_PATH, index=False)

    train_df, validation_df, test_df = get_train_validation_test(
        df=df,
        train=TRAIN,
        validation=VALIDATION,
    )

    if os.path.isfile(MODEL_PATH):
        print(f'Loading model from "{MODEL_PATH}" ...')
        with open(MODEL_PATH, "rb") as model_file:
            model: TemperatureModel = load(model_file)
            print("Model is loaded!\n")
    else:
        model = TemperatureModel(
            layers=[Layer(input_size=df.columns.size - 1, output_size=1, activation=LeakyReLu())],
            df=df,
            target=TARGET,
        )

        # Train
        print("Starting training ...")
        indexes1 = range(len(train_df))
        train_errors: list[float] = []
        for j in range(EPOCHS):
            mse: float = 0

            for i in tqdm(indexes1):
                current_row: pd.Series = train_df.iloc[i]
                real_target_value = current_row[TARGET]
                features: pd.Series = current_row.drop(TARGET)
                predicted_target_value: float = model.forward(features)
                error: float = predicted_target_value - real_target_value
                mse += error * error
                train_errors.append(error)
                model.backward(error=error, learning_rate=LEARNING_RATE)

            model.push_train_error(mse / len(train_df))
            print(f"EPOCH: {j + 1} finished, MSE: {mse / len(train_df)}")
        print("Training finished!\n")
        plt.plot(range(len(train_errors)), train_errors)
        plt.show()

        print(f'Dumping model to "{MODEL_PATH}" ...')
        with open(MODEL_PATH, "wb") as model_file:
            dump(model, model_file)
            print("Model is dumped!\n")

    test(
        df=df,
        test_df=test_df,
        model=model,
    )


if __name__ == "__main__":
    main()
