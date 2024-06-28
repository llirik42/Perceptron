import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import Sigmoid, Layer, get_train_validation_test, get_delta_of_tp_tn_fp_fn
from model import TemperatureModel
from preprocessing import preprocess
from ucimlrepo import fetch_ucirepo

FEATURES_COUNT: int = 10
TRAIN: float = 0.7
VALIDATION: float = 0.0
EPOCHS: int = 10
LEARNING_RATE: float = 0.01
THRESHOLD: float = 0.5
TARGET: str = 'poisonous'


def main() -> None:
    print("Fetching dataset ...")
    mushroom = fetch_ucirepo(id=73)
    print("Dataset fetched!\n")

    features_values: pd.DataFrame = mushroom.data.features
    target_values: pd.DataFrame = mushroom.data.targets
    original_target: str = target_values.columns[0]

    df: pd.DataFrame = features_values.join(target_values)

    print("Starting preprocessing ...")
    preprocessed_df: pd.DataFrame = preprocess(
        df=df,
        original_target=original_target,
        destination_target=TARGET,
        features_count=FEATURES_COUNT
    )
    print("Dataset preprocessed!\n")

    train, validation, test = get_train_validation_test(
        df=preprocessed_df,
        train=TRAIN,
        validation=VALIDATION,
    )

    model: TemperatureModel = TemperatureModel([
        Layer(input_size=preprocessed_df.columns.size - 1, output_size=1, activation=Sigmoid())
    ])

    print("Starting training ...")
    train_errors: list[float] = []
    for j in range(EPOCHS):
        print(f'Epoch {j + 1}')
        for i in tqdm(range(len(train))):
            current_row: pd.Series = train.iloc[i]
            real_target_value: float = current_row[original_target]
            current_features: pd.Series = current_row.drop(original_target)

            predicted_target_value: float = model.forward(current_features)
            error: float = predicted_target_value - real_target_value
            train_errors.append(error)

            model.backward(
                error=error,
                learning_rate=LEARNING_RATE
            )
    print("Training finished!\n")

    print("Starting test ...\n")

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(test)):
        current_row: pd.Series = test.iloc[i]
        real_target_value: float = current_row[TARGET]
        current_features: pd.Series = current_row.drop(TARGET)
        model_output: float = model.forward(current_features)
        model_prediction: float = float(model_output >= THRESHOLD)

        dtp, dtn, dfp, dfn = get_delta_of_tp_tn_fp_fn(model_prediction, real_target_value)
        tp += dtp
        tn += dtn
        fp += dfp
        fn += dfn

    # Доля правильных ответов
    # Не применима при несбалансированных классах
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Доля объектов, названных классификатором положительными и при этом действительно являющимися положительными
    # Применима при несбалансированных классах
    precision = tp / (tp + fp)

    # Какую долю объектов положительного класса из всех объектов положительного класса нашла модель
    # Чувствительность
    # Применима при несбалансированных классах
    recall = tp / (tp + fn)

    # Среднее гармоническое между прецизионностью и полнотой. Умножение на 2 чтобы число было в [0; 1]
    f_score = 2 * (precision * recall) / (precision + recall)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F-SCORE: {f_score}')

    x = []
    y = []

    for j in np.linspace(-0.5, 1.5, 40):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(len(test)):
            current_row: pd.Series = test.iloc[i]
            real_target_value: float = current_row[TARGET]
            current_features: pd.Series = current_row.drop(TARGET)
            model_output: float = model.forward(current_features)
            model_prediction: float = float(model_output >= j)

            dtp, dtn, dfp, dfn = get_delta_of_tp_tn_fp_fn(model_prediction, real_target_value)
            tp += dtp
            tn += dtn
            fp += dfp
            fn += dfn

        # Какую долю объектов положительного класса из всех объектов положительного класса нашел алгоритм
        tpr = recall

        # Какую долю из объектов negative класса алгоритм предсказал неверно
        fpr = fp / (fp + tn)

        x.append(fpr)
        y.append(tpr)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot(x, y, 'r-')
    plt.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
    plt.show()


if __name__ == '__main__':
    main()
