def get_delta_of_tp_tn_fp_fn(predicted_value: float, real_value: float) -> tuple[int, int, int, int]:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    if predicted_value == 1 and real_value == 1:
        tp += 1
    if predicted_value == 0 and real_value == 0:
        tn += 1
    if predicted_value == 1 and real_value == 0:
        fp += 1
    if predicted_value == 0 and real_value == 1:
        fn += 1

    return tp, tn, fp, fn
