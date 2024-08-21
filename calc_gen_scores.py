import json

import evaluate


def calc_acc(list_preds, list_targets):
    num_corrects = 0
    for pred, target in zip(list_preds, list_targets):
        # print(pred, target)
        if pred == target or pred in target:
            num_corrects += 1

    acc = num_corrects / len(list_preds)
    return acc


def calc_mt_scores(list_preds, list_targets, metric_name="nist_mt"):
    scorer = evaluate.load(metric_name)

    results = scorer.compute(predictions=list_preds, references=list_targets)
    print("results: ", results)

    if metric_name == "rouge":
        score = results.get("rougeL")
    else:
        score = results.get(metric_name)

    return score


def calc_scores(pred_file_path):

    list_preds, list_targets = read_prediction_file(pred_file_path)

    # acc
    acc = calc_acc(list_preds, list_targets)
    print("acc: ", acc)

    # BLEU:
    bleu_score = calc_mt_scores(list_preds, list_targets, metric_name="bleu")
    print("bleu_score: ", bleu_score)

    # nist_mt:
    nist_score = calc_mt_scores(list_preds, list_targets, metric_name="nist_mt")

    # meteor:
    meteor_score = calc_mt_scores(list_preds, list_targets, metric_name="meteor")

    # rouge:
    rougeL_score = calc_mt_scores(list_preds, list_targets, metric_name="rouge")

    # rouge:
    # CIDEr_score = calc_mt_scores(list_preds, list_targets, metric_name="CIDEr")

    return {
        "acc": acc,
        "bleu": bleu_score,
        "nist": nist_score,
        "meteor": meteor_score,
        "rougeL": rougeL_score,
        # "CIDEr": CIDEr_score,
    }


def read_prediction_file(pred_file_path):
    list_samples = []
    with open(pred_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = json.loads(line)
            list_samples.append(line)

    list_preds = []
    list_targets = []
    for samp in list_samples:
        pred = samp["pred"]
        target = samp["target"]
        list_preds.append(pred)
        list_targets.append([target])

    return list_preds, list_targets


if __name__ == "__main__":

    # pred_file_path = "experiments/llama2_7b_e2e_1/test_predictions_bak1.json"
    pred_file_path = "experiments/llama2_7b_e2e_2/test_predictions_2.json"
    pred_file_path = "resources/Llama-2-7b-hf/test_predictions.json"
    score = calc_scores(pred_file_path)
    print("score: ", score)

