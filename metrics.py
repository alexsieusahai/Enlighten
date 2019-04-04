def negative_accuracy(preds, actual):
    return -sum([preds[i] == actual[i] for i in range(len(preds))]) / len(preds)
