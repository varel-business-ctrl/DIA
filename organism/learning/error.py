def prediction_error(predicted, actual):
    if predicted is None:
        return None

    differences = {}

    keys = set(predicted) | set(actual)

    for key in keys:
        if predicted.get(key) != actual.get(key):
            differences[key] = {
                "predicted": predicted.get(key),
                "actual": actual.get(key),
            }

    return {
        "total_differences": len(differences),
        "differences": differences,
    }
