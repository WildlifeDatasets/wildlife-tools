import numpy as np
from collections import Counter

def majority_vote(inputs):
    '''
    Majority voting prediction with tie breaks based on ordering in inputs.
    
    Example:
    inputs = np.array([
        ['a', 'a', 'b', 'b', 'd'],
        ['a', 'b', 'c', 'c', 'd'],
        ['a', 'b', 'c', 'd', 'e'],
    ])
    majority_vote(inputs)
    >>> ['a', 'c', 'a']
    '''

    predictions = []
    for row in inputs:
        counter = Counter(row)

        # Majority voting
        best = []
        best_count = 0
        for key, count in counter.most_common():
            best_count = max(count, best_count)    
            if best_count == count:
                best.append(key)
            if best_count > count:
                break

        # Solve ties
        orders = np.array(list(counter.keys()))
        if len(best) == 0:
            best_label = best[0]
        else:
            order_scores = {}
            for i in best:
                order_scores[i] = -np.where(orders==i)[0].item()
            best_label = max(order_scores, key=order_scores.get)
        predictions.append(best_label)
    return predictions
