import numpy as np

def aggregate_rows_per_encounter(similarity, encounters_id):
    if similarity.shape[0] != len(encounters_id):
        raise ValueError("Length of encounters_id must match number of similarity rows.")

    unique_encounters = np.unique(encounters_id)
    aggregated_rows = []
    for eid in unique_encounters:
        mask = encounters_id == eid
        group_rows = similarity[mask]
        aggregated_rows.append(np.max(group_rows, axis=0))
    return np.vstack(aggregated_rows), unique_encounters
