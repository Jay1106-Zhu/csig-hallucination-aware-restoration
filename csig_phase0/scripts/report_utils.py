import numpy as np


def select_failures(table, count=4, patch_size=256):
    selected = []
    for index, row in table[table.label == 'BAD'].sort_values('gain', kind='stable').iterrows():
        if all(abs(row.x - table.loc[other].x) >= patch_size or abs(row.y - table.loc[other].y) >= patch_size for other in selected):
            selected.append(index)
        if len(selected) == count:
            break
    return table.loc[selected].copy()


def spatial_statistics(table):
    grid = table.pivot(index='y', columns='x', values='label').sort_index().sort_index(axis=1)
    bad = (grid.to_numpy() == 'BAD').astype(np.float32)
    agreements = np.concatenate([(bad[:, 1:] == bad[:, :-1]).ravel(), (bad[1:, :] == bad[:-1, :]).ravel()])
    fraction = float((table.label == 'BAD').mean())
    expected = fraction ** 2 + (1 - fraction) ** 2
    return {'patch_count': len(table), 'good_count': int((table.label == 'GOOD').sum()),
            'bad_count': int((table.label == 'BAD').sum()), 'bad_fraction': fraction,
            'mean_patch_gain': float(table.gain.mean()), 'median_patch_gain': float(table.gain.median()),
            'adjacent_pairs': len(agreements), 'adjacent_same_label': float(agreements.mean()),
            'independent_expected_agreement': expected, 'agreement_excess': float(agreements.mean()) - expected}
