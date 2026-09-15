import numpy as np


def validate_fold_checkpoint(payload, training_features, training_images, heldout_image):
    if set(payload['training_images']) != set(training_images) or heldout_image in payload['training_images']:
        raise ValueError('Heldout image leaked into training')
    if payload['heldout_image'] != heldout_image:
        raise ValueError('Wrong heldout image')
    values = np.asarray(training_features, dtype=np.float64)
    expected_mean = values.mean(axis=0).astype(np.float32)
    expected_scale = values.std(axis=0)
    expected_scale[expected_scale < 1e-8] = 1.0
    if not np.allclose(payload['mean'].numpy(), expected_mean, atol=1e-7, rtol=1e-6):
        raise ValueError('Scaler mean does not match training-only features')
    if not np.allclose(payload['scale'].numpy(), expected_scale.astype(np.float32), atol=1e-7, rtol=1e-6):
        raise ValueError('Scaler scale does not match training-only features')
