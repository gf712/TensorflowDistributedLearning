from tensorflow.python.client import device_lib
from tensorflow.python.estimator.canned import metric_keys


# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def metric_comparisson(best_eval_result,
                       current_eval_result,
                       key=metric_keys.MetricKeys.LOSS,
                       greater_is_better=True):
    if not best_eval_result or key not in best_eval_result:
        raise ValueError(
            'best_eval_result cannot be empty or no loss is found in it.')

    if not current_eval_result or key not in current_eval_result:
        raise ValueError(
            'current_eval_result cannot be empty or no loss is found in it.')

    if greater_is_better:
        compare = lambda x, y: x > y
    else:
        compare = lambda x, y: x < y

    return compare(best_eval_result[key], current_eval_result[key])