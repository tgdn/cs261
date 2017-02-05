import graphlab as _graphlab
from graphlab.toolkits._internal_utils import _raise_error_if_not_sframe
from graphlab.toolkits._internal_utils import _toolkits_select_columns


def create(dataset, target, model_name, env, features=None,
           validation_set='auto', verbose=True, **kwargs):

    _raise_error_if_not_sframe(dataset, "training dataset")

    # Create a validation set
    if isinstance(validation_set, str):
        if validation_set == 'auto':
            if dataset.num_rows() >= 100:
                if verbose:
                    print ("PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.\n"
                           "          You can set ``validation_set=None`` to disable validation tracking.\n")
                dataset, validation_set = dataset.random_split(.95)
            else:
                validation_set = None
        else:
            raise TypeError('Unrecognized value for validation_set.')

    # Target
    target_sframe = _toolkits_select_columns(dataset, [target])

    # Features
    if features is None:
        features = dataset.column_names()
        features.remove(target)
    if not hasattr(features, '__iter__'):
        raise TypeError("Input 'features' must be a list.")
    if not all([isinstance(x, str) for x in features]):
        raise TypeError("Invalid feature %s: Feature names must be of type str" % x)
    features_sframe = _toolkits_select_columns(dataset, features)

    options = {}
    _kwargs = {}
    for k in kwargs:
        _kwargs[k.lower()] = kwargs[k]
    options.update(_kwargs)
    options.update({'target': target_sframe,
                    'features': features_sframe,
                    'model_name': model_name})

    if validation_set is not None:

        if not isinstance(validation_set, _graphlab.SFrame):
            raise TypeError("validation_set must be either 'auto' or an SFrame matching the training data.")

        # Attempt to append the two datasets together to check schema
        validation_set.head().append(dataset.head())

        options.update({
            'features_validation': _toolkits_select_columns(validation_set, features),
            'target_validation': _toolkits_select_columns(validation_set, [target])})

    from . import _dml
    dml_obj = _dml.run("distributed_supervised_train", model_name, options, env)

    return dml_obj
