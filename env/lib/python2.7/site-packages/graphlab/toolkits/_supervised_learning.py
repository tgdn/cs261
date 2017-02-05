##\internal
"""@package graphlab.toolkits
This module defines the (internal) functions used by the supervised_learning_models.
"""
import graphlab as _graphlab

from graphlab.toolkits._model import Model
from graphlab.toolkits._internal_utils import _toolkits_select_columns
from graphlab.toolkits._internal_utils import _raise_error_if_not_sframe
from graphlab.toolkits._internal_utils import _map_unity_proxy_to_object
from graphlab.toolkits._internal_utils import _SGraphFromJsonTree
from graphlab.toolkits._main import ToolkitError

import graphlab.connect as _mt

class SupervisedLearningModel(Model):
    """
    Supervised learning module to predict a target variable as a function of
    several feature variables.
    """
    def __init__(self, model_proxy=None, name=None):
        """__init__(self)"""
        self.__proxy__ = model_proxy
        self.__name__ = name

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the model.
        """
        return self.__class__.__name__

    def __repr__(self):
        """
        Returns a string description of the model, including (where relevant)
        the schema of the training data, description of the training data,
        training statistics, and model hyperparameters.

        Returns
        -------
        out : string
            A description of the model.
        """
        return self.__class__.__name__

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and train the model.

        Returns
        -------
        out : dict
            Dictionary with options used to define and train the model.

        Examples
        --------

        >>> options = m.get_current_options()
        """

        opts = {'model': self.__proxy__,
                'model_name': self.__name__}
        return _graphlab.toolkits._main.run(
                      'supervised_learning_get_current_options', opts)

    def predict(self, dataset, missing_value_action='auto',
                output_type='', options={}, **kwargs):
        """
        Return predictions for ``dataset``, using the trained supervised_learning
        model. Predictions are generated as class labels (0 or
        1).

        Parameters
        ----------
        dataset : SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        missing_value_action: str, optional
            Action to perform when missing values are encountered. This can be
            one of:

            - 'auto': Choose a model dependent missing value policy.
            - 'impute': Proceed with evaluation by filling in the missing
                        values with the mean of the training data. Missing
                        values are also imputed if an entire column of data is
                        missing during evaluation.
            - 'none': Treat missing value as is. Model must be able to handle missing value.
            - 'error' : Do not proceed with prediction and terminate with
                        an error message.

        output_type : str, optional
            output type that maybe needed by some of the toolkits

        options : dict
            additional options to be passed in to prediction

        kwargs : dict
            additional options to be passed into prediction

        Returns
        -------
        out : SArray
            An SArray with model predictions.
        """
        if missing_value_action == 'auto':
            missing_value_action = select_default_missing_value_policy(self, 'predict')

        # Low latency path
        if isinstance(dataset, list):
            return _graphlab.extensions._fast_predict(self.__proxy__, dataset,
                    output_type, missing_value_action)
        if isinstance(dataset, dict):
            return _graphlab.extensions._fast_predict(self.__proxy__, [dataset],
                    output_type, missing_value_action)

        # Batch predictions path
        else:
            _raise_error_if_not_sframe(dataset, "dataset")

            options = options.copy()
            options.update(kwargs)

            options.update({'model': self.__proxy__,
                            'model_name': self.__name__,
                            'dataset': dataset,
                            'missing_value_action' : missing_value_action,
                            'output_type' : output_type
                            })

            target = _graphlab.toolkits._main.run(
                      'supervised_learning_predict', options)
            return _map_unity_proxy_to_object(target['predicted'])

    def evaluate(self, dataset, metric="auto",
                 missing_value_action='auto', options={}, **kwargs):
        """
        Evaluate the model by making predictions of target values and comparing
        these to actual values.

        Parameters
        ----------
        dataset : SFrame
            Dataset in the same format used for training. The columns names and
            types of the dataset must be the same as that used in training.

        metric : str, list[str]
            Evaluation metric(s) to be computed.

        missing_value_action: str, optional
            Action to perform when missing values are encountered. This can be
            one of:

            - 'auto': Choose a model dependent missing value policy.
            - 'impute': Proceed with evaluation by filling in the missing
                        values with the mean of the training data. Missing
                        values are also imputed if an entire column of data is
                        missing during evaluation.
            - 'none': Treat missing value as is. Model must be able to handle missing value.
            - 'error' : Do not proceed with prediction and terminate with
                        an error message.

        options : dict
            additional options to be passed in to prediction

        kwargs : dict
            additional options to be passed into prediction
        """
        if missing_value_action == 'auto':
            missing_value_action = select_default_missing_value_policy(
                                                             self, 'evaluate')

        _raise_error_if_not_sframe(dataset, "dataset")
        options = options.copy()
        options.update(kwargs)

        options.update({'model': self.__proxy__,
                        'dataset': dataset,
                        'model_name': self.__name__,
                        'missing_value_action': missing_value_action,
                        'metric': metric
                        })
        results = _graphlab.toolkits._main.run(
                'supervised_learning_evaluate', options)
        return _map_unity_proxy_to_object(results)

    def _training_stats(self):
        """
        Return a dictionary containing statistics collected during model
        training. These statistics are also available with the ``get`` method,
        and are described in more detail in the documentation for that method.

        Notes
        -----
        """
        opts = {'model': self.__proxy__, 'model_name': self.__name__}
        results = _graphlab.toolkits._main.run("supervised_learning_get_train_stats", opts)
        return _map_unity_proxy_to_object(results)

    def get(self, field):
        """
        Get the value of a given field.

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out : [various]
            The current value of the requested field.

        See Also
        --------
        list_fields
        """
        opts = {'model': self.__proxy__,
                'model_name': self.__name__,
                'field': field}
        response = _graphlab.toolkits._main.run('supervised_learning_get_value', opts)
        return _map_unity_proxy_to_object(response['value'])

    def show(self, view=None, model_type='regression'):
        """
        show(view=None)
        Visualize with GraphLab Canvas :mod:`~graphlab.canvas`.
        This function starts Canvas if it is not already running.
        If the Model has already been plotted, this function will update the plot.

        Parameters
        ----------
        view : str, optional
            The name of the Model view to show. Can be one of:

            - *Summary*: The summary description of a Model.
            - *Evaluation*: A visual representation of the evaluation results for
              a Model.

        Returns
        -------
        view : graphlab.canvas.view.View
            An object representing the GraphLab Canvas view

        See Also
        --------
        canvas

        Examples
        --------
        Suppose 'm' is a Model, we can view it in GraphLab Canvas using:

        >>> m.show()
        """
        from graphlab.visualization.show import show
        show(self, view=view, model_type=model_type)

    @classmethod
    def _get_queryable_methods(cls):
        '''Returns a list of method names that are queryable through Predictive
        Service'''
        return {'predict': {}}


class Classifier(SupervisedLearningModel):
    """
    Clasifier module to predict a discrete target variable as a function of
    several feature variables.
    """
    
    def classify(self, dataset, missing_value_action='auto'):
        """
        Return predictions for ``dataset``, using the trained supervised_learning
        model. Predictions are generated as class labels (0 or
        1).

        Parameters
        ----------
        dataset: SFrame
            Dataset of new observations. Must include columns with the same
            names as the features used for model training, but does not require
            a target column. Additional columns are ignored.

        missing_value_action: str, optional
            Action to perform when missing values are encountered. This can be
            one of:

            - 'auto': Choose model dependent missing value action
            - 'impute': Proceed with evaluation by filling in the missing
              values with the mean of the training data. Missing
              values are also imputed if an entire column of data is
              missing during evaluation.
            - 'error': Do not proceed with prediction and terminate with
              an error message.
        Returns
        -------
        out : SFrame
            An SFrame with model predictions.
        """
        if (missing_value_action == 'auto'):
            missing_value_action = select_default_missing_value_policy(self, 'classify')

        # Low latency path
        if isinstance(dataset, list):
            return _graphlab.extensions._fast_classify(self.__proxy__, dataset,
                    missing_value_action)
        if isinstance(dataset, dict):
            return _graphlab.extensions._fast_classify(self.__proxy__, [dataset],
                    missing_value_action)

        _raise_error_if_not_sframe(dataset, "dataset")
        options = {}
        options.update({'model': self.__proxy__,
                        'model_name': self.__name__,
                        'dataset': dataset,
                        'missing_value_action': missing_value_action,
                        })
        target = _graphlab.toolkits._main.run('supervised_learning_classify', options)
        return _map_unity_proxy_to_object(target['classify'])

    @classmethod
    def _get_queryable_methods(cls):
        '''Returns a list of method names that are queryable through Predictive
        Service'''
        return {'predict': {},
                'predict_topk': {},
                'classify': {}}


def print_validation_track_notification():
    print ("PROGRESS: Creating a validation set from 5 percent of training data. This may take a while.\n"
           "          You can set ``validation_set=None`` to disable validation tracking.\n")


def create(dataset, target, model_name, features=None,
           validation_set='auto', verbose=True, distributed='auto', **kwargs):
    """
    Create a :class:`~graphlab.toolkits.SupervisedLearningModel`,

    This is generic function that allows you to create any model that
    implements SupervisedLearningModel This function is normally not called, call
    specific model's create function instead

    Parameters
    ----------
    dataset : SFrame
        Dataset for training the model.

    target : string
        Name of the column containing the target variable. The values in this
        column must be 0 or 1, of integer type.

    model_name : string
        Name of the model

    features : list[string], optional
        List of feature names used by feature column

    validation_set : SFrame, optional

        A dataset for monitoring the model's generalization performance.
        For each row of the progress table, the chosen metrics are computed
        for both the provided training dataset and the validation_set. The
        format of this SFrame must be the same as the training set.
        By default this argument is set to 'auto' and a validation set is
        automatically sampled and used for progress printing. If
        validation_set is set to None, then no additional metrics
        are computed. The default value is 'auto'.

    distributed: env
        The distributed environment

    verbose : boolean
        whether print out messages during training

    kwargs : dict
        Additional parameter options that can be passed
    """

    _raise_error_if_not_sframe(dataset, "training dataset")

    # Create a validation set
    if isinstance(validation_set, str):
        if validation_set == 'auto':
            if dataset.num_rows() >= 100:
                if verbose:
                    print_validation_track_notification()
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
            'features_validation' : _toolkits_select_columns(validation_set, features),
            'target_validation' : _toolkits_select_columns(validation_set, [target])})


    ret = _graphlab.toolkits._main.run("supervised_learning_train",
                                       options, verbose)
    model = SupervisedLearningModel(ret['model'], model_name)

    return model


def create_regression_with_model_selector(dataset, target, model_selector,
    features = None, validation_set='auto', verbose = True):
    """
    Create a :class:`~graphlab.toolkits.SupervisedLearningModel`,

    This is generic function that allows you to create any model that
    implements SupervisedLearningModel This function is normally not called, call
    specific model's create function instead

    Parameters
    ----------
    dataset : SFrame
        Dataset for training the model.

    target : string
        Name of the column containing the target variable. The values in this
        column must be 0 or 1, of integer type.

    model_name : string
        Name of the model

    model_selector: function
        Provide a model selector.

    features : list[string], optional
        List of feature names used by feature column

    verbose : boolean
        whether print out messages during training

    """

    # Error checking
    _raise_error_if_not_sframe(dataset, "training dataset")
    if features is None:
        features = dataset.column_names()
        if target in features:
            features.remove(target)
    if not hasattr(features, '__iter__'):
        raise TypeError("Input 'features' must be a list.")
    if not all([isinstance(x, str) for x in features]):
        raise TypeError("Invalid feature %s: Feature names must be of type str" % x)

    # Sample the data
    features_sframe = _toolkits_select_columns(dataset, features)
    if features_sframe.num_rows() > 1e5:
        fraction = 1.0 * 1e5 / features_sframe.num_rows()
        features_sframe = features_sframe.sample(fraction, seed = 0)

    # Run the model selector.
    selected_model_name = model_selector(features_sframe)
    model = create_selected(selected_model_name, dataset, target, features, validation_set, verbose)

    return model


def create_classification_with_model_selector(dataset, target, model_selector,
    features=None, validation_set='auto', verbose=True):
    """
    Create a :class:`~graphlab.toolkits.SupervisedLearningModel`,

    This is generic function that allows you to create any model that
    implements SupervisedLearningModel. This function is normally not called, call
    specific model's create function instead.

    Parameters
    ----------
    dataset : SFrame
        Dataset for training the model.

    target : string
        Name of the column containing the target variable. The values in this
        column must be 0 or 1, of integer type.

    model_name : string
        Name of the model

    model_selector: function
        Provide a model selector.

    features : list[string], optional
        List of feature names used by feature column

    verbose : boolean
        whether print out messages during training

    """

    # Error checking
    _raise_error_if_not_sframe(dataset, "training dataset")
    if features is None:
        features = dataset.column_names()
        if target in features:
            features.remove(target)
    if not hasattr(features, '__iter__'):
        raise TypeError("Input 'features' must be a list.")
    if not all([isinstance(x, str) for x in features]):
        raise TypeError("Invalid feature %s: Feature names must be of type str" % x)

    # Sample the data
    features_sframe = _toolkits_select_columns(dataset, features)
    if features_sframe.num_rows() > 1e5:
        fraction = 1.0 * 1e5 / features_sframe.num_rows()
        features_sframe = features_sframe.sample(fraction, seed = 0)

    # Get available models for this dataset
    num_classes = dataset[target].unique().size()
    selected_model_names = model_selector(num_classes, features_sframe)

    # Create a validation set
    if isinstance(validation_set, str):
        if validation_set == 'auto':
            if dataset.num_rows() >= 100:
                if verbose:
                    print_validation_track_notification()
                dataset, validation_set = dataset.random_split(.95)
            else:
                validation_set = None
        else:
            raise TypeError('Unrecognized value for validation_set.')

    # Match C++ model names with user model names
    python_names = {'boosted_trees_classifier': 'BoostedTreesClassifier',
                    'random_forest_classifier': 'RandomForestClassifier',
                    'decision_tree_classifier': 'DecisionTreeClassifier',
                    'classifier_logistic_regression': 'LogisticClassifier',
                    'classifier_svm': 'SVMClassifier',
                    'neuralnet_classifier': 'NeuralNetClassifier',
                    'neuralnet_classifier_v2': 'NeuralNetClassifier'}

    # Print useful user-facing progress messages
    if verbose:
        print('PROGRESS: The following methods are available for this type of problem.')
        print('PROGRESS: ' + ', '.join([python_names[x] for x in selected_model_names]))
        if len(selected_model_names) > 1:
            print('PROGRESS: The returned model will be chosen according to validation accuracy.')

    models = {}
    metrics = {}
    for model_name in selected_model_names:

        # Fit each of the available models
        m = create_selected(model_name, dataset, target, features, validation_set, verbose)
        models[model_name] = m

        if 'validation_accuracy' in m.list_fields():
            metrics[model_name] = m['validation_accuracy']

        # Most models have this.
        elif 'progress' in m.list_fields():
            prog = m['progress']
            validation_column = 'Validation-accuracy'
            accuracy_column = 'Training-accuracy'
            if validation_column in prog.column_names():
                metrics[model_name] = float(prog[validation_column].tail(1)[0])
            else:
                metrics[model_name] = float(prog[accuracy_column].tail(1)[0])
        else:
            raise ValueError("Model does not have metrics that can be used for model selection.")

    # Choose model based on either validation, if available.
    best_model = None
    best_acc = None
    for model_name in selected_model_names:
        if best_acc is None:
            best_model = model_name
            best_acc = metrics[model_name]
        if best_acc is not None and best_acc < metrics[model_name]:
            best_model = model_name
            best_acc = metrics[model_name]

    ret = []
    width = 32
    if len(selected_model_names) > 1:
        ret.append('PROGRESS: Model selection based on validation accuracy:')
        ret.append('---------------------------------------------')
        key_str = '{:<{}}: {}'
        for model_name in selected_model_names:
            name = python_names[model_name]
            row = key_str.format(name, width, str(metrics[model_name]))
            ret.append(row)
        ret.append('---------------------------------------------')
        ret.append('Selecting ' + python_names[best_model] + ' based on validation set performance.')

    if verbose:
        print('\nPROGRESS: '.join(ret))
    return models[best_model]

def create_selected(selected_model_name, dataset, target, features,
                    validation_set='auto', verbose=True):

    if (selected_model_name == 'neuralnet_classifier_v2'):
        model = _graphlab.classifier.neuralnet_classifier.create(dataset,
            target,
            features=features,
            validation_set=validation_set,
            verbose=verbose)
        return model

    else:

        # Create the model
        model = create(dataset,
            target,
            selected_model_name,
            features=features,
            validation_set=validation_set,
            verbose=verbose)

        # Return the model
        if selected_model_name == 'boosted_trees_regression':
            return _graphlab.boosted_trees_regression.BoostedTreesRegression(\
                model.__proxy__)
        elif selected_model_name == 'random_forest_regression':
            return _graphlab.random_forest_regression.RandomForestRegression(\
                model.__proxy__)
        elif selected_model_name == 'decision_tree_regression':
            return _graphlab.decision_tree_classifier.DecisionTreeRegression(\
              model.__proxy__)
        elif selected_model_name == 'regression_linear_regression':
            return _graphlab.linear_regression.LinearRegression(\
                model.__proxy__)
        elif selected_model_name == 'boosted_trees_classifier':
            return _graphlab.boosted_trees_classifier.BoostedTreesClassifier(\
              model.__proxy__)
        elif selected_model_name == 'random_forest_classifier':
            return _graphlab.random_forest_classifier.RandomForestClassifier(\
              model.__proxy__)
        elif selected_model_name == 'decision_tree_classifier':
            return _graphlab.decision_tree_classifier.DecisionTreeClassifier(\
              model.__proxy__)
        elif selected_model_name == 'classifier_logistic_regression':
            return _graphlab.logistic_classifier.LogisticClassifier(\
              model.__proxy__)
        elif selected_model_name == 'classifier_svm':
            return _graphlab.svm_classifier.SVMClassifier(model.__proxy__)
        else:
            raise ToolkitError("Internal error: Incorrect model returned.")

    def show(self, view=None, model_type='classification'):
        """
        show(view=None)
        Visualize a Classifier model with GraphLab Create
        :mod:`~graphlab.canvas`.
        This function starts Canvas if it is not already running.
        If the Model has already been plotted, this function will update the
        plot.

        Parameters
        ----------
        view : str, optional
            The name of the Model view to show. Can be one of:

            - <Type>: <Description>

        Returns
        -------
        view : graphlab.canvas.view.View
            An object representing the GraphLab Canvas view

        See Also
        --------
        canvas

        Examples
        --------
        Suppose 'm' is a Model, we can view it in GraphLab Canvas using:

        >>> m.show()
        """
        from graphlab.visualization.show import show
        show(self, view=view, model_type=model_type)


def _show_model_tree(model, tree_id=0, vlabel_hover=False):
    """
    Plot the tree as an SGraph in canvas.

    Parameters
    ----------
    tree_id : int
        The id of the tree to show. Starting from 0 to num_trees-1.

    vlabel_hover : bool, optional
        If True, hide the label on vertex, and only show the label
        with mouse hovering.

    See Also
    --------
    summary

    Examples
    --------
    >>> model.show(0)
    """
    tree_json = model.get('trees_json')[tree_id]
    g = _SGraphFromJsonTree(tree_json)

    # Make the label for each vertex based on its type
    def get_vlabel(record):
        if record['type'] == 'leaf':
            return str(record['value'])
        elif record['type'] == 'indicator':
            return str(record['name'])
        else:
            return str(record['name']) + '<' + str(record['value'])
    g.vertices['__repr__'] = g.vertices.apply(get_vlabel, str)

    fuchsia = [0.69, 0., 0.48]
    blue = [0.039, 0.55, 0.77]
    orange = [1.0, 0.33, 0.]
    green = [0.52, 0.74, 0.]

    # Make the color for leaf nodes
    def get_leaf_vcolor(record):
        '''assign color to leaf vertex, negative_leaf->blue, positive_leaf->orange'''
        return blue if record['value'] < 0 else orange

    leaf_vertices = g.vertices[g.vertices['type'] == 'leaf']
    root_vertex_color = {0: fuchsia}
    leaf_vertex_color = dict([(x['__id'], get_leaf_vcolor(x)) for x in leaf_vertices])
    highlight = {}
    highlight.update(root_vertex_color)
    highlight.update(leaf_vertex_color)
    elabel = 'value' if len(g.edges) > 0 else None

    # Hack: we want the canvas to show tree_{i} instead of g, so here is how we do it.
    graph_name = 'tree_' + str(tree_id)
    locals().update({graph_name: g})
    del g
    locals()[graph_name].show(vlabel='__repr__', elabel=elabel,
                              vlabel_hover=vlabel_hover,
                              highlight=highlight, arrows=True)


def select_default_missing_value_policy(model, action):
    from .classifier.boosted_trees_classifier import BoostedTreesClassifier
    from .classifier.random_forest_classifier import RandomForestClassifier
    from .classifier.decision_tree_classifier import DecisionTreeClassifier
    from .classifier.neuralnet_classifier import NeuralNetClassifier
    from .regression.boosted_trees_regression import BoostedTreesRegression
    from .regression.random_forest_regression import RandomForestRegression
    from .regression.decision_tree_regression import DecisionTreeRegression

    tree_models = [BoostedTreesClassifier, BoostedTreesRegression,
                   RandomForestClassifier, RandomForestRegression,
                   DecisionTreeClassifier, DecisionTreeRegression]

    neuralnet_model = NeuralNetClassifier

    if (any(isinstance(model, tree_model) for tree_model in tree_models)):
        return 'none'
    elif isinstance(model, neuralnet_model):
        return 'error'
    else:
        return 'impute'
