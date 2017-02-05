from ... import _supervised_learning as _sl
import graphlab.connect as _mt

_DEFAULT_SOLVER_OPTIONS = {
'convergence_threshold': 1e-2,
'max_iterations': 10,
'lbfgs_memory_level': 11,
}

def submit_training_job(env, dataset, target, features=None,
    penalty=1.0, solver='auto',
    feature_rescaling=True,
    convergence_threshold = _DEFAULT_SOLVER_OPTIONS['convergence_threshold'],
    lbfgs_memory_level = _DEFAULT_SOLVER_OPTIONS['lbfgs_memory_level'],
    max_iterations = _DEFAULT_SOLVER_OPTIONS['max_iterations'],
    class_weights = None,
    validation_set = None):
    """
    Submit job to create a :class:`~graphlab.svm_classifier.SVMClassifier` to
    predict the class of a binary target variable based on a model of which
    side of a hyperplane the example falls on. In addition to standard numeric
    and categorical types, features can also be extracted automatically from
    list- or dictionary-type SFrame columns.

    This loss function for the SVM model is the sum of an L1 mis-classification
    loss (multiplied by the 'penalty' term) and a l2-norm on the weight vectors.

    Parameters
    ----------
    env : graphlab.deploy.hadoop_cluster.HadoopCluster
        Hadoop cluster to submit the training job

    dataset : SFrame
        Dataset for training the model.

    target : string
        Name of the column containing the target variable. The values in this
        column must be of string or integer type. String target variables are
        automatically mapped to integers in alphabetical order of the variable
        values. For example, a target variable with 'cat' and 'dog' as possible
        values is mapped to 0 and 1 respectively with 0 being the base class
        and 1 being the reference class.

    features : list[string], optional
        Names of the columns containing features. 'None' (the default) indicates
        that all columns except the target variable should be used as features.

        The features are columns in the input SFrame that can be of the
        following types:

        - *Numeric*: values of numeric type integer or float.

        - *Categorical*: values of type string.

        - *Array*: list of numeric (integer or float) values. Each list element
          is treated as a separate feature in the model.

        - *Dictionary*: key-value pairs with numeric (integer or float) values
          Each key of a dictionary is treated as a separate feature and the
          value in the dictionary corresponds to the value of the feature.
          Dictionaries are ideal for representing sparse data.

        Columns of type *list* are not supported. Convert them to array in
        case all entries in the list are of numeric types and separate them
        out into different columns if they are of mixed type.

    penalty : float, optional
        Penalty term on the mis-classification loss of the model. The larger
        this weight, the more the model coefficients shrink toward 0.  The
        larger the penalty, the lower is the emphasis placed on misclassified
        examples, and the classifier would spend more time maximizing the
        margin for correctly classified examples. The default value is 1.0;
        this parameter must be set to a value of at least 1e-10.


    solver : string, optional
        Name of the solver to be used to solve the problem. See the
        references for more detail on each solver. Available solvers are:

        - *auto (default)*: automatically chooses the best solver (from the ones
          listed below) for the data and model parameters.
        - *lbfgs*: lLimited memory BFGS (``lbfgs``) is a robust solver for wide
          datasets(i.e datasets with many coefficients).

        The solvers are all automatically tuned and the default options should
        function well. See the solver options guide for setting additional
        parameters for each of the solvers.

    feature_rescaling : bool, default = true

        Feature rescaling is an important pre-processing step that ensures
        that all features are on the same scale. An l2-norm rescaling is
        performed to make sure that all features are of the same norm. Categorical
        features are also rescaled by rescaling the dummy variables that
        are used to represent them. The coefficients are returned in original
        scale of the problem.

    convergence_threshold :

        Convergence is tested using variation in the training objective. The
        variation in the training objective is calculated using the difference
        between the objective values between two steps. Consider reducing this
        below the default value (0.01) for a more accurately trained model.
        Beware of overfitting (i.e a model that works well only on the training
        data) if this parameter is set to a very low value.

    max_iterations : int, optional

        The maximum number of allowed passes through the data. More passes over
        the data can result in a more accurately trained model. Consider
        increasing this (the default value is 10) if the training accuracy is
        low and the *Grad-Norm* in the display is large.

    lbfgs_memory_level : int, optional

        The L-BFGS algorithm keeps track of gradient information from the
        previous ``lbfgs_memory_level`` iterations. The storage requirement for
        each of these gradients is the ``num_coefficients`` in the problem.
        Increasing the ``lbfgs_memory_level`` can help improve the quality of
        the model trained. Setting this to more than ``max_iterations`` has the
        same effect as setting it to ``max_iterations``.

    class_weights : {dict, `auto`}, optional

        Weights the examples in the training data according to the given class
        weights. If set to `None`, all classes are supposed to have weight one. The
        `auto` mode set the class weight to be inversely proportional to number of
        examples in the training data with the given class.

    validation_set : SFrame, optional

        A dataset for monitoring the model's generalization performance.
        For each row of the progress table, the chosen metrics are computed
        for both the provided training dataset and the validation_set. The
        format of this SFrame must be the same as the training set.
        By default this argument is set to 'auto' and a validation set is
        automatically sampled and used for progress printing. If
        validation_set is set to None, then no additional metrics
        are computed.

    Returns
    -------
      out : :class:`~graphlab.distributed._dml_job_status.DMLJobStatus`
          An object that tracks the execution of the distributed training job.

    See Also
    --------
    SVMClassifier

    Notes
    -----
    - Categorical variables are encoded by creating dummy variables. For
      a variable with :math:`K` categories, the encoding creates :math:`K-1`
      dummy variables, while the first category encountered in the data is used
      as the baseline.

    - For prediction and evaluation of SVM models with sparse dictionary
      inputs, new keys/columns that were not seen during training are silently
      ignored.

    - The penalty parameter is analogous to the 'C' term in the C-SVM. See the
      reference on training SVMs for more details.

    - Any 'None' values in the data will result in an error being thrown.

    - A constant term of '1' is automatically added for the model intercept to
      model the bias term.

    - Note that the hinge loss is approximated by the scaled logistic loss
      function. (See user guide for details)

    References
    ----------
    - `Wikipedia - Support Vector Machines
      <http://en.wikipedia.org/wiki/svm>`_

    - Zhang et al. - Modified Logistic Regression: An Approximation to
      SVM and its Applications in Large-Scale Text Categorization (ICML 2003)


    Examples
    --------

    Given an :class:`~graphlab.SFrame` ``sf``, a list of feature columns
    [``feature_1`` ... ``feature_K``], and a target column ``target`` with 0 and
    1 values, create a
    :class:`~graphlab.svm.SVMClassifier` as follows:

    >>> hdp_env = graphlab.deploy.hadoop_cluster.create('my-first-hadoop-cluster',
    ...    'hdfs://path-to-turi-distributed-installation')
    >>> data =  graphlab.SFrame('https://static.turi.com/datasets/regression/houses.csv')
    >>> data['is_expensive'] = data['price'] > 30000
    >>> distr_job = graphlab.distributed.svm_classifier.submit_training_job(hdp_env, data, 'is_expensive')
    >>> model = distr_job.get_results()
    """
    _mt._get_metric_tracker().track('distributed.toolkit.classifier.svm_classifier.submit_training_job')

    # Regression model names.
    model_name = "classifier_svm"
    solver = solver.lower()

    dml_obj = _sl.create(dataset, target, model_name, env=env, features=features,
                        validation_set = validation_set,
                        penalty = penalty,
                        feature_rescaling = feature_rescaling,
                        convergence_threshold = convergence_threshold,
                        lbfgs_memory_level = lbfgs_memory_level,
                        max_iterations = max_iterations,
                        class_weights = class_weights)

    return dml_obj
