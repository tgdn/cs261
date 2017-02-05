from ... import _supervised_learning as _sl
import graphlab.connect as _mt
from graphlab.util import _make_internal_url

def submit_training_job(env, dataset, target,
    features=None, max_iterations=10,
    validation_set=None,
    class_weights = None,
    max_depth=6, step_size=0.3,
    min_loss_reduction=0.0, min_child_weight=0.1,
    row_subsample=1.0, column_subsample=1.0,
    random_seed = None,
    metric='auto',
    model_checkpoint_path='auto',
    **kwargs):
    """
    Submit a job to create a (binary or multi-class) classifier model of type
    :class:`~graphlab.boosted_trees_classifier.BoostedTreesClassifier` using
    gradient boosted trees (sometimes known as GBMs).

    Parameters
    ----------
    env : graphlab.deploy.hadoop_cluster.HadoopCluster
        Hadoop cluster to submit the training job

    dataset : SFrame
        A training dataset containing feature columns and a target column.

    target : str
        Name of the column containing the target variable. The values in this
        column must be of string or integer type.  String target variables are
        automatically mapped to integers in alphabetical order of the variable values.
        For example, a target variable with 'cat', 'dog', and 'foosa' as possible
        values is mapped to 0, 1, and, 2 respectively.

    features : list[str], optional
        A list of columns names of features used for training the model.
        Defaults to None, which uses all columns in the SFrame ``dataset``
        excepting the target column..

    max_iterations : int, optional
        The maximum number of iterations for boosting. Each iteration results
        in the creation of an extra tree.

    validation_set : SFrame, optional
        A dataset for monitoring the model's generalization performance.
        For each row of the progress table, the chosen metrics are computed
        for both the provided training dataset and the validation_set. The
        format of this SFrame must be the same as the training set.
        By default this argument is set to 'auto' and a validation set is
        automatically sampled and used for progress printing. If
        validation_set is set to None, then no additional metrics
        are computed. This is computed once per full iteration. Large
        differences in model accuracy between the training data and validation
        data is indicative of overfitting.

    class_weights : {dict, `auto`}, optional

        Weights the examples in the training data according to the given class
        weights. If provided, the dictionary must contain a key for each class
        label. The value can be any positive number greater than 1e-20. Weights
        are interpreted as relative to each other. So setting the weights to be
        2.0 for the positive class and 1.0 for the negative class has the same
        effect as setting them to be 20.0 and 10.0, respectively. If set to
        `None`, all classes are taken to have weight 1.0. The `auto` mode sets
        the class weight to be inversely proportional to the number of examples
        in the training data with the given class.

    max_depth : float, optional
        Maximum depth of a tree. Must be at least 1.

    step_size : float, [0,1], optional
        Step size (shrinkage) used in update to prevents overfitting.  It
        shrinks the prediction of each weak learner to make the boosting
        process more conservative.  The smaller the step size, the more conservative
        the algorithm will be. Smaller step_size work well when
        `max_iterations` is large.

    min_loss_reduction : float, optional (non-negative)
        Minimum loss reduction required to make a further partition/split a
        node during the tree learning phase. Larger (more positive) values
        can help prevent overfitting by avoiding splits that do not
        sufficiently reduce the loss function.

    min_child_weight : float, optional (non-negative)
        Controls the minimum weight of each leaf node. Larger values result in
        more conservative tree learning and help prevent overfitting.
        Formally, this is minimum sum of instance weights (hessians) in each
        node. If the tree learning algorithm results in a leaf node with the
        sum of instance weights less than `min_child_weight`, tree building
        will terminate.

    row_subsample : float, [0,1], optional
        Subsample the ratio of the training set in each iteration of tree
        construction.  This is called the bagging trick and can usually help
        prevent overfitting.  Setting this to a value of 0.5 results in the
        model randomly sampling half of the examples (rows) to grow each tree.

    column_subsample : float, [0,1], optional
        Subsample ratio of the columns in each iteration of tree
        construction.  Like row_subsample, this can also help prevent
        model overfitting.  Setting this to a value of 0.5 results in the
        model randomly sampling half of the columns to grow each tree.

    random_seed : int, optional
        Seeds random opertations such as column and row subsampling, such that
        results are reproducable.

    metric : str or list[str], optional
        Performance metric(s) that are tracked during training. When specified,
        the progress table will display the tracked metric(s) on training and
        validation set.
        Supported metrics are: {'accuracy', 'auc', 'log_loss'}

    kwargs : dict, optional
        Additional arguments for training the model.

        - ``early_stopping_rounds`` : int, default None
            If the validation metric does not improve after <early_stopping_rounds>,
            stop training and return the best model.
            If multiple metrics are being tracked, the last one is used.

        - ``model_checkpoint_path`` : str, default None
            If specified, checkpoint the model training to the given path every n iterations,
            where n is specified by ``model_checkpoint_interval``.
            For instance, if `model_checkpoint_interval` is 5, and `model_checkpoint_path` is
            set to ``/tmp/model_tmp``, the checkpoints will be saved into
            ``/tmp/model_tmp/model_checkpoint_5``, ``/tmp/model_tmp/model_checkpoint_10``, ... etc.
            Training can be resumed by setting ``resume_from_checkpoint`` to one of these checkpoints.

        - ``model_checkpoint_interval`` : int, default 5
            If model_check_point_path is specified,
            save the model to the given path every n iterations.

        - ``resume_from_checkpoint`` : str, default None
            Continues training from a model checkpoint. The model must take
            exact the same training data as the checkpointed model.

    Returns
    -------
      out : :class:`~graphlab.distributed._dml_job_status.DMLJobStatus`
          An object that tracks the execution of the distributed training job.

    References
    ----------

    - `Wikipedia - Gradient tree boosting
      <http://en.wikipedia.org/wiki/Gradient_boosting#Gradient_tree_boosting>`_
    - `Trevor Hastie's slides on Boosted Trees and Random Forest
      <http://jessica2.msri.org/attachments/10778/10778-boost.pdf>`_

    See Also
    --------
    BoostedTreesClassifier, graphlab.logistic_classifier.LogisticClassifier, graphlab.svm_classifier.SVMClassifier, graphlab.neuralnet_classifier.NeuralNetClassifier

    Examples
    --------

    .. sourcecode:: python

      >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
      >>> data = graphlab.SFrame.read_csv(url)
      >>> hdp_env = graphlab.deploy.hadoop_cluster.create('my-first-hadoop-cluster',
      ...    'hdfs://path-to-turi-distributed-installation')

      >>> train, test = data.random_split(0.8)
      >>> distr_job = graphlab.distributed.boosted_trees_classifier.submit_training_job(hdp_env, train, target='label')
      >>> model = distr_job.get_results()

      >>> predicitons = model.classify(test)
      >>> results = model.evaluate(test)
    """
    _mt._get_metric_tracker().track('distributed.toolkit.classifier.boosted_trees_classifier.submit_training_job')

    if random_seed is not None:
        kwargs['random_seed'] = random_seed
    if model_checkpoint_path != 'auto':
        model_checkpoint_path = _make_internal_url(model_checkpoint_path)
    if 'resume_from_checkpoint' in kwargs:
        kwargs['resume_from_checkpoint'] = _make_internal_url(kwargs['resume_from_checkpoint'])

    dml_obj = _sl.create(dataset = dataset,
                        target = target,
                        features = features,
                        model_name = 'boosted_trees_classifier',
                        env = env,
                        max_iterations = max_iterations,
                        validation_set = validation_set,
                        class_weights = class_weights,
                        max_depth = max_depth,
                        step_size = step_size,
                        min_loss_reduction = min_loss_reduction,
                        min_child_weight = min_child_weight,
                        row_subsample = row_subsample,
                        column_subsample = column_subsample,
                        metric = metric,
                        model_checkpoint_path=model_checkpoint_path,
                        **kwargs)
    return dml_obj
