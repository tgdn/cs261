from ... import _supervised_learning as _sl
import logging as _logging
import graphlab.connect as _mt
from graphlab.util import _make_internal_url

def submit_training_job(env, dataset, target,
    features=None,
    max_iterations=10,
    validation_set=None,
    random_seed = None,
    metric = 'auto',
    model_checkpoint_path='auto',
    **kwargs):
    """
    Submit a job to create a :class:`~graphlab.random_forest_regression.RandomForestRegression` to predict
    a scalar target variable using one or more features. In addition to standard
    numeric and categorical types, features can also be extracted automatically
    from list- or dictionary-type SFrame columns.


    Parameters
    ----------
    env : graphlab.deploy.hadoop_cluster.HadoopCluster
        Hadoop cluster to submit the training job

    dataset : SFrame
        A training dataset containing feature columns and a target column.
        Only numerical typed (int, float) target column is allowed.

    target : str
        The name of the column in ``dataset`` that is the prediction target.
        This column must have a numeric type.

    features : list[str], optional
        A list of columns names of features used for training the model.
        Defaults to None, using all columns.

    max_iterations : int, optional
        The number of iterations to perform.

    max_depth : float, optional
        Maximum depth of a tree.

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

    row_subsample : float, optional
        Subsample the ratio of the training set in each iteration of tree
        construction.  This is called the bagging trick and usually can help
        prevent overfitting.  Setting it to 0.5 means that model randomly
        collected half of the examples (rows) to grow each tree.

    column_subsample : float, optional
        Subsample ratio of the columns in each iteration of tree
        construction.  Like row_subsample, this also usually can help
        prevent overfitting.  Setting it to 0.5 means that model randomly
        collected half of the columns to grow each tree.

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


    random_seed : int, optional
        Seeds random opertations such as column and row subsampling, such that
        results are reproducable.

    metric : str or list[str], optional
        Performance metric(s) that are tracked during training. When specified,
        the progress table will display the tracked metric(s) on training and
        validation set.
        Supported metrics are: {'rmse', 'max_error'}

    kwargs : dict, optional
        Additional arguments for training the model.

        - ``model_checkpoint_path`` : str, default 'auto'
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
    - `Trevor Hastie's slides on Boosted Trees and Random Forest
      <http://jessica2.msri.org/attachments/10778/10778-boost.pdf>`_

    See Also
    --------
    RandomForestRegression, graphlab.linear_regression.LinearRegression, graphlab.regression.create

    Examples
    --------

    Setup the data:

    >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
    >>> data = graphlab.SFrame.read_csv(url)
    >>> data['label'] = data['label'] == 'p'
    >>> hdp_env = graphlab.deploy.hadoop_cluster.create('my-first-hadoop-cluster',
    ...    'hdfs://path-to-turi-distributed-installation')

    Split the data into training and test data:

    >>> train, test = data.random_split(0.8)

    Create the model:

    >>> distr_job = graphlab.random_forest_regression.submit_training_job(hdp_env, train, target='label')
    >>> model = distr_job.get_results()

    Make predictions and evaluate the model:

    >>> predictions = model.predict(test)
    >>> results = model.evaluate(test)

    """
    logger = _logging.getLogger(__name__)

    if random_seed is not None:
        kwargs['random_seed'] = random_seed
    if model_checkpoint_path != 'auto':
        model_checkpoint_path = _make_internal_url(model_checkpoint_path)
    if 'resume_from_checkpoint' in kwargs:
        kwargs['resume_from_checkpoint'] = _make_internal_url(kwargs['resume_from_checkpoint'])

    if 'num_trees' in kwargs:
        logger.warning("The `num_trees` keyword argument is deprecated. Please "
              "use the `max_iterations` argument instead. Any value provided "
              "for `num_trees` will be used in place of `max_iterations`.")
        max_iterations = kwargs['num_trees']
        del kwargs['num_trees']

    _mt._get_metric_tracker().track('distributed.toolkit.regression.random_forest_regression.submit_training_job')
    dml_obj = _sl.create(dataset = dataset,
                        target = target,
                        features = features,
                        model_name = 'random_forest_regression',
                        env = env,
                        max_iterations = max_iterations,
                        validation_set = validation_set,
                        metric = metric,
                        model_checkpoint_path=model_checkpoint_path,
                        **kwargs)
    return dml_obj
