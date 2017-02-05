import graphlab.connect as _mt
from graphlab.util import _raise_error_if_not_of_type
from graphlab.data_structures.sgraph import SGraph as _SGraph


def submit_training_job(env, graph, label_field,
           threshold=1e-3,
           weight_field='',
           self_weight=1.0,
           undirected=False,
           max_iterations=None,
           _single_precision=False):
    """
    Given a weighted graph with observed class labels of a subset of vertices,
    infer the label probability for the unobserved vertices using the
    "label propagation" algorithm.

    The algorithm iteratively updates the label probability of current vertex
    as a weighted sum of label probability of self and the neighboring vertices
    until converge.  See
    :class:`graphlab.label_propagation.LabelPropagationModel` for the details
    of the algorithm.

    Notes: label propagation works well with small number of labels, i.e. binary
    labels, or less than 1000 classes. The toolkit will throw error
    if the number of classes exceeds the maximum value (1000).

    Parameters
    ----------
    env : graphlab.deploy.hadoop_cluster.HadoopCluster
        Hadoop cluster to submit the training job

    graph : SGraph
        The graph on which to compute the label propagation.

    label_field: str
        Vertex field storing the initial vertex labels. The values in
        must be [0, num_classes). None values indicate unobserved vertex labels.

    threshold : float, optional
        Threshold for convergence, measured in the average L2 norm
        (the sum of squared values) of the delta of each vertex's
        label probability vector.

    max_iterations: int, optional
        The max number of iterations to run. Default is unlimited.
        If set, the algorithm terminates when either max_iterations
        or convergence threshold is reached.

    weight_field: str, optional
        Vertex field for edge weight. If empty, all edges are assumed
        to have unit weight.

    self_weight: float, optional
        The weight for self edge.

    undirected: bool, optional
        If true, treat each edge as undirected, and propagates label in
        both directions.

    _single_precision : bool, optional
        If true, running label propagation in single precision. The resulting
        probability values may less accurate, but should run faster
        and use less memory.

    Returns
    -------
      out : :class:`~graphlab.distributed._dml_job_status.DMLJobStatus`
          An object that tracks the execution of the distributed training job.

    References
    ----------
    - Zhu, X., & Ghahramani, Z. (2002). `Learning from labeled and unlabeled data
      with label propagation <http://www.cs.cmu.edu/~zhuxj/pub/CMU-CALD-02-107.pdf>`_.

    Examples
    --------
    If given an :class:`~graphlab.SGraph` ``g``, we can create
    a :class:`~graphlab.label_propagation.LabelPropagationModel` as follows:

    >>> hdp_env = graphlab.deploy.hadoop_cluster.create('my-first-hadoop-cluster',
    ...    'hdfs://path-to-turi-distributed-installation')
    >>> g = graphlab.load_graph('http://snap.stanford.edu/data/email-Enron.txt.gz',
    ...                         format='snap')
    # Initialize random classes for a subset of vertices
    # Leave the unobserved vertices with None label.
    >>> import random
    >>> def init_label(vid):
    ...     x = random.random()
    ...     if x < 0.2:
    ...         return 0
    ...     elif x > 0.9:
    ...         return 1
    ...     else:
    ...         return None
    >>> g.vertices['label'] = g.vertices['__id'].apply(init_label, int)
    >>> distr_obj = graphlab.distributed.label_propagation.submit_training_job(hdp_env, g, 'label')
    >>> model = distr_job.get_results()

    We can obtain for each vertex the predicted label and the probability of
    each label in the graph ``g`` using:

    >>> labels = m['labels']     # SFrame
    >>> labels
    +------+-------+-----------------+-------------------+----------------+
    | __id | label | predicted_label |         P0        |       P1       |
    +------+-------+-----------------+-------------------+----------------+
    |  5   |   1   |        1        |        0.0        |      1.0       |
    |  7   |  None |        0        |    0.8213214997   |  0.1786785003  |
    |  8   |  None |        1        | 5.96046447754e-08 | 0.999999940395 |
    |  10  |  None |        0        |   0.534984718273  | 0.465015281727 |
    |  27  |  None |        0        |   0.752801638549  | 0.247198361451 |
    |  29  |  None |        1        | 5.96046447754e-08 | 0.999999940395 |
    |  33  |  None |        1        | 5.96046447754e-08 | 0.999999940395 |
    |  47  |   0   |        0        |        1.0        |      0.0       |
    |  50  |  None |        0        |   0.788279032657  | 0.211720967343 |
    |  52  |  None |        0        |   0.666666666667  | 0.333333333333 |
    +------+-------+-----------------+-------------------+----------------+
    [36692 rows x 5 columns]

    See Also
    --------
    LabelPropagationModel
    """
    _mt._get_metric_tracker().track('distributed.graph_analytics.label_propagation.create')

    _raise_error_if_not_of_type(label_field, str)
    _raise_error_if_not_of_type(weight_field, str)

    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')

    if graph.vertices[label_field].dtype() != int:
        raise TypeError('label_field %s must be integer typed.' % label_field)

    opts = {'label_field': label_field,
            'threshold': threshold,
            'weight_field': weight_field,
            'self_weight': self_weight,
            'undirected': undirected,
            'max_iterations': max_iterations,
            'single_precision': _single_precision,
            'graph': graph.__proxy__}

    from ... import _dml
    dml_obj = _dml.run('distributed_labelprop', 'label_propagation', opts, env)

    return dml_obj
