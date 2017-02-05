import graphlab.connect as _mt
from graphlab.data_structures.sgraph import SGraph as _SGraph

def submit_training_job(env, graph, reset_probability=0.15,
    threshold=1e-2,
    max_iterations=20,
    _single_precision=False):
    """
    Submit job to compute the PageRank for each vertex in the graph. Return an
    object that tracks the execution of this job, from which a model with total
    PageRank as well as the PageRank value for each vertex in the graph is
    obtained.

    Parameters
    ----------
    env : graphlab.deploy.hadoop_cluster.HadoopCluster
        Hadoop cluster to submit the training job

    graph : SGraph
        The graph on which to compute the pagerank value.

    reset_probability : float, optional
        Probability that a random surfer jumps to an arbitrary page.

    threshold : float, optional
        Threshold for convergence, measured in the L1 norm
        (the sum of absolute value) of the delta of each vertex's
        pagerank value.

    max_iterations : int, optional
        The maximun number of iterations to run.

    _single_precision : bool, optional
        If true, running pagerank in single precision. The resulting
        pagerank values may not be accurate for large graph, but
        should run faster and use less memory.

    Returns
    -------
      out : :class:`~graphlab.distributed._dml_job_status.DMLJobStatus`
          An object that tracks the execution of the distributed training job.

    References
    ----------
    - `Wikipedia - PageRank <http://en.wikipedia.org/wiki/PageRank>`_
    - Page, L., et al. (1998) `The PageRank Citation Ranking: Bringing Order to
      the Web <http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf>`_.

    Examples
    --------
    If given an :class:`~graphlab.SGraph` ``g``, we can create
    a :class:`~graphlab.pagerank.PageRankModel` as follows:

    >>> hdp_env = graphlab.deploy.hadoop_cluster.create('my-first-hadoop-cluster',
    ...    'hdfs://path-to-turi-distributed-installation')
    >>> g = graphlab.load_graph('http://snap.stanford.edu/data/email-Enron.txt.gz', format='snap')
    >>> distr_obj = graphlab.distributed.pagerank.submit_training_job(hdp_env, g)
    >>> model = distr_job.get_results()

    We can obtain the page rank corresponding to each vertex in the graph ``g``
    using:

    >>> pr_out = pr['pagerank']     # SFrame

    We can add the new pagerank field to the original graph g using:

    >>> g.vertices['pagerank'] = pr['graph'].vertices['pagerank']

    Note that the task above does not require a join because the vertex
    ordering is preserved through ``create()``.

    See Also
    --------
    PagerankModel
    """
    _mt._get_metric_tracker().track('distributed.toolkit.graph_analytics.pagerank.submit_training_job')

    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')

    opts = {'threshold': threshold, 'reset_probability': reset_probability,
            'max_iterations': max_iterations,
            'single_precision': _single_precision,
            'graph': graph.__proxy__}

    from ... import _dml
    dml_obj = _dml.run('distributed_pagerank', 'pagerank', opts, env)

    return dml_obj
