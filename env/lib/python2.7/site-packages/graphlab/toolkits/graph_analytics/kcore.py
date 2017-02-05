'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''
import graphlab.connect as _mt
from graphlab.data_structures.sgraph import SGraph as _SGraph
import graphlab.toolkits._main as _main
from graphlab.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase


def get_default_options():
    """
    Get the default options for :func:`graphlab.kcore.create`.

    Returns
    -------
    out : dict

    See Also
    --------
    KcoreModel.get_current_options

    Examples
    --------
    >>> graphlab.kcore.get_default_options()
    """
    _mt._get_metric_tracker().track('toolkit.graph_analytics.kcore.get_default_options')

    return _main.run('kcore_default_options', {})


class KcoreModel(_ModelBase):
    """
    A KcoreModel object contains a core ID for each vertex, and the total
    number of cores in the graph.

    The core ID of a vertex is a measure of its global centrality.

    The algorithms iteratively remove vertices that has less than :math:`k`
    neighbors **recursively**. The algorithm guarantees that at iteration
    :math:`k+1`, all vertices left in the graph will have at least :math:`k+1`
    neighbors.  The vertices removed at iteration :math:`k` is assigned with a
    core ID equal to :math:`k`.

    Below is a list of queryable fields for this model:

    +---------------+----------------------------------------------------+
    | Field         | Description                                        |
    +===============+====================================================+
    | core_id       | An SFrame with each vertex's core id               |
    +---------------+----------------------------------------------------+
    | graph         | A new SGraph with the core id as a vertex property |
    +---------------+----------------------------------------------------+
    | kmax          | The maximum core id assigned to any vertex         |
    +---------------+----------------------------------------------------+
    | kmin          | The minimum core id assigned to any vertex         |
    +---------------+----------------------------------------------------+
    | training_time | Total training time of the model                   |
    +---------------+----------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`graphlab.kcore.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    See Also
    --------
    create
    """
    def __init__(self, model):
        '''__init__(self)'''
        self.__proxy__ = model
        self.__model_name__ = "kcore"

    def _result_fields(self):
        ret = super(KcoreModel, self)._result_fields()
        ret["vertex core id"] = "SFrame. See m['core_id']"
        return ret

    def _setting_fields(self):
        ret = super(KcoreModel, self)._setting_fields()
        ret['minimum core id assigned to any vertex'] = 'kmin'
        ret['maximum core id assigned to any vertex '] = 'kmax'
        return ret


def create(graph, kmin=0, kmax=10, verbose=True):
    """
    Compute the K-core decomposition of the graph. Return a model object with
    total number of cores as well as the core id for each vertex in the graph.

    Parameters
    ----------
    graph : SGraph
        The graph on which to compute the k-core decomposition.

    kmin : int, optional
        Minimun core id. Vertices having smaller core id than `kmin` will be
        assigned with core_id = `kmin`.

    kmax : int, optional
        Maximun core id. Vertices having larger core id than `kmax` will be
        assigned with core_id=`kmax`.

    verbose : bool, optional
        If True, print progress updates.

    Returns
    -------
    out : KcoreModel

    References
    ----------
    - Alvarez-Hamelin, J.I., et al. (2005) `K-Core Decomposition: A Tool for the
      Visualization of Large Networks <http://arxiv.org/abs/cs/0504107>`_.

    Examples
    --------
    If given an :class:`~graphlab.SGraph` ``g``, we can create
    a :class:`~graphlab.kcore.KcoreModel` as follows:

    >>> g = graphlab.load_graph('http://snap.stanford.edu/data/email-Enron.txt.gz', format='snap')
    >>> kc = graphlab.kcore.create(g)

    We can obtain the ``core id`` corresponding to each vertex in the graph
    ``g`` using:

    >>> kcore_id = kc['core_id']     # SFrame

    We can add the new core id field to the original graph g using:

    >>> g.vertices['core_id'] = kc['graph'].vertices['core_id']

    Note that the task above does not require a join because the vertex
    ordering is preserved through ``create()``.

    See Also
    --------
    KcoreModel
    """
    _mt._get_metric_tracker().track('toolkit.graph_analytics.kcore.create')

    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')

    opts = {'graph': graph.__proxy__, 'kmin': kmin, 'kmax': kmax}
    params = _main.run('kcore', opts, verbose)

    return KcoreModel(params['model'])
