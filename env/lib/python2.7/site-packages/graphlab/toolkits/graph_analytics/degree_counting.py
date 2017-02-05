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
    Get the default options for :func:`graphlab.degree_counting.create`.

    Returns
    -------
    out : dict

    Examples
    --------
    >>> graphlab.degree_counting.get_default_options()
    """
    _mt._get_metric_tracker().track('toolkit.graph_analytics.degree_counting.get_default_options')

    return _main.run('degree_count_default_options', {})


class DegreeCountingModel(_ModelBase):
    """
    Model object containing the in degree, out degree and total degree for each vertex,

    Below is a list of queryable fields for this model:

    +---------------+------------------------------------------------------------+
    | Field         | Description                                                |
    +===============+============================================================+
    | graph         | A new SGraph with the degree counts as vertex properties   |
    +---------------+------------------------------------------------------------+
    | training_time | Total training time of the model                           |
    +---------------+------------------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`graphlab.degree_counting.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    See Also
    --------
    create
    """
    def __init__(self, model):
        '''__init__(self)'''
        self.__proxy__ = model


def create(graph, verbose=True):
    """
    Compute the in degree, out degree and total degree of each vertex.

    Parameters
    ----------
    graph : SGraph
        The graph on which to compute degree counts.

    verbose : bool, optional
        If True, print progress updates.

    Returns
    -------
    out : DegreeCountingModel

    Examples
    --------
    If given an :class:`~graphlab.SGraph` ``g``, we can create
    a :class:`~graphlab.degree_counting.DegreeCountingModel` as follows:

    >>> g = graphlab.load_graph('http://snap.stanford.edu/data/web-Google.txt.gz',
    ...                         format='snap')
    >>> m = graphlab.degree_counting.create(g)
    >>> g2 = m['graph']
    >>> g2
    SGraph({'num_edges': 5105039, 'num_vertices': 875713})
    Vertex Fields:['__id', 'in_degree', 'out_degree', 'total_degree']
    Edge Fields:['__src_id', '__dst_id']

    >>> g2.vertices.head(5)
    Columns:
        __id	int
        in_degree	int
        out_degree	int
        total_degree	int
    <BLANKLINE>
    Rows: 5
    <BLANKLINE>
    Data:
    +------+-----------+------------+--------------+
    | __id | in_degree | out_degree | total_degree |
    +------+-----------+------------+--------------+
    |  5   |     15    |     7      |      22      |
    |  7   |     3     |     16     |      19      |
    |  8   |     1     |     2      |      3       |
    |  10  |     13    |     11     |      24      |
    |  27  |     19    |     16     |      35      |
    +------+-----------+------------+--------------+

    See Also
    --------
    DegreeCountingModel
    """
    _mt._get_metric_tracker().track('toolkit.graph_analytics.degree_counting.create')

    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')

    params = _main.run('degree_count', {'graph': graph.__proxy__}, verbose)
    return DegreeCountingModel(params['model'])
