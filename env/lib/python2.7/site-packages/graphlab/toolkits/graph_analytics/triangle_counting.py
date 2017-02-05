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
    Get the default options for :func:`graphlab.triangle_counting.create`.

    Returns
    -------
    out : dict

    Examples
    --------
    >>> graphlab.triangle_counting.get_default_options()
    """
    _mt._get_metric_tracker().track('toolkit.graph_analytics.triangle_counting.get_default_options')

    return _main.run('triangle_counting_default_options', {})


class TriangleCountingModel(_ModelBase):
    """
    Model object containing the traingle count for each vertex, and the total
    number of triangles. The model ignores the edge directions in that
    it assumes there are no multiple edges between
    the same source ang target pair and ignores bidirectional edges.

    The triangle count of individual vertex characterizes the importance of the
    vertex in its neighborhood. The total number of triangles characterizes the
    density of the graph. It can also be calculated using

    >>> m['triangle_count']['triangle_count'].sum() / 3.

    Below is a list of queryable fields for this model:

    +---------------+------------------------------------------------------------+
    | Field         | Description                                                |
    +===============+============================================================+
    | triangle_count| An SFrame with each vertex's id and triangle count         |
    +---------------+------------------------------------------------------------+
    | num_triangles | Total number of triangles in the graph                     |
    +---------------+------------------------------------------------------------+
    | graph         | A new SGraph with the triangle count as a vertex property  |
    +---------------+------------------------------------------------------------+
    | training_time | Total training time of the model                           |
    +---------------+------------------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`graphlab.triangle_counting.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    See Also
    --------
    create
    """
    def __init__(self, model):
        '''__init__(self)'''
        self.__proxy__ = model

    def _result_fields(self):
        ret = super(TriangleCountingModel, self)._result_fields()
        ret['total number of triangles'] = self['num_triangles']
        ret["vertex triangle count"] = "SFrame. See m['triangle_count']"
        return ret


def create(graph, verbose=True):
    """
    Compute the number of triangles each vertex belongs to, ignoring edge
    directions. A triangle is a complete subgraph with only three vertices.
    Return a model object with total number of triangles as well as the triangle
    counts for each vertex in the graph.

    Parameters
    ----------
    graph : SGraph
        The graph on which to compute triangle counts.

    verbose : bool, optional
        If True, print progress updates.

    Returns
    -------
    out : TriangleCountingModel

    References
    ----------
    - T. Schank. (2007) `Algorithmic Aspects of Triangle-Based Network Analysis
      <http://digbib.ubka.uni-karlsruhe.de/volltexte/documents/4541>`_.

    Examples
    --------
    If given an :class:`~graphlab.SGraph` ``g``, we can create a
    :class:`~graphlab.traingle_counting.TriangleCountingModel` as follows:

    >>> g =
    >>> graphlab.load_graph('http://snap.stanford.edu/data/email-Enron.txt.gz',
            >>> format='snap') tc = graphlab.triangle_counting.create(g)

    We can obtain the number of triangles that each vertex in the graph ``g``
    is present in:

    >>> tc_out = tc['triangle_count']  # SFrame

    We can add the new "triangle_count" field to the original graph g using:

    >>> g.vertices['triangle_count'] = tc['graph'].vertices['triangle_count']

    Note that the task above does not require a join because the vertex
    ordering is preserved through ``create()``.

    See Also
    --------
    TriangleCountingModel
    """
    _mt._get_metric_tracker().track('toolkit.graph_analytics.triangle_counting.create')

    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')

    params = _main.run('triangle_counting', {'graph': graph.__proxy__}, verbose)
    return TriangleCountingModel(params['model'])
