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
    Get the default options for :func:`graphlab.graph_coloring.create`.

    Returns
    -------
    out : dict

    Examples
    --------
    >>> graphlab.graph_coloring.get_default_options()
    """
    _mt._get_metric_tracker().track('toolkit.graph_analytics.graph_coloring.get_default_options')
    return _main.run('graph_coloring_default_options', {})


class GraphColoringModel(_ModelBase):
    """
    A GraphColoringModel object contains color ID assignments for each vertex
    and the total number of colors used in coloring the entire graph.

    The coloring is the result of a greedy algorithm and therefore is not
    optimal.  Finding optimal coloring is in fact NP-complete.

    Below is a list of queryable fields for this model:

    +----------------+-----------------------------------------------------+
    | Field          | Description                                         |
    +================+=====================================================+
    | graph          | A new SGraph with the color id as a vertex property |
    +----------------+-----------------------------------------------------+
    | training_time  | Total training time of the model                    |
    +----------------+-----------------------------------------------------+
    | num_colors     | Number of colors in the graph                       |
    +----------------+-----------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`graphlab.graph_coloring.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.


    See Also
    --------
    create
    """
    def __init__(self, model):
        '''__init__(self)'''
        self.__proxy__ = model
        self.__model_name__ = "graph_coloring"

    def _result_fields(self):
        ret = super(GraphColoringModel, self)._result_fields()
        ret['number of colors in the graph'] = self['num_colors']
        ret['vertex color id'] = "SFrame. See m['color_id']"
        return ret


def create(graph, verbose=True):
    """
    Compute the graph coloring. Assign a color to each vertex such that no
    adjacent vertices have the same color. Return a model object with total
    number of colors used as well as the color ID for each vertex in the graph.
    This algorithm is greedy and is not guaranteed to find the **minimum** graph
    coloring. It is also not deterministic, so successive runs may return
    different answers.

    Parameters
    ----------
    graph : SGraph
        The graph on which to compute the coloring.

    verbose : bool, optional
        If True, print progress updates.

    Returns
    -------
    out : GraphColoringModel

    References
    ----------
    - `Wikipedia - graph coloring <http://en.wikipedia.org/wiki/Graph_coloring>`_

    Examples
    --------
    If given an :class:`~graphlab.SGraph` ``g``, we can create
    a :class:`~graphlab.graph_coloring.GraphColoringModel` as follows:

    >>> g = graphlab.load_graph('http://snap.stanford.edu/data/email-Enron.txt.gz', format='snap')
    >>> gc = graphlab.graph_coloring.create(g)

    We can obtain the ``color id`` corresponding to each vertex in the graph ``g``
    as follows:

    >>> color_id = gc['color_id']  # SFrame

    We can obtain the total number of colors required to color the graph ``g``
    as follows:

    >>> num_colors = gc['num_colors']

    See Also
    --------
    GraphColoringModel
    """
    _mt._get_metric_tracker().track('toolkit.graph_analytics.graph_coloring.create')

    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')

    params = _main.run('graph_coloring', {'graph': graph.__proxy__}, verbose)
    return GraphColoringModel(params['model'])
