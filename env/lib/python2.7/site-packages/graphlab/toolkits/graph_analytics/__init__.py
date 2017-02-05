"""
The graph analytics toolkit contains methods for analyzing a
:class:`~graphlab.SGraph`. Each method takes an input graph and returns a model
object, which contains the training time, an :class:`~graphlab.SFrame` with the
desired output for each vertex, and a new graph whose vertices contain the
output as an attribute. Note that all of the methods in the graph analytics
toolkit are available from the top level graphlab import.

For this example we download a dataset of James Bond characters to an
SFrame, then build the SGraph.

.. sourcecode:: python

    >>> from graphlab import SFrame, SGraph
    >>> url = 'https://static.turi.com/datasets/bond/bond_edges.csv'
    >>> data = SFrame.read_csv(url)
    >>> g = SGraph().add_edges(data, src_field='src', dst_field='dst')

The :py:func:`degree_counting.create()
<graphlab.degree_counting.create>` method computes the inbound, outbound,
and total degree for each vertex.

.. sourcecode:: python

    >>> from graphlab import degree_counting
    >>> deg = degree_counting.create(g)
    >>> deg_graph = deg['graph'] # a new SGraph with degree data attached to each vertex
    >>> in_degree = deg_graph.vertices[['__id', 'in_degree']]
    >>> out_degree = deg_graph.vertices[['__id', 'out_degree']]

The :py:func:`connected_components.create()
<graphlab.connected_components.create>` method finds all weakly connected
components in the graph and returns a
:py:class:`ConnectedComponentsModel<graphlab.connected_components.ConnectedComponentsModel>`.
A connected component is a group of vertices such that
there is a path between each vertex in the component and all other vertices in
the group. If two vertices are in different connected components there is no
path between them.

.. sourcecode:: python

    >>> from graphlab import connected_components
    >>> cc = connected_components.create(g)
    >>> print cc.summary()
    >>> print cc.list_fields()

    >>> cc_ids = cc.get('component_id')  # an SFrame
    >>> cc_ids = cc['component_id']      # equivalent to the above line
    >>> cc_graph = cc['graph']

The :py:func:`shortest_path.create() <graphlab.shortest_path.create>`
method finds the shortest directed path distance from a single source vertex to
all other vertices and returns a
:py:class:`ShortestPathModel<graphlab.shortest_path.ShortestPathModel>`.
The output model contains this information and a method to
retrieve the actual path to a particular vertex.

.. sourcecode:: python

    >>> from graphlab import shortest_path
    >>> sp = shortest_path.create(g, source_vid=123)
    >>> sp_sframe = sp['distance']
    >>> sp_graph = sp['graph']
    >>> path = sp.get_path('98')

The :py:func:`triangle_counting.create() <graphlab.triangle_counting.create>`
counts the number of triangles in the graph and for each vertex and returns
a :py:class:`TriangleCountingModel<graphlab.triangle_counting.TriangleCountingModel>`.
A graph triangle is a complete subgraph of three vertices. The number of
triangles to which a vertex belongs is an indication of the connectivity of the
graph around that vertex.

.. sourcecode:: python

    >>> from graphlab import triangle_counting
    >>> tc = triangle_counting.create(g)
    >>> tc_out = tc['triangle_count']

The :py:func:`pagerank.create() <graphlab.pagerank.create>` method computes
the pagerank for each vertex and returns a :py:class:`PagerankModel<graphlab.pagerank.PagerankModel>`.
The pagerank value indicates the centrality of each node in the graph.

.. sourcecode:: python

    >>> from graphlab import pagerank
    >>> pr = pagerank.create(g)
    >>> pr_out = pr['pagerank']

The :py:func:`label_propagation.create() <graphlab.label_propagation.create>`
method computes the label probability for the vertices with unobserved labels
by propagating information from the vertices with observed labels along the edges.
The method returns a :py:class:`LabelPropagationModel<graphlab.label_propagation.LabelPropagationModel>`,
which contains the probability that a vertex belongs to each of the label classes.

.. sourcecode:: python

    >>> from graphlab import label_propagation
    >>> import random
    >>> def init_label(vid):
    ...     x = random.random()
    ...     if x > 0.9:
    ...         return 0
    ...     elif x < 0.1:
    ...         return 1
    ...     else:
    ...         return None
    >>> g.vertices['labels'] = g.vertices['__id'].apply(init_label, int)
    >>> m = label_propagation.create(g)
    >>> labels = m['labels']


The K-core decomposition recursively removes vertices from the graph with degree
less than k. The value of k where a vertex is removed is its core ID; the
:py:func:`kcore.create() <graphlab.kcore.create>` method returns
a :py:class:`KcoreModel<graphlab.kcore.KcoreModel>` which contains the core ID for
every vertex in the graph.

.. sourcecode:: python

    >>> from graphlab import kcore
    >>> kc = kcore.create(g)
    >>> kcore_id = kc['core_id']

Graph coloring assigns each vertex in the graph to a group in such a way that no
two adjacent vertices share the same group.
:py:func:`graph_coloring.create() <graphlab.graph_coloring.create>` method returns
a :py:class:`GraphColoringModel<graphlab.graph_coloring.GraphColoringModel>` which contains
the color group ID for every vertex in the graph.

.. sourcecode:: python

    >>> from graphlab import graph_coloring
    >>> color = graph_coloring.create(g)
    >>> color_id = color['color_id']
    >>> num_colors = color['num_colors']

For more information on the models in the graph analytics toolkit, plus extended
examples, please see the model definitions and create methods in the API
documentation, as well as the data science `Gallery
<https://turi.com/learn/gallery/>`_, the `How-Tos
<https://turi.com/learn/how-to/>`_, and the `graph analytics chapter of
the User Guide
<https://turi.com/learn/userguide/graph_analytics/intro.html>`_.
"""

'''
Copyright (C) 2016 Turi
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license. See the TURI-PYTHON-LICENSE file for details.
'''


from . import pagerank
from . import triangle_counting
from . import connected_components
from . import kcore
from . import graph_coloring
from . import shortest_path
from . import degree_counting
from . import label_propagation
