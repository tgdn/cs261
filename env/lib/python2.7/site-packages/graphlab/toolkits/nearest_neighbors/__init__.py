"""
The GraphLab Create nearest neighbors toolkit finds the rows in a tabular
reference dataset that are most similar to a set of queries with the same
schema.

A :py:class:`~graphlab.nearest_neighbors.NearestNeighborsModel` is created with
a reference dataset contained in an :class:`~graphlab.SFrame`, a distance
function, and an indexing method (the latter two options can be done
automatically by the model). An instantiated model has two key methods:
**query**, for finding the closest points in the reference dataset to *new* data
points; and **similarity_graph**, for finding the nearest neighbors of each
point in the original reference set.

.. sourcecode:: python

    >>> references = graphlab.SFrame({'x1': [0.98, 0.62, 0.11],
    ...                               'x2': [0.69, 0.58, 0.36]})
    >>> references.print_rows()
    +------+------+
    |  x1  |  x2  |
    +------+------+
    | 0.98 | 0.69 |
    | 0.62 | 0.58 |
    | 0.11 | 0.36 |
    +------+------+
    [3 rows x 2 columns]
    ...
    >>> model = graphlab.nearest_neighbors.create(references)
    ...
    >>> sim_graph = model.similarity_graph(k=1)
    >>> sim_graph.show(vlabel='__id')
    >>> sim_graph.edges
    +----------+----------+----------------+------+
    | __src_id | __dst_id |    distance    | rank |
    +----------+----------+----------------+------+
    |    0     |    1     | 0.376430604494 |  1   |
    |    2     |    1     | 0.55542776308  |  1   |
    |    1     |    0     | 0.376430604494 |  1   |
    +----------+----------+----------------+------+
    ...
    >>> queries = graphlab.SFrame({'x1': [0.05, 0.61, 0.99],
    ...                            'x2': [0.06, 0.97, 0.86]})
    >>> queries.print_rows()
    +------+------+
    |  x1  |  x2  |
    +------+------+
    | 0.05 | 0.06 |
    | 0.61 | 0.97 |
    | 0.99 | 0.86 |
    +------+------+
    [3 rows x 2 columns]
    ...
    >>> model.query(queries, k=2)
    +-------------+-----------------+----------------+------+
    | query_label | reference_label |    distance    | rank |
    +-------------+-----------------+----------------+------+
    |      0      |        2        | 0.305941170816 |  1   |
    |      0      |        1        | 0.771556867638 |  2   |
    |      1      |        1        | 0.390128184063 |  1   |
    |      1      |        0        | 0.464004310325 |  2   |
    |      2      |        0        | 0.170293863659 |  1   |
    |      2      |        1        | 0.464004310325 |  2   |
    +-------------+-----------------+----------------+------+

In addition to the API documentation, please see the data science `Gallery
<https://turi.com/learn/gallery>`_, `How-tos <https://turi.com/learn/how-to>`_,
and the `nearest neighbors chapter of the User Guide
<https://turi.com/learn/userguide/nearest_neighbors/nearest_neighbors.html>`_
for more details and extended examples.
"""

from ._nearest_neighbors import create
from ._nearest_neighbors import get_default_options
from ._nearest_neighbors import NearestNeighborsModel
