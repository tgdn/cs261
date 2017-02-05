"""
The GraphLab Create data matching toolkit links records that correspond to the
same real-world entity. It includes four tools: **record_linker**,
**deduplication**, **autotagger**, and **similarity_search**.

The **record linker** is the simplest data matching tool; it matches query
records from a tabular dataset to fixed dataset of reference records, also in
tabular form. ``record_linker`` uses a nearest neighbors search to match
records, but it extends the ``nearest_neighbors`` tool by automatically
performing basic feature engineering and composite distance construction where
applicable.

.. sourcecode:: python

     >>> homes = graphlab.SFrame({'sqft': [1230, 875, 1745],
    ...                          'street': ['phinney', 'fairview', 'cottage'],
    ...                          'city': ['seattle', 'olympia', 'boston'],
    ...                          'state': ['WA', 'WA', 'MA']})
    ...
    >>> model = graphlab.record_linker.create(homes, features=['city'],
    ...                                       distance='levenshtein')
    ...
    >>> queries = graphlab.SFrame({'sqft': [986, 1320],
    ...                            'street': ['fremont', 'phiney'],
    ...                            'city': ['sea', 'seattle'],
    ...                            'state': ['WA', 'WA']})
    ...
    >>> model.link(queries, k=2, radius=5.)
    +-------------+-----------------+----------+------+
    | query_label | reference_label | distance | rank |
    +-------------+-----------------+----------+------+
    |      0      |        0        |   4.0    |  1   |
    |      0      |        2        |   5.0    |  2   |
    |      1      |        0        |   0.0    |  1   |
    +-------------+-----------------+----------+------+

**Deduplication** also matches records from tabular datasets, but it has no
fixed reference dataset against which to match queries. Instead, all
records---possibly from many different SFrames---are processed and matched when
the ``deduplication`` model is created. The input records are aggregated and
stored in the model, along with an integer "entity" label: records with the same
entity label are considered duplicates by the model.

The
:meth:`~graphlab.data_matching.nearest_neighbor_deduplication.create` method
provides more detailed control over deduplication parameters.

.. sourcecode:: python

    >>> sf1 = graphlab.SFrame({'id': [0, 1, 2],
    ...                        'x0': [0.5, 0.5, 0.3],
    ...                        'x1': [1., 0.8, 0.6],
    ...                        'city': ['seattle', 'olympia', 'boston'],
    ...                        'state': ['WA', 'WA', 'MA']})
    ...
    ... # note: misspellings in the following dataset do not prevent correct
    ... # matches.
    >>> sf2 = graphlab.SFrame({'id': [9, 10],
    ...                 'x0': [0.35, 0.4],
    ...                 'x1': [0.65, 0.8],
    ...                 'city': ['bostan', 'seatle'],
    ...                 'state': ['MA', 'WA']})
    ...
    >>> dist = [[('state',), 'exact', 1],
    ...         [('city',), 'levenshtein', 2],
    ...         [('x0', 'x1'), 'euclidean', 1.5]]
    ...
    >>> m = graphlab.nearest_neighbor_deduplication.create({'a': sf1, 'b': sf2},
    ...                                                    row_label='id',
    ...                                                    distance=dist, k=None,
    ...                                                    radius=3)
    ...
    >>> print m['entities']
    +----------+----+----------+-------+------+---------+------+
    | __sframe | id | __entity | state |  x0  |   city  |  x1  |
    +----------+----+----------+-------+------+---------+------+
    |    a     | 1  |    0     |   WA  | 0.5  | olympia | 0.8  |
    |    a     | 0  |    1     |   WA  | 0.5  | seattle | 1.0  |
    |    b     | 10 |    1     |   WA  | 0.4  |  seatle | 0.8  |
    |    a     | 2  |    2     |   MA  | 0.3  |  boston | 0.6  |
    |    b     | 9  |    2     |   MA  | 0.35 |  bostan | 0.65 |
    +----------+----+----------+-------+------+---------+------+
    [5 rows x 7 columns]

Autotagging and similarity search are intended to be used with more complex
data. **Autotagging** makes it easier to navigate and understand the contents of
a large body of documents, by matching the documents to relevant tags from a
fixed reference set. For example, a large collection of product reviews may be
annotated with tags from a catalog of product names and descriptions.

.. sourcecode:: python

    >>> tags = graphlab.SFrame({'product': ['Canon PowerShot SX170',
    ...                                     'Canon EOS Rebel T3i',
    ...                                     'Nikon D3200'],
    ...                          'price': [129.0, 599.0, 369.]})
    ...
    ... ## Messy, unstructured text from merchant websites
    >>> reviews = graphlab.SFrame({
    ...    'text': ["The 16x Optical Zoom on the PowerShot SX170 IS camera gets you" +
    ...              "up close to kids and wildlife.",
    ...              "CANON - PowerShot SX170 IS 16 Megapixel, 16x IS Wide-Angle Optical" +
    ...              "Zoom, HD Video, 3.0 Inch LCD Screen Digital Camera - Black *FREE " +
    ...              "SHIPPING*",
    ...              "A great entry-level DX camera, the Nikon D3200shoots high" +
    ...              "resolution images while beingsimple-to-use for those unfamiliar with" +
    ...              "DSLRs."],
    ...     'merchant': ['wwww.beachcamera.com',
    ...                  'www.tristatecamera.com',
    ...                  'www.lensauthority.com']})
    ...
    >>> m = graphlab.autotagger.create(tags, tag_name='product')
    >>> product_tags = m.tag(reviews, query_name='text', similarity_threshold=0.5,
                             verbose=False)
    >>> print product_tags
    +---------+-------------------------------+-----------------------+
    | text_id |              text             |        product        |
    +---------+-------------------------------+-----------------------+
    |    0    | The 16x Optical Zoom on th... | Canon PowerShot SX170 |
    |    1    | CANON - PowerShot SX170 IS... | Canon PowerShot SX170 |
    |    2    | A great entry-level DX cam... |      Nikon D3200      |
    +---------+-------------------------------+-----------------------+
    +----------------+
    |     score      |
    +----------------+
    | 0.666666666667 |
    |      1.0       |
    |      0.8       |
    +----------------+
    [? rows x 4 columns]
    Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.
    You can use len(sf) to force materialization.

The **similarity_search** tool provides the ability to retrieve complex data
objects like images and text from a reference collection, based on object
similarity. For these types of complex objects, feature extraction is required
before the similarity can be measured. **similarity_search** encapsulates the
entire process and provides the ability for users to customize the various steps
in the process.

.. sourcecode:: python

    >>> sf1 = graphlab.SFrame({'id': [0, 1, 2],
    ...                        'image': [0.5, 0.5, 0.3]})

"""

__all__ = ['deduplication',
           'autotagger',
           'record_linker',
           'nearest_neighbor_deduplication',
           'nearest_neighbor_autotagger',
           'similarity_search'
           '_util']

from . import autotagger
from . import deduplication
from . import record_linker
from . import similarity_search
from . import nearest_neighbor_deduplication
from . import nearest_neighbor_autotagger
from . import _util