"""
Creation and abstract class methods for deduplication models, which match rows
within a set of SFrames to each other.
"""

from graphlab.toolkits._model import CustomModel as _CustomModel
from graphlab.toolkits._model import ProxyBasedModel as _ProxyBasedModel
from graphlab.toolkits._model import PythonProxy as _PythonProxy

import graphlab as _gl
import graphlab.connect as _mt
from graphlab.toolkits._internal_utils import _toolkit_repr_print


def get_default_options():
    """
    Return information about options for the top level deduplication tool.

    Returns
    -------
    out : SFrame
        Each row in the output SFrames correspond to a parameter, and includes
        columns for default values, lower and upper bounds, description ,and
        type.
    """
    out = _gl.SFrame({'name': ['distance', 'verbose']})
    out['default_value'] = ['euclidean', 'True']
    out['description'] = [
        'Function to compare dissimilarity of two records',
        'Verbose printing']
    out['lower_bound'] = [None, 0]
    out['upper_bound'] = [None, 1]
    out['parameter_type'] = ['String, function, or composite_distance',
                             'boolean']

    return out


def create(datasets, row_label=None, features=None, grouping_features=None,
           distance=None, verbose=True):
    """
    Create a model for deduplication of records in one or more SFrames.

    .. warning::

        The 'dot_product' distance is deprecated and will be removed in future
        versions of GraphLab Create. Please use 'transaformed_dot_product'
        distance instead, although note that this is more than a name change; it
        is a *different* transformation of the dot product of two vectors.
        Please see the distances module documentation for more details.

    Parameters
    ----------
    datasets : SFrame or list[SFrame] or dict(string: SFrame)
        Input datasets. Each SFrame in the list must include all of the features
        specified in the `features` or composite_params` parameters, but may
        have additional columns as well. SFrames can be input as values in a
        dictionary, where the keys are strings used in the output to identify
        the SFrame from which each record originated.

    row_label : string, optional
        Name of the SFrame column with row labels. If not specified, row numbers
        are used to identify rows in the output.

    features : list[string], optional
        Name of the columns with features to use in comparing records. 'None'
        (the default) indicates the intersection of columns over all SFrames in
        `datasets` should be used (except the label column, if specified). Each
        column can be one of the following types:

        - *Numeric*: values of numeric type integer or float.

        - *Array*: array of numeric (integer or float) values. Each array
          element is treated as a separate variable in the model.

        - *Dictionary*: key-value pairs with numeric (integer or float) values.
          Each key indicates a separate variable in the model.

        - *String*: string values.

        Please note: if `distance` is specified as a composite distance, then
        that parameter controls which features are used in the model. Any
        additional columns named in 'features' will be included in the model
        output but not used for distance computations.

    grouping_features : list[string], optional
            Names of features to use in grouping records before finding
            approximate matches. These columns must have string or integer type
            data. See the Notes section for more details on grouping.

    distance : string or list[list], optional
        Function to measure the distance between any two input data rows. This
        may be one of two types:

        - *String*: the name of a standard distance function. One of
          'euclidean', 'squared_euclidean', 'manhattan', 'levenshtein',
          'jaccard', 'weighted_jaccard', 'cosine', 'dot_product' (deprecated),
          or 'transformed_dot_product'.

        - *Composite distance*: the weighted sum of several standard distance
          functions applied to various features. This is specified as a list of
          distance components, each of which is itself a list containing three
          items:

          1. list or tuple of feature names (strings)

          2. standard distance name (string)

          3. scaling factor (int or float)

        For more information about GraphLab Create distance functions, please
        see the :py:mod:`~graphlab.toolkits.distances` module.

        For sparse vectors, missing keys are assumed to have value 0.0.

        If 'distance' is left unspecified or set to 'auto', a composite distance
        is constructed automatically based on feature types.

    verbose : bool, optional
        If True, print progress updates and model details.

    Returns
    -------
    out : A deduplication model.
        Currently only the `NearestNeighborDeduplication` model is implemented.

    See Also
    --------
    nearest_neighbor_deduplication.create, NearestNeighborDeduplication

    Notes
    -----
    - Standardizing features is often a good idea with distance-based methods,
      but this model does *not* standardize features.

    - For datasets with more than about 10,000 records, *grouping* (also known
      as *blocking*) is a critical step to avoid computing distances between all
      pairs of records. The grouping step simply assigns each record to a group
      that has identical values for all `grouping_features`, and only looks for
      duplicates within each group.

    - Records with missing data in the `grouping_features` are removed from
      consideration as duplicates. These records are given the entity label
      "None".

    - For features that all have the same type, the distance parameter may be a
      single standard distance function name (e.g. "euclidean"). In the model,
      however, all distances are first converted to composite distance
      functions; as a result, the 'distance' field in the model is always a
      composite distance.

    Examples
    --------
    >>> sf1 = graphlab.SFrame({'id': [0, 1, 2, 3],
    ...                        'x0': [0.5, 0.5, 0.3, 0.6],
    ...                        'x1': [1., 0.8, 0.6, 0.9],
    ...                        'city': ['seattle', 'olympia', 'boston', 'olympia'],
    ...                        'state': ['WA', 'WA', 'MA', 'WA']})
    ...
    ... # note: misspellings in the following dataset do not prevent correct
    ... # matches.
    >>> sf2 = graphlab.SFrame({'id': [9, 10],
    ...                        'x0': [0.35, 0.4],
    ...                        'x1': [0.65, 0.8],
    ...                        'city': ['bostan', 'seatle'],
    ...                        'state': ['MA', 'WA']})
    ...
    >>> dist = [[('city',), 'levenshtein', 2],
    ...         [('x0', 'x1'), 'euclidean', 1.5]]
    ...
    >>> m = graphlab.deduplication.create({'a': sf1, 'b': sf2}, row_label='id',
    ...                                    grouping_features=['state'],
    ...                                    distance=dist)
    ...
    >>> print m['entities']
    +----------+----+----------+------+-------+---------+------+
    | __sframe | id | __entity |  x0  | state |   city  |  x1  |
    +----------+----+----------+------+-------+---------+------+
    |    a     | 1  |    0     | 0.5  |   WA  | olympia | 0.8  |
    |    a     | 3  |    0     | 0.6  |   WA  | olympia | 0.9  |
    |    a     | 0  |    1     | 0.5  |   WA  | seattle | 1.0  |
    |    b     | 10 |    1     | 0.4  |   WA  |  seatle | 0.8  |
    |    a     | 2  |    2     | 0.3  |   MA  |  boston | 0.6  |
    |    b     | 9  |    2     | 0.35 |   MA  |  bostan | 0.65 |
    +----------+----+----------+------+-------+---------+------+
    [6 rows x 7 columns]
    """
    _mt._get_metric_tracker().track('{}.create'.format(__name__))

    m = _gl.nearest_neighbor_deduplication.create(datasets, row_label, features,
                                                  grouping_features, distance,
                                                  k=2, radius=None,
                                                  verbose=verbose)
    return m


class _Deduplication(_CustomModel, _ProxyBasedModel):
    """
    Parent class for GraphLab Create Deduplication models. This class lists
    methods common to all deduplication models, but leaves details up to each
    model for methods like predict.
    """

    def __str__(self):
        """
        Return a string description of the model to the ``print`` method.

        Returns
        -------
        out : string
            A description of the model.
        """
        return self.__repr__()

    def _get_summary_struct(self):
        """
        Returns a structured description of the model, including (where relevant)
        the schema of the training data, description of the training data,
        training statistics, and model hyperparameters.

        Returns
        -------
        sections : list (of list of tuples)
            A list of summary sections.
              Each section is a list.
                Each item in a section list is a tuple of the form:
                  ('<label>','<field>')
        section_titles: list
            A list of section names.
              The order matches that of the 'sections' object.
        """

        settings = [
            ('Number of input datasets', 'num_datasets'),
            ('Number of feature columns', 'num_features'),
            ('Included features', 'features'),
            ('Row Label','row_label'),
            ('Number of neighbors per point (k)', 'k'),
            ('Max distance to a neighbor (radius)', 'radius'),
            ('Number of entities', 'num_entities'),
        ]

        training = []
        training.append('Training time','training_time')

        return ([settings, training], ["Schema", "Training"])

    def __repr__(self):
        """
        Print a string description of the model when the model name is entered
        in the terminal.
        """
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=30)

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and create the
        current model instance.
        """
        raise NotImplementedError("The 'get_current_options' method has not been implemented for " +\
            "this model.")


