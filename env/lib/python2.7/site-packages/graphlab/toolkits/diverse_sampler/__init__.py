"""
The GraphLab Create diversity toolkit is used to select a diverse subset of
items from an SFrame or SGraph.

Every diverse sampling method uses a measure of intrinsic item quality as well
as a measure of similarity between items. The quality of an item is usually a
single real-valued feature, while the similarity function takes two sets of
features and computes some distance between the feature vectors. Using these two
measures, the diverse sampler returns a high-quality and diverse set (i.e. low
similarity between items in the set).

A :py:class:`~graphlab.diversity.diverse_sampler` instance can either take an
SFrame with a defined quality feature and a defined set of similarity features,
or an SGraph where the item similarities are defined along edges and the item
qualities are stored on vertices. After creating a diverse sampler, you can
choose to greedily stochasticly sample a diverse set. The greedy method will
select the optimal subset, while the stochastic version will return different
but diverse sets every time.

.. sourcecode:: python 

    >>> ground_set = graphlab.SFrame({'id': [0, 1, 2],
    ...                               'feature_1': [3, 1, 2],
    ...                               'feature_2': [[0, 1], [0, 1], [1, 0]]})
    >>> sampler = graphlab.diversity.diverse_sampler.create(data=ground_set,
    ...                                                     item_id='id',
    ...                                                     quality_feature='feature_1',
    ...                                                     similarity_features=['feature_2'])
    >>> sampler.sample(k=2)
    +-----------+------------+----+
    | feature_1 | feature_2  | id |
    +-----------+------------+----+
    |     2     | [0.0, 1.0] | 1  |
    |     1     | [1.0, 0.0] | 2  |
    +-----------+------------+----+

    >>> sampler.sample(k=2, greedy=True)
    +-----------+------------+----+
    | feature_1 | feature_2  | id |
    +-----------+------------+----+
    |     3     | [0.0, 1.0] | 0  |
    |     1     | [1.0, 0.0] | 2  |
    +-----------+------------+----+
"""

import graphlab as _gl

import graphlab.connect.main as glconnect
import graphlab.connect as _mt
from graphlab.toolkits._model import SDKModel as _SDKModel
from graphlab.util import _make_internal_url
from graphlab.util import _raise_error_if_not_of_type
from graphlab.toolkits._main import ToolkitError as _ToolkitError


def create(data, item_id=None, quality_feature=None, similarity_features=None):
    """
    Create a diverse sampler that samples from a given set of items.

    Parameters
    ----------
    data : SFrame or SGraph 
        Original reference data. The reference data must contain at least some
        item ID, as other features can be added by using side data with the
        sample() method. If the quality_feature or similarity_features exist in
        this reference data, then those features will be used when sampling if
        no side data is provided. This can be used to sample from some initial
        reference set.

        The diversity toolkit will either use a user- defined set of
        similarities when given an SGraph, or calculate them on-the- fly when
        given an SFrame. For an SGraph, the quality feature must be attached to
        each vertex as a field, while the similarities must be defined on the
        edges connecting vertices. For an SFrame, a single column will be used
        for item qualities, while a set of columns can be used to compute inter-
        item similarity.

    item_id : string, required 
        This is a unique identifier for each item in the list. An item_id is
        required, as it is possible to only sample from a subset of the
        reference set by providing a list of item_ids to the sample() method.

    quality_feature : optional 
        The single numeric feature to be used to sample high-quality items. In
        the case of an SFrame, this must be a column name. For an SGraph, this
        must be a vertex field. If no quality feature is specified, then the
        sampler will only consider item similarity when choosing items. In other
        words, the qualities of the items are set to the same value. TODO:
        implement this

    similarity_features : list[string], optional
        The set of features used to compute similarity between two items. The
        features are weighted equally. That is, the similarity between two
        single numeric features is weighted the same as the similarity between
        two vectors of numeric features. The similarity is computed using the
        method specified in the similarity_function keyword argument. In the
        case of an SFrame, the names must correspond to columns, while in an
        SGraph, they must correspond to edge fields.

        If no similarity features are given, then only the item qualities are
        considered when choosing items.

    Returns
    -------
    out: A DiverseSampler model.

    References (TODO)
    ----------
    - DPP for machine learning
    - Weighted vertex cover

    Examples
    --------
    A diverse sampler can be created directly from an SFrame.
    >>> sf = graphlab.SFrame.read_csv(
                'https://static.turi.com/datasets/auto-mpg/auto-mpg.csv')
    >>> sampler = graphlab.diverse_sampler.create(data=sf, 
                                                  item_id='name', 
                                                  quality_feature='accel', 
                                                  similarity_features=['mpg', 
                                                                       'displ', 
                                                                       'hp', 
                                                                       'weight'])

    You can also create a diverse sampler with an SGraph. Assuming you have some
    SGraph with an 'accel' vertex field on the vertices and a 'similarity' feature
    on the edges,

    >>> sampler = graphlab.diverse_sampler.create(data=sg, 
                                                  item_id='name', 
                                                  quality_feature='accel', 
                                                  similarity_features=['similarity'])

    Depending on which features are passed to the sampler, different methods will
    be selected as a default:

    - Random sampler: No quality or similarity features given
    - Quality-only sampler: Only quality feature specified
    - Vertex cover: Only similarity features specified
    - Weighted vertex cover: Quality and similarity features specified

    """
    proxy = _gl.extensions.diverse_sampler()
    sampler = DiverseSampler(data=data, 
                             item_id=item_id,
                             quality_feature=quality_feature,
                             similarity_features=similarity_features,
                             model_proxy = proxy)

    return sampler


class DiverseSampler(_SDKModel):

    def __init__(self, 
                 data=None,
                 item_id=None,
                 quality_feature=None,
                 similarity_features=None,
                 model_proxy = None,
                 _class = None):
        """ 
        Create a DiverseSampler object. This should never be called directly,
        because it is necessary to set up an SDK proxy prior to calling
        __init__.
        """
        if _class:
            self.__class__ = _class

        self._init_with_frame = False 

        self.__proxy__ = model_proxy
        self.__name__ = 'diverse_sampler'
        self._quality_feature = quality_feature
        self._similarity_features = similarity_features

        if data is None and model_proxy is None:
            raise ValueError("The diverse sampler must be initialized with a " +
                                             "reference SFrame or SGraph.")
        elif data is not None:
            if not (isinstance(data, _gl.SFrame) or isinstance(data, _gl.SGraph)):
                raise ValueError("Unknown data type " + str(type(data)) + ".")

        if item_id is None and model_proxy is None:
            # Note that for SGraphs, the __id vertex field is intrinsic to each
            # gl.Vertex, so we don't actually need to specify item_id
            if isinstance(data, _gl.SFrame):
                raise ValueError("An item_id must be specified.")

        if isinstance(data, _gl.SFrame):
            col_names = data.column_names()
        elif isinstance(data, _gl.SGraph):
            if similarity_features is not None and len(similarity_features) > 1:
                raise _ToolkitError("Only 1 similarity feature is supported for SGraph.")
            col_names = data.get_fields()

        if isinstance(data, _gl.SFrame) and item_id not in col_names:
            raise ValueError("Item ID "+item_id+" does not name " +
                                             "a column in the SFrame.")

        if quality_feature is not None and quality_feature not in col_names:
            raise ValueError("Quality feature "+quality_feature+" does not name " +
                                             "a column in the SFrame.")

        if similarity_features is not None:
            for sname in similarity_features:
                if sname not in col_names:
                    raise ValueError("Similarity feature "+sname+" does not name " +
                                                     "a column in the SFrame.")

        opts = dict()
        if item_id is None and isinstance(data, _gl.SGraph):
            item_id = "__id"
        opts["item_id"] = item_id

        if quality_feature is not None:
            opts["quality_feature"] = quality_feature
        if similarity_features is not None:
            opts["similarity_features"] = similarity_features

        if isinstance(data, _gl.SFrame):
            self._init_with_frame = True
            self.__proxy__.init_with_frame(data, opts)
        elif isinstance(data, _gl.SGraph):
            self._init_with_frame = False
            self.__proxy__.init_with_graph(data, opts)


    def sample(self, k, diversity=0.1, subset_ids=None, **kwargs):
        """
        After constructing a diverse sampler, sample a diverse set
        stochastically. The stochastic algorithm depends on the sampling method
        itself.

        Parameters
        ----------
        k : int
            The number of items to sample.

        diversity : double in [0, 1], optional 
            This is a tunable parameter that trades off between quality and
            diversity. A diversity factor of 0 will only consider quality when
            building a set, while a diversity factor of 1 will only consider
            item similarity and will ignore quality. A value between 0 and 1
            will force the algorithm to trade off between quality and diversity.
            Note that this keyword argument is only applicable if both quality
            and similarity features were passed to create().

            The actual effect of the diversity factor depends on the algorithm:
                - When the method is vertex cover or weighted vertex cover, the
                  diversity factor changes the number of nearest-neighbors to
                  remove when sampling an item. Specifically, the number of
                  neighbors is set to the value floor( (N-1)/(k-1) *
                  diversity_factor).


        subset_ids: SArray, optional
            A list of IDs to sample from. Sometimes you may wish to sample from
            only a subset of the original data - e.g., only provide a diverse
            sample of movies from a particular user's top recommendations.
            If subset_ids is empty, then the sampler will return subsets from
            the original SFrame or SGraph passed in with the data parameter used
            in create().


        **kwargs : optional
            Additional method-specific parameters for fine-tuning.

            - *greedy*: Use the greedy algorithm to generate a set. Instead of
              stochastically building a set based on a distribution, for each
              item, take the mode of the current distribution. For instance, if
              only quality features are being considered, using the greedy
              option will return the top-k items. Usually the greedy algorithm
              provides the highest-quality and most-diverse set, but for each
              set of items and algorithm, there is only one set that greedy can
              generate.


        Based on which features were given to create(), different sampling
        methods will be used. One of the four following algorithms are chosen
        based on the initial feature set.

        - *"random"*: If no quality or similarity features are given. Returns a
          completely random set of items, with no reference to item qualities or
          similarities. Note that the greedy method is undefined for a random
          sampler, so it is ignored.

        - *"quality-only"*: If only a quality feature are given. Generate a
          distribution over items based on their quality, and sample from this
          distribution. If greedy is specified, the top-k items in terms of
          quality are returned.

        - *"vertex-cover"*: If only similarity features are given. An internal
          graph is generated if an SFrame is given, and each item is connected
          to it's k-nearest neighbors, where k is determined by the diversity
          factor. When an item is sampled at random, its neighbors are removed
          from the candidate set. If an SGraph is given initially, all vertices
          connected to a sampled point are removed. Note that the greedy method
          for this algorithm is undefined, so it is ignored.

        - *"weighted_vertex_cover"*: The same as vertex cover, except each
          vertex has an associated quality field. When selecting the next point,
          it is sampled from a distribution over the remaining points'
          qualities. If greedy is specified, then the next point is the point
          with the highest quality in the remaining points.

        Examples
        --------
        Sample k items directly from the reference set passed in via create()
        with the default sampling methods:

        >>> cars = graphlab.SFrame.read_csv('https://static.turi.com/datasets/auto-mpg/auto-mpg.csv')
        >>> sampler = graphlab.diverse_sampler.create(data=cars, 
                                                      item_id='name', 
                                                      quality_feature='accel', 
                                                      similarity_features=['mpg', 
                                                      'displ', 
                                                      'hp', 
                                                      'weight',
                                                      'origin'])
        >>> sampler.sample(k=5)
        +-----+-----+-------+-----+--------+-------+----+--------+----------------+
        | mpg | cyl | displ |  hp | weight | accel | yr | origin |      name      |
        +-----+-----+-------+-----+--------+-------+----+--------+----------------+
        |  26 |  4  | 121.0 | 113 |  2234  |  12.5 | 70 |   2    |    bmw 2002    |
        |  18 |  6  | 232.0 | 100 |  2945  |  16.0 | 73 |   1    |   amc hornet   |
        |  24 |  4  | 116.0 |  75 |  2158  |  15.5 | 73 |   2    |   opel manta   |
        |  36 |  4  |  98.0 |  70 |  2125  |  17.3 | 82 |   1    | mercury lynx l |
        |  44 |  4  |  97.0 |  52 |  2130  |  24.6 | 82 |   2    |   vw pickup    |
        +-----+-----+-------+-----+--------+-------+----+--------+----------------+

        This method returns an SFrame (or SGraph, depending on what was used to
        create the sampler) containing the sampled items. If the diverse sampler
        was created with an SGraph, the sampler will return an SFrame containing
        the sampled vertices and their associated fields. 

        Instead of stochastic sampling, you can also force the algorithm to try
        to form the best possible set by using the greedy method:

        >>> sampler.sample(k=5, greedy=True)
        +-----+-----+-------+----+--------+-------+----+--------+-------------------------------+
        | mpg | cyl | displ | hp | weight | accel | yr | origin |              name             |
        +-----+-----+-------+----+--------+-------+----+--------+-------------------------------+
        |  19 |  4  | 120.0 | 88 |  3270  |  21.9 | 76 |   2    |          peugeot 504          |
        |  27 |  4  | 141.0 | 71 |  3190  |  24.8 | 79 |   2    |          peugeot 504          |
        |  23 |  8  | 260.0 | 90 |  3420  |  22.2 | 79 |   1    | oldsmobile cutlass salon b... |
        |  43 |  4  |  90.0 | 48 |  2335  |  23.7 | 80 |   2    |       vw dasher (diesel)      |
        |  44 |  4  |  97.0 | 52 |  2130  |  24.6 | 82 |   2    |           vw pickup           |
        +-----+-----+-------+----+--------+-------+----+--------+-------------------------------+

        In this example, two Peugeot cars were selected. Although they were
        somewhat different based on the original similarity features we
        specified, it's possible to get an even more diverse sample. To increase
        diversity, the "diversity" keyword (which can range between 0 and 1) can
        be increased. Larger values will favor reducing inter-item similarity
        (increasing diversity), while smaller values will favor high- quality
        items (decreasing diversity).

        >>> sampler.sample(k=5, diversity=0.8, greedy=True)
        +-----+-----+-------+-----+--------+-------+----+--------+-------------------------------+
        | mpg | cyl | displ |  hp | weight | accel | yr | origin |              name             |
        +-----+-----+-------+-----+--------+-------+----+--------+-------------------------------+
        |  27 |  4  |  97.0 |  60 |  1834  |  19.0 | 71 |   2    |      volkswagen model 111     |
        |  32 |  4  |  71.0 |  65 |  1836  |  21.0 | 74 |   3    |      toyota corolla 1200      |
        |  17 |  6  | 231.0 | 110 |  3907  |  21.0 | 75 |   1    |         buick century         |
        |  27 |  4  | 141.0 |  71 |  3190  |  24.8 | 79 |   2    |          peugeot 504          |
        |  23 |  8  | 260.0 |  90 |  3420  |  22.2 | 79 |   1    | oldsmobile cutlass salon b... |
        +-----+-----+-------+-----+--------+-------+----+--------+-------------------------------+

        Finally, if you want to restrict the reference set to a smaller subset,
        you can pass in a list of IDs with the "subset_ids" keyword:

        >>> ford_names = gl.SArray([n for n in cars['name'] if 'ford' in n])
        >>> sampler.sample(5, diversity=1.0, subset_ids=ford_names)
        +-----------------------+-----+-----+-------+-----+--------+-------+----+--------+
        |          name         | mpg | cyl | displ |  hp | weight | accel | yr | origin |
        +-----------------------+-----+-----+-------+-----+--------+-------+----+--------+
        | ford gran torino (sw) |  13 |  8  | 302.0 | 140 |  4294  |  16.0 | 72 |   1    |
        |     ford maverick     |  15 |  6  | 250.0 |  72 |  3158  |  19.5 | 75 |   1    |
        |      ford fiesta      |  36 |  4  |  98.0 |  66 |  1800  |  14.4 | 78 |   1    |
        |     ford escort 2h    |  29 |  4  |  98.0 |  65 |  2380  |  20.7 | 81 |   1    |
        |  ford fairmont futura |  24 |  4  | 140.0 |  92 |  2865  |  16.4 | 82 |   1    |
        +-----------------------+-----+-----+-------+-----+--------+-------+----+--------+
        """
        _raise_error_if_not_of_type(k, int)

        if subset_ids is not None:
            _raise_error_if_not_of_type(subset_ids, _gl.SArray)

        if diversity < 0.0 or diversity > 1.0:
            raise ValueError("The diversity parameter must be between 0.0 and 1.0.")

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        opts = dict()
        opts["diversity"] = diversity

        if "wvc_neighbors" in kwargs.keys():
            opts["num_neighbors"] = kwargs["wvc_neighbors"]
        if "greedy" in kwargs.keys():
            opts["greedy"] = kwargs["greedy"]

        if subset_ids is None:
            return self.__proxy__.sample_from_ground_set(k, opts)
        else:
            return self.__proxy__.sample_from_id_subset(k, subset_ids, opts)


    def evaluate(self, 
                 data,
                 methods=['average_similarity', 'average_quality', 'log_det']):
        """
        Objectively evaluate the quality and diversity of a data subset.

        There are several quantitaive measures of the quality and diversity of
        some set. This method provides three:
            - Average quality: The average over the quality features of each of
              the items in data.
            - Average similarity: The average of the pairwise similarities
              between every item in data.
            - Log-determinant: This simultaneously measures both the quality and
              diversity of a set. To measure the log-determinant of a given set,
              we first form the similarity matrix L, where a diagonal entry L_ii
              corresponds to the quality of item i, and an off diagonal entry
              L_ij corresponds to the similarity between items i and j. We then
              take the log of the determinant of this matrix. This type of
              matrix is also referred to as a Gramian matrix.

              The determinant of a Gramian matrix corresponds to the volume
              spanned by the vectors used to construct the matrix. If an item
              has a large quality, it corresponds to a longer vector, which will
              increase the volume (and determinant) of L. If two feature vectors
              are similar, then the volume decreases (because the vectors point
              in a similar direction), which correspondingly decreases the
              determinant. Thus, both quality and similarity are encapsulated by
              the log-determinant.

        Parameters
        ----------
        data: SFrame or SGraph
            The subset of data to evaluate.

        methods: list[string], {'average_similarity', 'average_quality', 'log_det'}
            The set of methods to measure. If methods is None, then all
            possible evaluation methods will be used.

        Returns
        -------
        out: dict
            Dictionary of values with keys corresponding to measurement types and values
            corresponding to the actual evaluation scores.

        Examples
        --------
        >>> cars = graphlab.SFrame.read_csv('https://static.turi.com/datasets/auto-mpg/auto-mpg.csv')
        >>> sampler = graphlab.diverse_sampler.create(data=cars, 
                                                      item_id='name', 
                                                      quality_feature='accel', 
                                                      similarity_features=['mpg', 
                                                      'displ', 
                                                      'hp', 
                                                      'weight',
                                                      'origin'])
        >>> 

        >>> sf_simple_dd = gl.SFrame({'id': [0, 1, 2],
                                      'q':  [10, 10, 10],
                                      's1': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]})
        >>> sampler = gl.diverse_sampler.create(data=sg_simple_dd,
                                                item_id='id',
                                                quality_feature='q',
                                                similarity_features=['s1'])
        >>> sf = sampler.sample(5, greedy=True, diversity=0.2)
        >>> sampler.evaluate(sf)
        {'log_det': 15.819720050211457, 'average_quality': 23.76, 
            'average_similarity': 0.999730969627407}
        """
        eval_frame = False
        if isinstance(data, _gl.SFrame):
            eval_frame = True
        elif not isinstance(data, _gl.SGraph):
            raise ValueError("Unknown data type " + str(type(data)) + ".")

        div_eval = _gl.extensions.diversity_eval()

        options = dict()
        options["eval_methods"] = methods

        if self._quality_feature is not None:
            options["quality_feature"] = self._quality_feature
        if self._similarity_features is not None:
            options["similarity_features"] = self._similarity_features

        if eval_frame:
            if not self._init_with_frame:
                raise _ToolkitError("Sampler initialized with SGraph, but eval "+ \
                                                        "was called with an SFrame.")
            return div_eval.evaluate_frame(data, options)
        else:
            if self._init_with_frame:
                raise _ToolkitError("Sampler initialized with SFrame, but eval "+ \
                                                        "was called with an SGraph.")
            return div_eval.evaluate_graph(data, options)

    def save(self, location):
        """
        Save the model. The model is saved as a directory which can then be
        loaded using the :py:func:`~graphlab.load_model` method.

        Note that the diverse_sampler stores the data internally, so you can
        save the model, then load it later and sample from the loaded model
        immediately.

        Parameters
        ----------
        location : string
            Target destination for the model. Can be a local path or remote URL.

        See Also
        ----------
        graphlab.load_model

        Examples
        ----------
        .. sourcecode:: python 
            >>> ground_set = graphlab.SFrame({'id': [0, 1, 2],
                                              'feature_1': [3, 1, 2],
                                              'feature_2': [[0, 1], [0, 1], [1, 0]]})
            >>> sampler = graphlab.diversity.diverse_sampler.create(data=ground_set,
                                                                    item_id='id',
                                                                    quality_feature='feature_1',
                                                                    similarity_features=['feature_2'])
            >>> sampler.save('my_sampler')
            >>> loaded_sampler = graphlab.load_model('my_sampler')
            >>> loaded_sampler.sample(k=2)
            +-----------+------------+----+
            | feature_1 | feature_2  | id |
            +-----------+------------+----+
            |     2     | [0.0, 1.0] | 1  |
            |     1     | [1.0, 0.0] | 2  |
            +-----------+------------+----+

        """
        _mt._get_metric_tracker().track(self.__class__.__module__ + '.save')
        return glconnect.get_unity().save_model(self.__proxy__,
                                                     _make_internal_url(location), self._get_wrapper())


    def _get_wrapper(self):
        """
        Utility function for save(). This should never be called manually.
        """
        _class = self.__class__
        proxy_wrapper = self.__proxy__._get_wrapper()
        def model_wrapper(unity_proxy):
            model_proxy = proxy_wrapper(unity_proxy)
            return DiverseSampler(model_proxy=model_proxy, _class=_class)
        return model_wrapper
