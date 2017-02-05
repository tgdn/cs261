import graphlab as _gl
import graphlab.connect as _mt

# Utils
from graphlab.util import _raise_error_if_not_of_type
from graphlab.toolkits._internal_utils import _toolkit_repr_print, \
                                              _precomputed_field, \
                                              _raise_error_if_not_sframe, \
                                              _check_categorical_option_type
from graphlab.toolkits._model import _get_default_options_wrapper
from graphlab.toolkits._model import SDKModel as _SDKModel

_DEFAULT_OPTIONS = {
    'min_support': 1,
    'max_patterns': 100,
    'min_length': 1,
}

get_default_options = _get_default_options_wrapper(
    '_FPGrowth', 'frequent_pattern_mining', 'FrequentPatternMiner', True)


def create(dataset, item, features=None, min_support=1, max_patterns=100,
           min_length=1):
    """
    Create a :class:`~graphlab.frequent_pattern_mining.FrequentPatternMiner` to
    extract the set of frequently occurring items in an event-series.

    Parameters
    ----------

    dataset : SFrame
        Dataset for training the model.

    item: string
        Name of the column containing the item. The values in this column must
        be of string or integer type.

    features : list[string], optional
        Names of the columns containing features. 'None' (the default) indicates
        that all columns except the target variable should be used as features.

        The feature columns are the ones that together identify a unique
        transaction ID for the item.

    min_support : int, optional
        The minimum number of times that a pattern must occur in order for it
        to be considered `frequent`.

    max_patterns : int, optional
        The maximum number of frequent patterns to be mined.

    min_length: int, optional
        The minimum size (number of elements in the set) of each pattern being
        mined.

    Returns
    -------
    out : FrequentPatternMiner
        A trained model of type
        :class:`~graphlab.frequent_pattern_mining.FrequentPatternMiner`.

    Notes
    -----
    Frequent closed itemests are mined using the `top-k FP growth` algorithm.
    Mining occurs until the top max_patterns closed itemsets of size min_length
    and support greater than min_support are found.

    See Also
    --------
    FrequentPatternMiner

    References
    ----------

    - Wikipedia - Association Rule Learning
      <https://en.wikipedia.org/wiki/Association_rule_learning>
    - Han, Jiawei, et al. "Mining top-k frequent closed patterns without minimum
      support." Data Mining, 2002. ICDM 2003.
    - Wang, Jianyong, et al. "TFP: An efficient algorithm for mining top-k
      frequent closed itemsets." Knowledge and Data Engineering, IEEE Transactions
      on 17.5 (2005): 652-663.

    Examples
    --------

    .. sourcecode:: python

        # Load the data
        >>> import graphlab as gl
        >>> bakery_sf = gl.SFrame("https://static.turi.com/datasets/extended-bakery/bakery.sf")
        >>> bakery_sf
        +---------+-------------+-------+----------+----------+-----------------+
        | Receipt |   SaleDate  | EmpId | StoreNum | Quantity |       Item      |
        +---------+-------------+-------+----------+----------+-----------------+
        |    1    | 12-JAN-2000 |   20  |    20    |    1     |  GanacheCookie  |
        |    1    | 12-JAN-2000 |   20  |    20    |    5     |     ApplePie    |
        |    2    | 15-JAN-2000 |   35  |    10    |    1     |   CoffeeEclair  |
        |    2    | 15-JAN-2000 |   35  |    10    |    3     |     ApplePie    |
        |    2    | 15-JAN-2000 |   35  |    10    |    4     |   AlmondTwist   |
        |    2    | 15-JAN-2000 |   35  |    10    |    3     |    HotCoffee    |
        |    3    |  8-JAN-2000 |   13  |    13    |    5     |    OperaCake    |
        |    3    |  8-JAN-2000 |   13  |    13    |    3     |   OrangeJuice   |
        |    3    |  8-JAN-2000 |   13  |    13    |    3     | CheeseCroissant |
        |    4    | 24-JAN-2000 |   16  |    16    |    1     |   TruffleCake   |
        +---------+-------------+-------+----------+----------+-----------------+
        [266209 rows x 6 columns]

        # Train a model.
        >>> model = gl.frequent_pattern_mining.create(train, 'Item',
                         features=['Receipt'], min_length=4, max_patterns=500)
    """
    _mt._get_metric_tracker().track('toolkit.frequent_pattern_mining.create')

    # Type checking.
    _raise_error_if_not_sframe(dataset, "dataset")
    _raise_error_if_not_of_type(item, str, "item")
    _raise_error_if_not_of_type(features, [list, type(None)], "features")
    _raise_error_if_not_of_type(min_support, [int, float], "min_support")
    _raise_error_if_not_of_type(max_patterns, [int, float], "max_patterns")
    _raise_error_if_not_of_type(min_length, [int, float], "min_length")

    # Value checking.
    column_names = dataset.column_names()

    # If features is None, then use all other column names than item
    if features is None:
        features = column_names
        features.remove(item)

    # Call the C++ create function.
    proxy = _gl.extensions._pattern_mining_create(
            dataset, item, features, min_support, max_patterns, min_length)
    return FrequentPatternMiner(proxy)


class FrequentPatternMiner(_SDKModel):
    """
    The FrequentPatternMiner extracts and analyzes 'closed' frequent patterns in
    itemset data using a modified FP-growth algorithm (TFP).

    This model cannot be constructed directly.  Instead, use
    :func:`graphlab.frequent_pattern_mining.create` to create an instance of
    this model. A detailed list of parameter options and code samples are
    available in the documentation for the create function.

    Examples
    --------

    .. sourcecode:: python

        >>> import graphlab as gl
        >>> bakery_sf = gl.SFrame("https://static.turi.com/datasets/extended-bakery/bakery.sf")

        # Create a train/test split
        >> train, test = bakery_sf.random_split(0.95)

        # Create a FrequentPatternMiner
        >>> model = gl.frequent_pattern_mining.create(train, 'Item',
                         features=['Receipt'], min_length=4, max_patterns=500)

        # Extract features
        >>> features = model.extract_features(test)

        # Predict items to add
        >>> predictions = model.predict(test)

    See Also
    --------
    create

    References
    ----------
    - Wikipedia - Association Rule Learning
      <https://en.wikipedia.org/wiki/Association_rule_learning>

    - Han, Jiawei, et al. "Mining top-k frequent closed patterns without minimum
      support." Data Mining, 2002. ICDM 2003.

    - Wang, Jianyong, et al. "TFP: An efficient algorithm for mining top-k
      frequent closed itemsets." Knowledge and Data Engineering, IEEE Transactions
      on 17.5 (2005): 652-663.

    """
    def __init__(self, model_proxy):
        self.__proxy__ = model_proxy

    def _get_wrapper(self):
        proxy_wrapper = self.__proxy__._get_wrapper()

        def model_wrapper(unity_proxy):
            model_proxy = proxy_wrapper(unity_proxy)
            return FrequentPatternMiner(model_proxy)
        return model_wrapper

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
            A list of section titles.
              The order matches that of the 'sections' object.
        """
        fields = [
            ("Min support", 'min_support'),
            ("Max patterns", 'max_patterns'),
            ("Min pattern length", 'min_length'),
        ]

        patterns_sf = self.get('frequent_patterns').topk('support')
        patterns = [(p['pattern'], _precomputed_field(p['support']))
                    for p in patterns_sf]
        section_titles = ['Model fields', 'Most frequent patterns']
        return ([fields, patterns], section_titles)

    def __repr__(self):
        """
        Print a string description of the model, when the model name is entered
        in the terminal.
        """

        (sections, section_titles) = self._get_summary_struct()

        return _toolkit_repr_print(self, sections, section_titles, width=30)

    def get(self, field):
        """
        Return the value of a given field. The list of all queryable fields is
        detailed below, and can be obtained programmatically with the
        :func:`~graphlab.frequent_pattern_mining.FrequentPatternMiner.list_fields`
        method.

        +------------------------+---------------------------------------------+
        |      Field             | Description                                 |
        +========================+=============================================+
        | features               | Feature column names                        |
        +------------------------+---------------------------------------------+
        | frequent_patterns      | Most frequent closed itemsets in the        |
        |                        | training data                               |
        +------------------------+---------------------------------------------+
        | item                   | Item column name                            |
        +------------------------+---------------------------------------------+
        | max_patterns           | Maximum number of itemsets to mine          |
        +------------------------+---------------------------------------------+
        | min_support            | Minimum number of transactions for an       |
        |                        | itemset to be frequent                      |
        +------------------------+---------------------------------------------+
        | num_examples           | Number of examples (transactions) in the    |
        |                        | dataset                                     |
        +------------------------+---------------------------------------------+
        | num_features           | Number of feature columns                   |
        +------------------------+---------------------------------------------+
        | num_frequent_patterns  | Number of frequent itemsets mined           |
        +------------------------+---------------------------------------------+
        | num_items              | Number of unique items in the training data |
        +------------------------+---------------------------------------------+
        | training_time          | Total time taken to mine the data           |
        +------------------------+---------------------------------------------+

        Parameters
        ----------
        field : string
            Name of the field to be retrieved.

        Returns
        -------
        out
            Value of the requested fields.

        See Also
        --------
        list_fields

        Examples
        --------

        .. sourcecode:: python

            >>> model['num_frequent_patterns']
            500
        """

        _mt._get_metric_tracker().track(
            'toolkits.frequent_pattern_mining.get')
        return self.__proxy__.get(field)

    def get_current_options(self):
        """
        Return a dictionary with the options used to define and train the model.

        Returns
        -------
        out : dict
            Dictionary with options used to define and train the model.

        See Also
        --------
        list_fields, get

        Examples
        --------
        >>> model.get_current_options()
        {'max_patterns': 1000, 'min_length': 1, 'min_support': 1}

        """

        _mt._get_metric_tracker().track(
                  'toolkits.frequent_pattern_mining.get_current_options')
        return self.__proxy__.get_current_options()

    def get_frequent_patterns(self):
        """
        Use the trained model to obtain the most frequent patterns in the
        training data. The patterns are returned as an SFrame with three
        columns: `id`, `count`, and `patterns`,

        Returns
        -------
        out : SFrame
            An SFrame with the most frequent patterns in the training data.

        See Also
        --------
        predict, predict_topk, extract_features

        Examples
        --------

        .. sourcecode:: python

            >>> model.get_frequent_patterns()

            Columns:
                    pattern list
                    support int

            Rows: 500

            Data:
            +-------------------------------+---------+
            |            pattern            | support |
            +-------------------------------+---------+
            | [CoffeeEclair, HotCoffee, ... |   1704  |
            | [LemonLemonade, LemonCooki... |   1565  |
            | [LemonLemonade, LemonCooki... |   1290  |
            | [LemonLemonade, RaspberryL... |   1289  |
            | [LemonLemonade, LemonCooki... |   1279  |
            | [LemonCookie, RaspberryLem... |   1279  |
            | [AppleTart, AppleDanish, A... |   1253  |
            | [LemonLemonade, LemonCooki... |   1221  |
            | [CherryTart, ApricotDanish... |    61   |
            | [CherryTart, ApricotDanish... |    55   |
            +-------------------------------+---------+
            [500 rows x 2 columns]
        """
        _mt._get_metric_tracker().track(
                      'toolkits.frequent_pattern_mining.predict')
        return self.__proxy__.get_frequent_patterns()

    def list_fields(self):
        """
        List the fields stored in the model, including data, model, and training
        options. Each field can be queried with the ``get`` method.

        Returns
        -------
        out : list
            List of fields queryable with the ``get`` method.

        See Also
        --------
        get

        Examples
        --------
        >>> fields = model.list_fields()
        """
        _mt._get_metric_tracker().track(
            'toolkits.frequent_pattern_mining.list_fields')
        return self.__proxy__.list_fields()

    def predict(self, dataset):
        """
        Use the trained model to obtain top predictions for the most
        confident rules given a partial set of observations described in the
        ``dataset``. Special case of predict_topk.

        Parameters
        ----------
        dataset : SFrame
            A dataset that has the same columns that were used during training.
            If the item column exists in ``dataset`` it will be ignored
            while making predictions.

        Returns
        -------
        out : SFrame
            An SFrame with the top scoring association rule for each itemset
            in ``dataset``.
            The SFrame contains a row for each unique transaction in ``dataset``
            Each row of the SFrame consists of the original ``features`` and

            * prefix - the 'antecedent' or 'left-hand side' of an assocation
              rule. It must be a frequent itemset and a subset of the
              associated itemset.
            * prediction - the 'consequent' or 'right-hand side' of the
              assocation rule. It must be disjoint of the prefix.
            * confidence - the confidence of the assocation rule defined as:
              ``confidence(prefix => prediction) = Support(prefix U prediction) / Support(prefix)``
            * prefix support - the frequency of the 'prefix' itemset in the
              training data
            * prediction support - the frequency of the 'prediction' itemset in
              the training data
            * joint support - the frequency of the cooccurance
              ('prefix' + 'prediction') in the training data

            If no valid association rule exists for an itemset, then ``predict``
            will return a row of Nones.


        See Also
        --------
        get_frequent_patterns, extract_features, predict_topk

        Notes
        -----
        Prediction can be slow when max_patterns is set to a large value because
        there are more rules to consider for predictions.

        References
        ----------

        - Wikipedia - Association Rule Learning
          <https://en.wikipedia.org/wiki/Association_rule_learning>

        - Han, Jiawei, Micheline Kamber, and Jian Pei. Data mining: concepts and
          techniques: concepts and techniques. Elsevier, 2011.

        Examples
        --------
        .. sourcecode:: python

            # Make predictions based on frequent patterns that occur together.
            >>> model.predict(test)

            Rows: 13283

            Columns:
                Receipt	int
                prefix	list
                prediction	list
                confidence	float
                prefix support	int
                joint support	int

            Rows: 9

            Data:
            +---------+-------------------+----------------------+----------------+----------------+
            | Receipt |       prefix      |      prediction      |   confidence   | prefix support |
            +---------+-------------------+----------------------+----------------+----------------+
            |    13   |    [CherrySoda]   |    [AppleDanish]     | 0.352077687444 |      4428      |
            |    42   |  [ChocolateTart]  | [VanillaFrappuccino] | 0.461889374644 |      5261      |
            |    26   |    [LemonTart]    |     [LemonCake]      | 0.464450600185 |      5415      |
            |    40   |  [LemonLemonade]  |  [RaspberryCookie]   | 0.394504818536 |      4877      |
            |    35   |   [AppleDanish]   |   [AppleCroissant]   | 0.391169154229 |      4824      |
            |    43   |         []        |    [CoffeeEclair]    | 0.104013695516 |     74769      |
            |    48   |  [ChocolateCake]  |  [ChocolateCoffee]   | 0.499831876261 |      5948      |
            |    9    | [RaspberryCookie] |   [LemonLemonade]    | 0.39950166113  |      4816      |
            |    27   | [CheeseCroissant] |    [OrangeJuice]     | 0.501453736959 |      5847      |
            +---------+-------------------+----------------------+----------------+----------------+
            +---------------+
            | joint support |
            +---------------+
            |      1559     |
            |      2430     |
            |      2515     |
            |      1924     |
            |      1887     |
            |      7777     |
            |      2973     |
            |      1924     |
            |      2932     |
            +---------------+

            # For a single itemset, e.g. ['HotCoffee', 'VanillaEclair']
            >>> new_itemset = gl.SFrame({'Receipt': [-1, -1],
                                         'Item': ['HotCoffee', 'VanillaEclair']})
            >>> model.predict(new_itemset)

            Data:
            +---------+-------------+--------------------+----------------+----------------+
            | Receipt |    prefix   |     prediction     |     score      | prefix support |
            +---------+-------------+--------------------+----------------+----------------+
            |    -1   | [HotCoffee] | [ApricotCroissant] | 0.344545454545 |      7700      |
            +---------+-------------+--------------------+----------------+----------------+
            +---------------+
            | joint support |
            +---------------+
            |      2653     |
            +---------------+
            [1 rows x 6 columns]
        """
        return self.predict_topk(dataset, k=1)

    def predict_topk(self, dataset, k=5):
        """
        Use the trained model to obtain top-k predictions for the most
        confident rules given a partial set of observations described in the
        ``dataset``.

        Parameters
        ----------
        dataset : SFrame
            A dataset that has the same columns that were used during training.
            If the item column exists in ``dataset`` it will be ignored
            while making predictions.

        k : int, optional
            Number of predictions to return for each input example.

        Returns
        -------
        out : SFrame
            An SFrame with the top scoring association rules for each itemset
            in the dataset.
            The SFrame contains a row for each unique transaction in ``dataset``
            Each row of the SFrame consists of the 'features' and

            * prefix - the 'antecedent' or 'left-hand side' of an assocation
              rule. It must be a frequent itemset and a subset of the
              associated itemset.
            * prediction - the 'consequent' or 'right-hand side' of the
              assocation rule. It must be disjoint of the prefix.
            * confidence - the confidence of the assocation rule defined as:
              ``confidence(prefix => prediction) = Support(prefix U prediction) / Support(prefix)``
            * prefix support - the frequency of the 'prefix' itemset in the
              training data
            * prediction support - the frequency of the 'prediction' itemset in
              the training data
            * joint support - the frequency of the cooccurance
              ('prefix' + 'prediction') in the training data

            If there does not exist ``k`` valid association rules for an
            itemset, then ``predict_topk`` will return as many valid rules
            as possible.


        See Also
        --------
        get_frequent_patterns, extract_features, predict

        Notes
        -----
        Prediction can be slow when max_patterns is set to a large value because
        there are more rules to consider for predictions.

        References
        ----------

        - Wikipedia - Association Rule Learning
          <https://en.wikipedia.org/wiki/Association_rule_learning>
        - Han, Jiawei, Micheline Kamber, and Jian Pei. Data mining: concepts and
          techniques: concepts and techniques. Elsevier, 2011.

        Examples
        --------
        .. sourcecode:: python

            # For an SFrame
            >>> predictions = model.predict(bakery_sf, k = 5)

            Columns:
                Receipt int
                prefix  list
                prediction  list
                confidence  float
                prefix support  int
                joint support   int

            Rows: 13283

            Data:
            +---------+-----------------+-------------------------------+-----------------+
            | Receipt |      prefix     |           prediction          |    confidence   |
            +---------+-----------------+-------------------------------+-----------------+
            |    13   |   [CherrySoda]  |         [AppleDanish]         |  0.352077687444 |
            |    13   |   [CherrySoda]  |          [AppleTart]          |  0.349593495935 |
            |    13   |   [CherrySoda]  |        [AppleCroissant]       |  0.349141824752 |
            |    13   |   [CherrySoda]  | [AppleCroissant, AppleDanish] |  0.302619692864 |
            |    13   |   [CherrySoda]  |  [AppleCroissant, AppleTart]  |  0.301942186089 |
            |    42   | [ChocolateTart] |      [VanillaFrappuccino]     |  0.461889374644 |
            |    42   | [ChocolateTart] |         [WalnutCookie]        |  0.367990876259 |
            |    42   | [ChocolateTart] | [WalnutCookie, VanillaFrap... |  0.323322562251 |
            |    42   |        []       |         [CoffeeEclair]        |  0.104013695516 |
            |    42   |        []       |          [HotCoffee]          | 0.0976340461956 |
            +---------+-----------------+-------------------------------+-----------------+
            +----------------+---------------+
            | prefix support | joint support |
            +----------------+---------------+
            |      4428      |      1559     |
            |      4428      |      1548     |
            |      4428      |      1546     |
            |      4428      |      1340     |
            |      4428      |      1337     |
            |      5261      |      2430     |
            |      5261      |      1936     |
            |      5261      |      1701     |
            |     74769      |      7777     |
            |     74769      |      7300     |
            +----------------+---------------+
            [13283 rows x 7 columns]

            # For a single itemset, e.g. ['HotCoffee', 'VanillaEclair']
            >>> new_itemset = gl.SFrame({'Receipt': [-1, -1],
                                         'Item': ['HotCoffee', 'VanillaEclair']})
            >>> model.predict(new_itemset, k = 3)

            Data:
            +---------+-------------+-------------------------------+----------------+----------------+
            | Receipt |    prefix   |           prediction          |     score      | prefix support |
            +---------+-------------+-------------------------------+----------------+----------------+
            |    -1   | [HotCoffee] |       [ApricotCroissant]      | 0.344545454545 |      7700      |
            |    -1   | [HotCoffee] |        [BlueberryTart]        | 0.341298701299 |      7700      |
            |    -1   | [HotCoffee] | [BlueberryTart, ApricotCro... | 0.31974025974  |      7700      |
            +---------+-------------+-------------------------------+----------------+----------------+
            +---------------+
            | joint support |
            +---------------+
            |      2653     |
            |      2628     |
            |      2462     |
            +---------------+
            [3 rows x 7 columns]
        """

        _mt._get_metric_tracker().track(
                      'toolkits.frequent_pattern_mining.predict_topk')
        _raise_error_if_not_sframe(dataset, "dataset")
        score_function = "confidence" # For now, we only support confidence
        return self.__proxy__.predict_topk(dataset, score_function, k)

    def extract_features(self, dataset):
        """
        Use the mined patterns to convert itemsets to binary vectors.

        For each itemset in ``dataset``, extract_features returns a vector of
        binary indicator variables, marking which mined patterns contain the
        itemset.

        Parameters
        ----------
        dataset : SFrame
            A dataset that has the same columns that were used during training.
            If the item column exists in ``dataset`` it will be ignored
            while making predictions.

        Returns
        -------
        out : SFrame
            An SFrame of extracted features.
            The SFrame contains a row for each unique transaction in ``dataset``
            Each row of the SFrame consists of the 'features' and
            * extracted_features - an array.array of binary indicator variables

        See Also
        --------
        predict

        Examples
        --------
        .. sourcecode:: python

            >>> features = model.extract_features(bakery_sf)
            >>> features
            Data:
            +---------+-------------------------------+
            | Receipt |       extracted_features      |
            +---------+-------------------------------+
            |  21855  | [0.0, 1.0, 0.0, 0.0, 0.0, ... |
            |  63664  | [0.0, 0.0, 0.0, 0.0, 0.0, ... |
            |   7899  | [0.0, 0.0, 0.0, 0.0, 0.0, ... |
            |  25263  | [0.0, 0.0, 0.0, 0.0, 0.0, ... |
            |  30621  | [0.0, 0.0, 0.0, 0.0, 0.0, ... |
            |  43116  | [0.0, 0.0, 0.0, 1.0, 0.0, ... |
            |  27112  | [0.0, 0.0, 0.0, 0.0, 1.0, ... |
            |  26319  | [0.0, 1.0, 0.0, 0.0, 0.0, ... |
            |  26439  | [0.0, 0.0, 0.0, 0.0, 0.0, ... |
            |  62361  | [0.0, 0.0, 0.0, 0.0, 0.0, ... |
            +---------+-------------------------------+
            [75000 rows x 2 columns]
        """
        _mt._get_metric_tracker().track(
            'toolkits.frequent_pattern_mining.extract_features')
        _raise_error_if_not_sframe(dataset, "dataset")
        return self.__proxy__.extract_features(dataset)

    @classmethod
    def _get_queryable_methods(cls):
        """
        Returns a list of method names that are queryable through
        Predictive Services
        """
        return {'predict': {'dataset': 'sframe'},
                'predict_topk': {'dataset': 'sframe'}}
