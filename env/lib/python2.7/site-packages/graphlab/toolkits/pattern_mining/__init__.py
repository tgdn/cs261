"""
The GraphLab Create pattern mining toolkit contains models for extracting
frequently occuring sets from event log data.

The FrequentPatternMiner extracts and analyzes 'closed' frequent patterns in
itemset data using a modified FP-growth algorithm (TFP). You can create a model
using the function :func:`graphlab.frequent_pattern_mining.create`. A detailed
list of parameter options and code samples are available in the documentation
for the create function.

.. sourcecode:: python

    >>> import graphlab as gl
    >>> bakery_sf = gl.SFrame("https://static.turi.com/datasets/extended-bakery/bakery.sf")
    >>> bakery_sf
    Data:
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

    # Create a FrequentPatternMiner
    >>> model = gl.frequent_pattern_mining.create(dataset=bakery_sf, item="Item",
            features=["Receipt"])

    >>> model
    class                         : FrequentPatternMiner

    Model fields
    ------------
    Min support                   : 1
    Max patterns                  : 1000
    Min pattern length            : 1

    Most frequent patterns
    ----------------------
    ['CoffeeEclair']              : 8193
    ['HotCoffee']                 : 7700
    ['TuileCookie']               : 7556
    ['CherryTart']                : 6987
    ['StrawberryCake']            : 6948
    ['ApricotDanish']             : 6943
    ['OrangeJuice']               : 6871
    ['GongolaisCookie']           : 6783
    ['MarzipanCookie']            : 6733
    ['BerryTart']                 : 6362

    # Extract features
    >>> features = model.extract_features(dataset=bakery_sf)
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

    # Predict items to add
    >>> predictions = model.predict(dataset=bakery_sf)
    >>> predictions
    Data:
    +---------+-------------------------------+-------------------+----------------+
    | Receipt |             prefix            |     prediction    |     score      |
    +---------+-------------------------------+-------------------+----------------+
    |  21855  |          [HotCoffee]          |   [CoffeeEclair]  | 0.307402597403 |
    |  63664  |               []              |   [CoffeeEclair]  |    0.10924     |
    |   7899  |        [ApricotDanish]        |    [CherryTart]   | 0.573527293677 |
    |  25263  |         [TruffleCake]         | [GongolaisCookie] | 0.534046692607 |
    |  30621  | [RaspberryCookie, Raspberr... |  [LemonLemonade]  | 0.928811928812 |
    |  43116  |          [CherryTart]         |  [ApricotDanish]  | 0.569915557464 |
    |  27112  |        [StrawberryCake]       |   [NapoleonCake]  | 0.465745538284 |
    |  26319  | [ChocolateCoffee, Chocolat... |    [CasinoCake]   | 0.758098698153 |
    |  26439  |          [CasinoCake]         |  [ChocolateCake]  | 0.473693565588 |
    |  62361  |          [OperaCake]          |    [CherryTart]   | 0.528341724866 |
    +---------+-------------------------------+-------------------+----------------+
    [75000 rows x 7 columns]

"""
from . import frequent_pattern_mining
