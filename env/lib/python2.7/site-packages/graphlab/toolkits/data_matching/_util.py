"""
Data matching utility functions.
"""

import graphlab as _gl
import graphlab.aggregate as _agg
from graphlab.toolkits._main import ToolkitError as _ToolkitError
import string
from operator import iadd
import copy
import logging

import sys
if sys.version_info.major == 3:
    from functools import reduce

def cleanse_string(x):
    """
    Preprocess a string, by converting all characters to lower case, removing
    punctuation, and deleting unnecessary white space. Intended for use in
    SFrame and SArray apply functions.

    Parameters
    ----------
    x : string
        Input string

    Returns
    -------
    out : string
        Transformed string.
    """
    out = x.lower()

    # remove punctuation
    if sys.version_info.major == 2:
        out = out.translate(None, string.punctuation)
    else:
        out = out.translate({ord(c): None for c in string.punctuation})

    out = out.strip()
    out = ' '.join(out.split())
    return out


def validate_composite_distance(distance, row_label, allowed_dists, verbose):
    """
    Validate a composite distance function, in terms of allowable types and
    lengths. Do not check if features are included in any particular dataset.

    Parameters
    ----------
    distance : list[list]
        Composite distance function. Each element of the inner list must include
        three items:
        - feature names, in a list or a tuple
        - distance function, as a string or function handle
        - weight, as an int or a float.

    row_label : string
        Name of the SFrame column with row labels.

    allowed_dists : list[string]
        Names of allowed component distance functions.

    verbose : bool
        If True, print output when the distance input is modified.

    Returns
    -------
    distance : list[list]
        A modified composite distance function. Tuples of feature names are
        converted to lists and the row label is removed from all feature lists.
    """
    row_label_flag = False

    ## Preliminary type checks
    if not isinstance(distance, list):
        raise TypeError("Composite distance specification must be a list.")

    if len(distance) == 0:
        raise ValueError("Composite distance specification is empty." +
                         "Composite distances must contain at least one " +
                         "distance component.")


    ## Component-wise checks
    for d in distance:
        
        ## Extract the three required items from each component, validating the
        #  number of items in the component along the way.
        try:
            ftr_names, dist, weight = d
        except:
            raise TypeError("Elements of a composite distance function must " +
                            "have three items: a set of feature names (tuple or list), " +
                            "a distance function (string or function handle), " +
                            "and a weight (int or float).")

        ## Convert *tuples* of feature names into *lists* of feature names.
        if isinstance(ftr_names, tuple):
            d[0] = list(ftr_names)

        ## Ensure feature names are now in a list
        if not isinstance(d[0], list):
            raise TypeError("Cannot process input feature set '{}'. ".format(ftr_names) + 
                "Feature names must be contained in a tuple or a list.")

        ## Remove row label
        d[0] = [x for x in ftr_names if x != row_label]
                
        if len(d[0]) < len(ftr_names):
            row_label_flag = True

        ## Check that there is still at least one valid feature name left
        if len(d[0]) == 0:
            raise ValueError("Distance components may not have empty feature " +
                             "lists.")

        ## For standard distances to be string names, but not function names
        ## - for GLC v1.4 this should be re-enabled, but in a sane way that
        #  still gives meaningful feedback to the user.
        if not isinstance(dist, str):
            raise TypeError("Input 'distance' not specified correctly. For the" +
                            " data matching toolkit, component distances must " +
                            "be strings.")

        ## Check that the component distance is one of the allowed strings
        if not dist in allowed_dists:
            raise ValueError("Input 'distance' not recognized. Distances " +
                             "specified by string, and elements of a composite " +
                             "distance, must be one of 'euclidean', " +
                             "'squared_euclidean', 'manhattan', jaccard', " +
                             "'weighted_jaccard', 'cosine', 'dot_product', " +
                             "'transformed_dot_product', or 'levenshtein'.")
                
        ## Check the type of the weight
        if not isinstance(weight, (int, float)):
            raise ValueError("The weight of each distance component must be " +
                             "a single integer or a float value.")


    ## Print relevant stuff for the user    
    if row_label_flag and verbose:
        logging.warning("Row label removed from the set of features in one or" +
            " more components of the distance function.")
            
    return distance


def validate_distance_feature_types(dataset, distance, allowed_types):
    """
    Check that the features passed to each standard distance function are
    allowed for that distance. NOTE: this function *does not* check that each
    distance function is one of the standard types; only that the feature types
    are correct if a distance function *is* standard.

    Parameters
    ----------
    dataset : SFrame
        Input dataset.

    distance : list[list]
        Composite distance.

    allowed_types : dict(string, list[type])
        Feature types allowed for each distance function.
    """

    for d in distance:
        ftr_names, dist, weight = d

        if dist in allowed_types.keys():
            for ftr in ftr_names:
                try:
                    ftr_type = dataset[ftr].dtype()
                except:
                    raise _ToolkitError("Feature '{}' could not be found in".format(ftr) +
                                        " the input dataset.")

                if not ftr_type in allowed_types[dist]:
                    raise TypeError("Feature '{}' is type '{}'".format(ftr, ftr_type.__name__) +
                                    " in the input dataset, which is not allowed " +
                                    "for distance function '{}'.".format(dist))


def extract_composite_features(distance):
    """
    Extract the (set) union of features specified in a composite distance
    function. Two lists are returned: one for features specified with the
    distance 'exact', and one for features specified with any other distance
    function.

    Parameters
    ----------
    distance : list[list[list[string], string, float]]
        Composite distance function. Each element in this list is a distance
        component, which consists of 1. a list of feature names, 2. a distance
        name, and 3. and floating point multiplier for the component.

    Returns
    -------
    out : list[string]
        List of feature names. This is the (set) union of all features specified
        in 'distance'.
    """
    return list(set(reduce(iadd, [x[0] for x in distance], [])))


def concatenate_sframes(datasets, features, row_label, sf_index_name='__sframe'):
    """
    Append multiple SFrames into a single SFrame.

    Parameters
    ----------
    datasets : dict(string: SFrame)
        Input datasets. Each SFrame in the list must include the columns listed
        in 'features', but may include extra columns (which will be discarded).

    features : list[string]
        List of features to extract from each SFrame. The output SFrame will
        have exactly these columns, plus a label (if specified).

    row_label : string
        Name of the column in each SFrame with row labels. If this column is not
        present in an input SFrame, the row index is used instead.

    sf_index_name : string, optional
        Name of the column in the output dataset that indicates which input
        dataset each row in the output comes from. 

    Returns
    -------
    out : SFrame
        ALl input datasets appended into a single dataset, subset to the
        specified features. If `row_label` is not specified, also includes a
        column called '__id', which indicates the row index of each output row
        in its original dataset. The output SFrame also includes a column
        indicating which original dataset each row comes from. If `datasets` is
        a list, this is the index of the SFrame in the list, but if `datasets`
        is a dictionary, then the entries in this column are the keys of the
        dict.
    """
    if not isinstance(row_label, str):
        raise TypeError("Input 'row_label' must be a string.")

    first_key_type = type(list(datasets.keys())[0])
    if not all([isinstance(x, first_key_type) for x in datasets.keys()]):
        raise ValueError("The names for each input dataset must have the same type.")

    sf_out = _gl.SFrame()

    for k, sf in datasets.items():

        ## Add a row number with name '__id' if a column doesn't already exist
        #  by the name 'row_label'. Otherwise, just copy the SFrame (to avoid
        #  mutating the originals).
        if not row_label in sf.column_names():
            sf_temp = sf.add_row_number(row_label)
        else:
            sf_temp = copy.copy(sf)

        ## Add SFrame label column
        sf_temp[sf_index_name] = _gl.SArray.from_const(k, len(sf_temp))

        ## Subset the rows of the current dataset.
        projection_features = list(set(features + [row_label, sf_index_name]))

        try:
            sf_out = sf_out.append(sf_temp[projection_features])
        except:
            raise ValueError("Input SFrames cannot be combined. Please ensure " + 
                             "that all input SFrames contain the specified " +
                             "features, and for each feature the type is "+
                             "the same in every input SFrame.")

    return sf_out


def concat_string_features(dataset, features, prefix='__concat.'):
    """
    Turn each column in `features` into a string and concatenate into a single
    column.

    Parameters
    ----------
    dataset : SFrame
        Input dataset.

    features : list[string]
        Names of columns in `dataset` to convert to string type and concatenate.

    prefix : string, optional
        Prefix to attach to the column name of the transformed features. The
        name of each input feature is concatenated to the name as well,
        separated by periods.

    Returns
    -------
    new_ftr : string
        Name of new concatenated feature.

    dataset : SFrame
        The input `dataset` plus the transformed feature column.
    """

    new_ftr = '__concat.' + '.'.join(features)
    dataset[new_ftr] = dataset.apply(
        lambda x: ' '.join([str(x[ftr]) for ftr in features]))

    return new_ftr, dataset


def string_features_to_dict(dataset, features, prefix='__dict.'):
    """
    Transform string columns into dictionaries, where each key is a 3-character
    shingle and the corresponding value is the number of times that shingle
    appears in a string.

    Parameters
    ----------
    dataset : SFrame
        Dataset with feature columns to transform.

    features : list[string]
        Names of columns in 'dataset' that will be converted to 3-character
        shingle count dictionaries (if they are string type).

    prefix : string, optional
        Prefix to attach to the column names of transformed feature.

    Returns
    -------
    new_features : list[string]
        Names of transformed feature columns.

    dataset : SFrame
        Input SFrame with new, transformed columns.
    """

    ## Get a map of column names to column types
    col_type_dict = {k: v for k, v in zip(dataset.column_names(), 
                                          dataset.column_types())}

    new_features = []

    for ftr in features:
        if col_type_dict[ftr] == str:
            new_ftr = prefix + ftr
            new_features.append(new_ftr)
            dataset[new_ftr] = _gl.text_analytics.count_ngrams(
                dataset[ftr], n=3, method='character', to_lower=False, 
                ignore_space=False)
        else:
            new_features.append(ftr)

    return new_features, dataset


def construct_exact_blocks(dataset, features):
    """
    Identify blocks of SFrame rows that have identical values for specified
    columns. First, rows with missing data in the blocking features are dropped;
    second, the SFrame is sorted according to the blocking features; and
    finally, the minimum and maximum row indices are recorded for each block.

    This allows faster retrieval of the rows in a given block than naive random
    access.

    Parameters
    ----------
    dataset : SFrame
        Input dataset.

    features : list[string]
        Names of columns in `dataset` to use for exact blocking.

    Returns
    -------
    dataset : SFrame
        The input `dataset`, sorted according to the specified `features`, plus
        a row index column for looking up the rows in the block.

    error_rows : SFrame
        Rows of the input `dataset` dropped due to missing data.

    blocks : SFrame
        Min and max row indices for each block in the (sorted) output `dataset`.
    """

    if len(features) > 0:
        dataset, error_rows = dataset.dropna_split(columns=features, how='any')   
        dataset = dataset.sort(features)
        dataset = dataset.add_row_number('__blocking_idx')
        blocks = dataset.groupby(features, _agg.COUNT,
                                 {'min_idx': _agg.MIN('__blocking_idx'),
                                 'max_idx': _agg.MAX('__blocking_idx')})

    else:
        dataset = dataset.add_row_number('__blocking_idx')
        blocks = _gl.SFrame({'Count': [dataset.num_rows()],
                            'min_idx': [0],
                            'max_idx': [dataset.num_rows() - 1]})
        error_rows = _gl.SFrame()

    return dataset, error_rows, blocks


_MAX_SIMILARITY_RADIUS = 1e15

def distances_to_similarity_scores(distance_fn, distances):
    """
    Convert distances to similarity scores.

    Parameters
    ----------
    distance_fn : str
        The name of the distance function.

    distances : SArray or SFrame
        An `SArray` or `SFrame` of distances to convert to similarity scores. If
        distances is an SFrame, it is expected to contain the following columns:
        "distance", "query_label", and "reference_label", of types float, str,
        and str respectively. If an SFrame is provided that does not contain
        these fields, a ToolkitError is raised.

    label : string
        Name of the label column.

    Returns
    -------
    out : SArray
        The converted similarity scores.

    Notes
    -----
    - To convert Levenshtein distances to similarities, the distances parameter
    must by an `SFrame`, since we require both of the strings being compared in
    order to normalize.
    """
    if not (isinstance(distances, _gl.SFrame) or \
            isinstance(distances, _gl.SArray)):
        raise TypeError("distances parameter is of type %s must be an SFrame " \
                        "or an SArray" % type(distances))

    if isinstance(distances, _gl.SFrame):
        column_names = distances.column_names()
        required_names = ["distance", "query_label", "reference_label"]
        if not all([name in column_names for name in required_names]):
            raise _ToolkitError("distances SFrame is missing required " \
                                "columns; at a minimum, it should have the " \
                                "following columns: \"distance\", " \
                                "\"query_label\", and \"reference_label\"")

    if isinstance(distances, _gl.SArray):
        if distance_fn == "levenshtein":
            raise TypeError("Expected an SFrame but got a an SArray")

        distances = _gl.SFrame({"distance": distances})

    def levenshtein_sim(dist, s1, s2):
        return 1 - dist / max(len(s1), len(s2))

    scores = None

    if distance_fn == "levenshtein" and isinstance(distances, _gl.SFrame):
        scores = distances.apply(
            lambda x: levenshtein_sim(
                x["distance"], x["query_label"], x["reference_label"]))
    elif distance_fn in ("jaccard", "weighted_jaccard", "cosine"):
        scores = distances["distance"].apply(lambda dist: 1 - dist)
    elif distance_fn in ("manhattan", "euclidean", "squared_euclidean"):
        scores = distances["distance"].apply(
            lambda dist: 1 - dist / _MAX_SIMILARITY_RADIUS)
    else:
        raise _ToolkitError("Unsupported distance function: %s" % distance_fn)

    return scores


def impute_numeric_means(dataset, features=None, impute_in_place=False):
    """
    Replace missing values in the ``features`` columns of ``dataset`` with the
    mean of the appropriate column. Only applies to numeric columns; string,
    dict, and array columns are ignored.

    Parameters
    ----------
    dataset : SFrame
        Input dataset.

    features : list[string], optional
        Features to transform by imputing the mean for missing values.

    Returns
    -------
    dataset : SFrame
        Identical to the input dataset, but missing values in the (numeric)
        columns named in ``features`` have missing values replaced by the column
        mean. Integer columns listed in ``features`` become float columns.
    """

    ## Get a map of column names to column types
    col_type_dict = {k: v for k, v in zip(dataset.column_names(), 
                                          dataset.column_types())}

    if not features:
        features = dataset.column_names()

    if not impute_in_place:
        dataset = copy.copy(dataset)

    ## Transform features by imputing missing values
    for ftr in features:
        if col_type_dict[ftr] in (int, float):
            if col_type_dict[ftr] == int:
                dataset[ftr] = dataset[ftr].astype(float)
            meanval = dataset[ftr].mean()
            if meanval is None:
                dataset[ftr] = dataset[ftr].fillna(0.0)
            else:
                dataset[ftr] = dataset[ftr].fillna(meanval)

    return dataset
