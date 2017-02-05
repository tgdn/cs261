import random as _random
import six as _six
import sys as _sys

import graphlab as _gl
from graphlab import aggregate as _aggregate
from .. import _sframe_base_view
from graphlab.toolkits._internal import search as _search
from graphlab.util import _raise_error_if_not_of_type
from graphlab.toolkits._main import ToolkitError as _ToolkitError
from graphlab.toolkits._model import Model as _Model
from graphlab.toolkits._model import SDKModel as _SDKModel
from graphlab.toolkits._internal_utils import \
        _raise_error_if_not_sframe, _raise_error_if_column_exists

_NoneType = type(None)

def _raise_error_if_column_missing(colname, sf, sf_name):
    if colname is not None and sf is not None:
        if colname not in sf.column_names():
            raise ValueError("Expected a column named {0} in the provided {1}.".format(colname, sf_name))


class RecommenderExploreView(_sframe_base_view.SFrameBaseView):

    def __init__(self, model,
            observation_data=None,
            user_data=None,
            item_data=None,
            user_name_column=None,
            user_description_column=None,
            user_url_column=None,
            item_name_column=None,
            item_description_column=None,
            item_url_column=None,
            random_seed=None):
        """
        Create a visualization of a RecommenderModel that helps explore and
        qualitatively evaluate the recommendations made by the model.

        We recommend not constructing this object directly, but instead
        using the method model.views.explore().
        """

        super(RecommenderExploreView, self).__init__(
            name = 'Recommender Explore',
            label = 'Explore',
            tag_name = 'gl-recommender-explore',
            description = 'Explore view for recommender models'
        )
        self._model = model


        _raise_error_if_not_of_type(model, [_Model], 'model')
        _raise_error_if_not_of_type(observation_data, [_NoneType, _gl.SFrame], "observation_data")
        _raise_error_if_not_of_type(user_data, [_NoneType, _gl.SFrame], "user_data")
        _raise_error_if_not_of_type(item_data, [_NoneType, _gl.SFrame], "item_data")
        _raise_error_if_not_of_type(user_name_column, [_NoneType, str], 'user_name_column')
        _raise_error_if_not_of_type(user_description_column, [_NoneType, str], 'user_description_column')
        _raise_error_if_not_of_type(user_url_column, [_NoneType, str], 'user_url_column')
        _raise_error_if_not_of_type(item_name_column, [_NoneType, str], 'item_name_column')
        _raise_error_if_not_of_type(item_description_column, [_NoneType, str], 'item_description_column')
        _raise_error_if_not_of_type(item_url_column, [_NoneType, str], 'item_url_column')

        # Store column names
        self.user_id_column = model.user_id
        self.user_name_column = user_name_column
        self.user_description_column = user_description_column
        self.user_url_column = user_url_column

        self.item_id_column = model.item_id
        self.item_name_column = item_name_column
        self.item_description_column = item_description_column
        self.item_url_column = item_url_column

        for k in [self.user_id_column, user_name_column, user_description_column, user_url_column]:
            _raise_error_if_column_missing(k, user_data, "user_data")

        for k in [self.item_id_column, item_name_column, item_description_column, item_url_column]:
            _raise_error_if_column_missing(k, item_data, "item_data")

        # Get the number of occurrences of each item
        self._item_counts = model.get_num_users_per_item()
        self._item_counts.rename({'num_users': '_num_users'})
        self._user_counts = model.get_num_items_per_user()
        self._user_counts.rename({'num_items': '_num_items'})

        # Store types
        self._item_type = self._item_counts[self.item_id_column].dtype()
        self._user_type = self._user_counts[self.user_id_column].dtype()
        self._allowed_item_types = [self._item_type]
        self._allowed_user_types = [self._user_type]

        if _sys.version_info.major == 2:
            if self._item_type == str:
                self._allowed_item_types += [unicode]
            if self._user_type == str:
                self._allowed_user_types += [unicode]

        # If no user or item data exists, then we default to the per-user
        # and per-item counts.
        if observation_data is None:
            observation_data = _gl.SFrame({self.user_id_column: [], self.item_id_column: []})

        self._empty_observation_data = observation_data.head(0)

        if user_data is None:
            user_data = self._user_counts
        if item_data is None:
            item_data = self._item_counts

        # Check observation data has required columns.
        for k in [self.user_id_column, self.item_id_column]:
            _raise_error_if_column_missing(k, observation_data, "observation_data")

        # Store a copy of the data. Required for search/history methods.
        self._observations_by_user = observation_data._group(self.user_id_column)
        self._observations_by_item = observation_data._group(self.item_id_column)
        self._users = user_data.__copy__()
        self._items = item_data.__copy__()
        self._unique_items = self._items[self.item_id_column].unique()

        # Train search models
        self.search_models = {}
        datasets = [('item', item_name_column, self.item_id_column, item_data),
                    ('user', user_name_column, self.user_id_column, user_data)]

        for (name, column, default_column, data) in datasets:
            search_features = [default_column]
            if column is not None:
                search_features.append(column)

            search_data = data.__copy__()
            for search_feature in search_features:

                if search_feature not in data.column_names():
                    raise ValueError("Expected column {} in {}_data".format(search_feature, name))

                col_type = search_data[search_feature].dtype()
                convert_col_to_str = search_data[search_feature].dtype() != str
                if convert_col_to_str:
                    search_data[search_feature] = search_data[search_feature].astype(str)

            self.search_models[name] = _search.create(search_data,
                features=search_features, verbose=False)


    def schema(self):
        """
        Return metadata about the view.

        Returns
        -------
        out : dict
            A dictionary containing:

            * user_id_coluumn
            * user_name_column
            * user_description_column
            * user_url_column
            * item_id_column
            * item_name_column
            * item_description_column
            * item_url_column

        """
        return {
            'user_id_column': self.user_id_column,
            'user_id_column_type': self._user_type.__name__,
            'user_name_column': self.user_name_column,
            'user_description_column': self.user_description_column,
            'user_url_column': self.user_url_column,
            'item_id_column': self.item_id_column,
            'item_id_column_type': self._item_type.__name__,
            'item_name_column': self.item_name_column,
            'item_description_column': self.item_description_column,
            'item_url_column': self.item_url_column
            }


    def more_like_this(self, item_id, num_recs=5):
        """
        Obtain data for items that are similar to the provided item according
        to the model.

        Parameters
        ----------
        item_id : int, str
            The id of the query item.

        num_recs: int, optional
            The number of recommendations.

        Returns
        -------
        out : SFrame
            A subset of the original items data.

        Examples
        --------

        Using the same example in the documentation for model.views.explore:

        view = model.views.explore(data, items, item_name_column='movie')


        """
        recs = self._model.get_similar_items([item_id], k=num_recs)
        recs = recs.remove_columns([self.item_id_column])\
                   .rename({'similar': self.item_id_column})
        recs = recs.join(self._items, on=self.item_id_column).sort('rank')
        return recs

    def search_items(self, query, num_results=5):
        """
        Search items data using the provided query.

        Parameters
        ----------
        query : str
            The search query.

        num_results : int, optional
            The number of results to return.

        Returns
        -------
        out : SFrame
            Data for items that contain approximate matches to the query.
        """
        result = self.search_models['item'].query(query, expansion_epsilon=1.0)
        return result.head(num_results)


    def search_users(self, query, num_results=5):
        """
        Search users data using the provided query.

        Parameters
        ----------
        query : str
            The search query.

        num_results : int, optional
            The number of results to return.

        Returns
        -------
        out : SFrame
            Data for users that contain approximate matches to the query.
        """

        result = self.search_models['user'].query(query, expansion_epsilon=1.0)
        return result.head(num_results)

    def most_common_items(self, num_results=5):
        """
        Get the most common items in the observation data.

        Parameters
        ----------
        num_results : int
            The number of items to return.
        """
        return self._item_counts.join(self._items, on=self.item_id_column).topk('_num_users', num_results)

    def most_common_users(self, num_results=5):
        """
        Get the most common users in the observation data.

        Parameters
        ----------
        num_results : int
            The number of items to return.
        """
        return self._user_counts.join(self._users, on=self.user_id_column).topk('_num_items', num_results)

    def personalized_recommendations(self, user_id, num_recs=5):
        """
        Obtain data for recommended items for the given user.

        Parameters
        ----------
        user_id : int, str
            The desired user id.

        num_recs : int
            The number of recommendations to return.

        Returns
        -------
        out : SFrame
            Data for the recommended items. If that user_id was not seen at
            training time, then an empty SFrame is returned.

        """
        recs = self._model.recommend(users=[user_id], items=self._unique_items, k=num_recs)
        recs = recs.join(self._items, on=self.item_id_column).sort('rank')
        del recs[self.user_id_column]
        return recs

    def relevant_history(self, user_id, item_id, num_results=None):
        """
        For a given (user, item) pair, get the subset of the user's history
        sorted by events containing items that are similar to the provided item.
        """
        # Get more than the request similar items incase these items aren't included
        # in side data when provided.
        if num_results is None:
            num_similar_items = 100
        else:
            num_similar_items = num_results * 2
        similar_items = self._model.get_similar_items(items=[item_id], k=num_similar_items)
        similar_items = similar_items.select_columns(['similar', 'score'])\
                                     .rename({'similar': self.item_id_column})
        history = self.user_history(user_id)\
                      .select_columns([self.user_id_column, self.item_id_column])\
                      .join(similar_items, on=self.item_id_column)\
                      .sort('score', ascending=False)

        if num_results is not None:
            history = history.head(num_results)

        if self.item_name_column:
            history = history.join(self._items, on=self.item_id_column)

        return history

    def random_item(self):
        """
        Returns a randomly chosen item id.
        """
        ix = _random.randint(0, self._item_counts.num_rows()-1)
        return self._item_counts[self.item_id_column][ix]

    def random_user(self):
        """
        Returns a randomly chosen user id.
        """
        ix = _random.randint(0, self._users.num_rows()-1)
        return self._users[self.user_id_column][ix]

    def item_history(self, item_id, num_results=None):
        """
        Returns the observation data relevant to the provided user.
        """
        _raise_error_if_not_of_type(item_id, self._allowed_item_types, 'item_id')

        try:
            result = self._observations_by_item.get_group(item_id)
        except RuntimeError as e:
            result = self._empty_observation_data
        if self.user_name_column:
            result = result.join(self._users, on=self.user_id_column)

        if num_results is not None:
            result = result.head(num_results)

        return result

    def user_history(self, user_id, num_results=None):
        """
        Returns the observation data relevant to the provided user.
        """
        _raise_error_if_not_of_type(user_id, self._allowed_user_types, 'user_id')

        try:
            result = self._observations_by_user.get_group(user_id)
        except RuntimeError as e:
            result = self._empty_observation_data
        if self.item_name_column:
            result = result.join(self._items, on=self.item_id_column)

        if num_results is not None:
            result = result.head(num_results)
        return result

    def item_history_count(self, item_id):
        '''
        Returns the number of interactions per item.
        '''
        _raise_error_if_not_of_type(item_id, self._allowed_item_types, 'item_id')

        user = _gl.SFrame({self.item_id_column: [item_id]})
        return user.join(self._item_counts, on=self.item_id_column)

    def user_history_count(self, user_id):
        '''
        Returns the number of interactions per user.
        '''
        _raise_error_if_not_of_type(user_id, self._allowed_user_types, 'user_id')

        user = _gl.SFrame({self.user_id_column: [user_id]})
        return user.join(self._user_counts, on=self.user_id_column)

    def item_details(self, item_id):
        """
        Obtain data for a given item.

        Parameters
        ----------
        item_id : int, str
            The id of the desired item.

        Returns
        -------
        out : SFrame
            Data for the desired item. If no row has the desired item_id, then
            an empty SFrame is returned.

        """
        _raise_error_if_not_of_type(item_id, self._allowed_item_types, 'item_id')

        item = _gl.SFrame({self.item_id_column: [item_id]})
        return item.join(self._items, on=self.item_id_column)


    def user_details(self, user_id):
        '''
        Returns the row of the user table given a user id.
        '''
        _raise_error_if_not_of_type(user_id, self._allowed_user_types, 'user_id')

        user = _gl.SFrame({self.user_id_column: [user_id]})
        return user.join(self._users, on=self.user_id_column)

    def users_in_common(self, item_a, item_b, num_results=None):
        """
        Get data on the users in common between two items.

        Parameters
        ----------

        item_a : The id of one item.

        item_b : The id of the other item.

        num_results : int, optional
            The number of users in common to return.

        Returns
        -------
        out : SFrame
           A SFrame with the two item columns given above, the number of
           users that rated each, and a dictionary mapping the user to a
           pair of the ratings, with the first rating being the rating of
           the first item and the second being the rating of the second item.
           If no ratings are provided, these values are always 1.0.

        Returns the observation data relevant to the provided user.
        """
        _raise_error_if_not_of_type(item_a, self._allowed_item_types, 'item_id')
        _raise_error_if_not_of_type(item_b, self._allowed_item_types, 'item_id')

        item_a_history = self.item_history(item_a)
        item_b_history = self.item_history(item_b)

        item_a_users = item_a_history[self.user_id_column].unique()
        item_b_users = item_b_history[self.user_id_column].unique()
        users_in_common = set(item_a_users).intersection(set(item_b_users))

        users = _gl.SFrame({self.user_id_column: list(users_in_common)})
        users = users.join(self._users, on=self.user_id_column)
        if num_results is not None:
            users = users.head(num_results)
        result = {
            'item_a_count': len(item_a_users),
            'item_b_count': len(item_b_users),
            'in_common_count': len(users_in_common),
            'in_common_users': users
        }
        return result

