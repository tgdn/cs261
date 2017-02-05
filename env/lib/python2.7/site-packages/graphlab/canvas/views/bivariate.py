import sys
if sys.version_info.major == 3:
    long = int


class BivariateView(object):
    def __init__(self, obj, params):
        self.obj = obj
        self.params = params

    def get_column(self, name, avoid=None, numeric=False):
        if avoid is None:
            avoid = []
        if not(isinstance(avoid, list)):
            avoid = [avoid]

        columns = self.__type_columns([int, float, long]) if numeric else self.obj.column_names()

        # if a parameter name is specified and valid, use it
        if name in self.params and self.params[name] in columns:
            return self.params[name]

        # if there is something available other than the "avoid" column, use it
        filtered = [col for col in columns if not(col in avoid)]
        if len(filtered) > 0:
            return filtered[0]

        # no valid columns
        return None

    def get_numeric_column(self, name, avoid=None):
        return self.get_column(name, avoid, numeric=True)

    def get_typed_column(self, name, avoid=None, types=[]):
        column = self.get_column(name, avoid, False)
        dt = self.__type_columns(types)
        ret = column if column in dt else None
        return ret

    def __type_columns(self, types):
        '''
        type is a list of types can be ploted
        e.g, [int, float, long, datetime.datetime]
        '''
        columns = zip(self.obj.column_names(), self.obj.column_types())
        return [col for (col, dtype) in columns if dtype in types]
