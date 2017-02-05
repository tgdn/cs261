import string
from ._feature_engineering import Transformer
import types
import sys

def func_copy(f):
    """
    Make a copy of a function using the underlying function attributes.
    """

    if sys.version_info.major == 2:
        func_code = f.func_code
        func_globals = f.func_globals
        func_name = f.func_name
        func_defaults = f.func_defaults
        func_closure = f.func_closure
    else:
        func_code = f.__code__
        func_globals = f.__globals__
        func_name = __name__
        func_defaults = f.__defaults__
        func_closure = f.__closure__

    return types.FunctionType(func_code, func_globals, name = func_name,
                              argdefs = func_defaults, closure = func_closure)
def fit(self, data):
    return Transformer.fit(self, data)

def transform(self, data):
    return Transformer.transform(self, data)

def fit_transform(self, data):
    return Transformer.fit_transform(self, data)

def republish_docs(cls):
    """
    Republish the doc-strings for fit, transform, and fit_transform.
    """
    def get_doc_string(func_obj):
        if sys.version_info.major == 2:
            return func_obj.im_func.func_doc
        else:
            return func_obj.__doc__

    fit_copy = func_copy(fit)
    fit_copy.func_doc = get_doc_string(Transformer.fit)

    setattr(cls, 'fit', add_docstring(
            examples = cls._fit_examples_doc)(fit_copy))

    transform_copy = func_copy(transform)
    transform_copy.func_doc = get_doc_string(Transformer.transform)
    setattr(cls, 'transform', add_docstring(
            examples = cls._transform_examples_doc)(transform_copy))

    fit_transform_copy = func_copy(fit_transform)
    fit_transform_copy.func_doc = get_doc_string(Transformer.fit_transform)
    setattr(cls, 'fit_transform', add_docstring(
            examples = cls._fit_transform_examples_doc)(fit_transform_copy))
    return cls

class _Formatter(string.Formatter):
    """
    Format {strings} that are withing {brackets} as described in the doctring.
    """
    def get_value(self, key, args, kwargs):
        if hasattr(key,"__mod__") and key in args:
                return args[key]
        elif key in kwargs:
                return kwargs[key]
        return '{%s}' % key

def add_docstring(**format_dict):
    """
    __example_start = '''
       Examples
       ---------
    '''
    __create = '''
       >>> import graphlab as gl

       >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'
       >>> data = graphlab.SFrame.read_csv(url)
       >>> data['target'] = (data['target'] == 'e')

       >>> train, test = data.random_split(0.8)
       >>> model = graphlab.boosted_trees.create(train, target='label', *args, **kwargs)
    '''

    @add_docstring(create = __create, example_start = __example_start)
    def predict(x, **kwargs):
      '''
      {example_start}{create}
      '''
      return x

    """
    def add_doc_string_context(func):
        wrapper = func
        formatter = _Formatter()
        wrapper.func_doc = formatter.format(func.func_doc, **format_dict)
        return wrapper
    return add_doc_string_context

