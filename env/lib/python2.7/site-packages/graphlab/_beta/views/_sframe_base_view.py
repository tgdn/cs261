from .platform import View
import graphlab._json as _json
import graphlab.connect as _mt

class SFrameBaseView(View):
    def _create_cpo(self):
        """
        Take an object and create a CustomQueryPredictiveObject that exposes all of the
        object's methods. Unlike the base View class, this object supports automatic
        serialization of all SFrame types (including SFrame, SArray, SGraph, and
        the underlying flexible_type (dtypes of SArray).

        As an example, suppose we have a class Foo.

        class Foo(object):
            def __init__(self):
                pass
            def bar(self, x=3):
                return x+5

        f = Foo()
        f_cpo = f._create_cpo()

        Now f_cpo is a function whose first argument is the name of the desired method of Foo,
        and whose positional and keyword arguments are passed to that chosen method:

        >>> f_cpo(method="bar", x=3)

        will be identical to

        >>> f.bar(x=3)
        """
        def cpo(method, *args, **kwargs):
            f = getattr(self, method)
            out = f(*args, **kwargs)
            (data, schema) = _json.to_serializable(out)
            result = {
                'method': method,
                'result': data,
                'schema': schema
            }
            return result
        return cpo

    def show(self):
        # add GLC metrics tracking to the base class
        view_markup = super(SFrameBaseView, self).show()
        # TODO @rbkriesberg why is this event toolkits.*? Should Views have its
        # own event tracking namespace? Not everything here will be a toolkit-
        # related view.
        _mt._get_metric_tracker().track('toolkits.view.show', 
          properties={
          'view': type(view_markup).__name__,
          'tag_name': self.tag_name,
          'name': self.name
          })
        return view_markup
