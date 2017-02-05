import uuid as _uuid

from . import _view_markup
from . import _view_server

class View(object):
    def __init__(self, name, label, tag_name, description=''):
        """
        Initializes a View object

        Parameters
        ----------
        name        : The full name of the view
        label       : The label to display in a (tab) layout
        tag_name    : The web component labels
        description : Decription text for view
        """
        self.name = name
        self.label = label
        self.tag_name = tag_name
        self.description = description
        self.uri = None

    def _create_cpo(self):
        """
        Take an object and create a CustomQueryPredictiveObject that exposes all of the
        object's methods.

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
        # TODO: Rename this method. The "CPO" name is entirely vestigial, since this
        # doesn't integrate with Predictive Services at all. Really this just
        # wraps all the method calls into a JSON dictionary of output that carries
        # metadata about the request/response, and guarantees JSON-serializability.
        def cpo(method, *args, **kwargs):
            f = getattr(self, method)
            out = f(*args, **kwargs)
            result = {
                'method': method,
                'result': out,
                'schema': None
            }
            return result
        return cpo

    def _publish(self):
        """
        Publish this object to an endpoint.
        """
        name = str(_uuid.uuid4())
        server = _view_server.get_instance()
        cpo = self._create_cpo()
        server.add(name, cpo) # only add -- uuid4 usually should not collide
        server.apply_changes()
        status = server.get_status()
        dns_name = status[0]['dns_name']
        # side effect: store the deployed URI in this object for convenience
        self.uri = 'http://%s:%d/view/%s' % (dns_name, server.port, name)
        # create ViewMarkup from this deployed View
        self.html = _view_markup.ViewMarkup(
            tag_name = self.tag_name,
            uri = self.uri,
            api_key = server.api_key
        )
        return self.html

    def show(self):
        """
        Launch web browser tab containing visualization.
        """
        view_markup = self._publish()
        view_markup.show()
        return view_markup
