import cgi as _cgi
import json as _json

from . import _layout

class TabsView(_layout.LayoutView):
    def __init__(self, *args, **kwargs):
        """
        Constructs a view containing one or more tabs, the contents of each tab
        being another View or ViewMarkup (Views will get implicitly published
        and displayed as ViewMarkup).

        If headers is omitted, the names of the views
        will be used as headers.
        """

        headers = None
        if 'headers' in kwargs:
            headers = kwargs['headers']
        if headers is None:
            headers = [v.label for v in args]

        super(TabsView, self).__init__(
            'gl-layout-tabs',
            """
headers="%s"
            """ % _cgi.escape(_json.dumps(headers), quote=True),
            *args
        )
