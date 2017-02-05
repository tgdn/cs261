import cgi as _cgi
import json as _json

from .platform import App

class OverviewApp(App):
    def __init__(self, *args, **kwargs):
        """
        Constructs a view containing a sidebar and one or more tabs.
        The contents of the sidebar and of each tab should be
        another View or ViewMarkup (Views will get implicitly published
        and displayed as ViewMarkup).

        If headers is omitted, the names of the views
        will be used as headers.

        A title may also be supplied.
        """

        sidebar_contents = args[0]
        tab_contents = args[1:]

        headers = None
        if 'headers' in kwargs:
            headers = kwargs['headers']
        if headers is None:
            headers = [v.label for v in tab_contents]

        # Encode json for html attribute
        headers = _cgi.escape(_json.dumps(headers), quote=True)

        title = "Model Overview"
        if 'title' in kwargs:
            title = kwargs['title']

        super(OverviewApp, self).__init__(
            'gl-model-overview',
            """
headers="%s" title="%s"
            """ % (headers, title),
            *args
        )
