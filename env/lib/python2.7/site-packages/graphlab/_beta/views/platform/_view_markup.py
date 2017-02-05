import uuid as _uuid
import webbrowser as _webbrowser

from . import _httpresponse
from . import _view_server

class ViewMarkup(object):
    def _publish(self):
        server = _view_server.get_instance()
        name = str(_uuid.uuid4())

        server._add_or_update(name, self)
        server.apply_changes()
        status = server.get_status()
        dns_name = status[0]['dns_name']
        port = server.port
        if _view_server.server_port_override is not None:
            port = _view_server.server_port_override
        # side effect: store the deployed URI in this object for convenience
        self.__uri = 'http://%s:%s/view/%s' % (dns_name, port, name)

    def __init__(self, tag_name, uri, api_key):
        self.html = """
<%s
    uri="%s"
    api_key="%s"
/>
        """ % (tag_name, uri, api_key)
        self._publish()

    @property
    def uri(self):
        return self.__uri

    def _html_for_layout(self):
        return self.html

    def __repr__(self):
        r = "View object\n"
        r += "\nURI: \t\t{}".format(self.uri)
        r += "\nHTML: \t\t{}".format(self.html)
        return r

    def show(self):
        _webbrowser.open_new_tab(self.__uri)

    def _view(self):
        # renders as a full page
        contents = """
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8" />
        <style>
           body {
              margin : 0;
           }
        </style>
    </head>
    <body>
        %s
        <script src="%s"></script>
    </body>
</html>
        """ % (self.html, _view_server.js_url)
        return _httpresponse._HTTPResponse(contents, 200, 'text/html')
