from .. import _view_markup
from .. import _view

class LayoutView(_view_markup.ViewMarkup):
    def __init__(self, tag_name, attributes, *args):
        self._views = [self._convert_to_view_markup(v) for v in args]
        self._tag_name = tag_name
        self._attributes = attributes
        self._view_html = ''.join([v._html_for_layout() for v in self._views])
        self._ = ''.join([v._html_for_layout() for v in self._views])
        # Top level layout component -- escape with html comments.
        # If the contents of the wrapper is not HTML commented, it renders into
        # the page in seemingly nondeterministic order, and the contents will
        # get rendered as-is, even if the outer component is supposed to hide
        # them or transform them.
        # If wrapped in an HTML comment, the browser doesn't interpret the
        # contents, but they still get passed to the parent component as
        # children and handled in React.
        # See:
        # https://github.com/greenish/react-mount/blob/f3bee05f34838a2b657fff5d5812ab73ed9a60bb/readme.md#nested-components
        # https://github.com/greenish/react-mount/blob/f3bee05f34838a2b657fff5d5812ab73ed9a60bb/readme.md#html-comments
        self.html = """
<%s %s>
    <!--
        %s
    -->
</%s>
        """ % (self._tag_name, self._attributes, self._view_html, self._tag_name)
        self._publish()

    def _convert_to_view_markup(self, v):
        if isinstance(v, _view.View):
            return v._publish() # publishes to local ViewServer
        if not(isinstance(v, _view_markup.ViewMarkup)):
            raise TypeError('Expected parameters to a layout view to be View or ViewMarkup. Found %s.' % type(v))
        return v

    def _html_for_layout(self):
        # overrides View._html_for_layout
        # don't escape inner contents with <!-- -->
        # if this method is being called, it means the comment wrapping
        # has already been done by a parent layout component.
        # This is to prevent double-nested-commenting, as follows:
        # <!-- something <!-- something else --> end something -->
        # (nested comments are not supported in HTML).
        return """
<%s %s>
    %s
</%s>
        """ % (self._tag_name, self._attributes, self._view_html, self._tag_name)
