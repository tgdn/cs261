from .layout import _layout

class App(_layout.LayoutView):
    """
    A view markup that cannot be embedded because it affects
    global browser state.
    """
    pass
