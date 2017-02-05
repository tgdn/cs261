from . import _layout

class StackedView(_layout.LayoutView):
    def __init__(self, *args):
        super(StackedView, self).__init__(
            'gl-layout-stacked',
            '',
            *args
        )
