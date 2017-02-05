from .. import _sframe_base_view

def _flexible_type_name_from_dtype(dtype):
    name = dtype.__name__
    return {
        'int': 'integer',
        'float': 'float',
        'str': 'string',
        'list': 'list',
        'array': 'array',
        'dict': 'dictionary',
        'datetime': 'datetime',
        'Image': 'image'
    }[name]

class TableView(_sframe_base_view.SFrameBaseView):
    def __init__(self, sf):
        super(TableView, self).__init__(
            name='Table',
            label='Table',
            tag_name='gl-table',
            description='Scalable, scrollable table of data'
        )
        # TODO -- support more types than SFrame
        # (decouple column-header handling from SFrame-specific methods)
        self.sf = sf

    def rows(self, start, end):
        return self.sf[start:end]

    def metadata(self):
        return {
            'columns': zip(self.sf.column_names(), map(_flexible_type_name_from_dtype, self.sf.column_types())),
            'num_rows': len(self.sf)
        }
