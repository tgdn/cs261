from .. import _sframe_base_view
from graphlab.toolkits._internal_utils import _toolkit_serialize_summary_struct

class RecommenderDescriptionView(_sframe_base_view.SFrameBaseView):
    def __init__(self, model):
        super(RecommenderDescriptionView, self).__init__(
            name='Recommender Model Description',
            label='Description',
            tag_name='gl-model-description',
            description='Description of a trained recommender model.'
        )

        fields, schema = model._get_summary_struct()
        self.__info = _toolkit_serialize_summary_struct(model, fields, schema)

    def info(self):
        return self.__info
