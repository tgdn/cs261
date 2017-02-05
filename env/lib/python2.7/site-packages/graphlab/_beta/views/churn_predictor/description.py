from .. import _sframe_base_view
from graphlab.toolkits._internal_utils import _toolkit_serialize_summary_struct

class ChurnPredictorDescriptionView(_sframe_base_view.SFrameBaseView):
    def __init__(self, model, extra_info):
        super(ChurnPredictorDescriptionView, self).__init__(
            name='Churn Predictor Model Description',
            label='Description',
            tag_name='gl-model-description',
            description='Description of a trained churn predictor model.'
        )

        fields, schema = model._get_summary_struct()
        if type(extra_info) == tuple:
            extra_fields, extra_schema = extra_info
            fields.extend(extra_fields)
            schema.extend(extra_schema)
        self.__info = _toolkit_serialize_summary_struct(model, fields, schema)

    def info(self):
        return self.__info
