"""
GraphLab Create includes GraphLab Canvas, a built-in visualization platform and
web application designed to compliment the GraphLab Create data science and
development workflow. GraphLab Canvas can be used interactively in
a browser-based application or directly via API methods.

**Visualizing GraphLab data structures in GraphLab Canvas**

GraphLab Canvas allows you to explore interactive visuals and a table of the
data in an :class:`~graphlab.SFrame`, :class:`~graphlab.SGraph` and
:class:`~graphlab.SArray` in the local machine's default browser or inside an
IPython Notebook or Jupyter Notebook. To launch GraphLab Canvas and generate 
views of these data structures, simply call `.show()` on a supported GraphLab 
Create data structure. Subsequent calls of `.show()` will update the GraphLab 
Canvas session.

Details on the use of GraphLab Canvas with GraphLab Create data structures are
provided below. In addition to the API documentation, please see the data
science `Gallery <https://turi.com/learn/gallery>`_, `How-tos
<https://turi.com/learn/gallery>`_, and the `visualization chapter of
the User Guide
<https://turi.com/learn/userguide/sframe/visualization.html>`_
for more details and extended examples.


**Resuming GraphLab Canvas after ending browser session**

The GraphLab Canvas session is hosted by the GraphLab Create Python session.
While the GraphLab Create Python session is active you can re-launch GraphLab
Canvas and resume its prior state.

>>> gl.canvas.show()

**Using GraphLab Canvas with IPython Notebook or Jupyter Notebook**

GraphLab Canvas `.show()` methods can be used in IPython Notebook or Jupyter 
Notebook. By default, `.show()` will launch GraphLab Canvas in the browser.
Optionally, you can output a single GraphLab Canvas view to an IPython Notebook 
or Jupyter Notebook cell using the :func:`~graphlab.canvas.set_target` method, 
as follows:

>>> gl.canvas.set_target('ipynb')

In the same session, you can set the output to the browser using

>>> gl.canvas.set_target('browser')
"""

from graphlab.canvas.utils import set_target
from graphlab.canvas.utils import get_target
from graphlab.canvas.utils import show
from graphlab.canvas.utils import _get_id
from graphlab.canvas.utils import _same_object
from . import glc_display_dispatch
