import graphlab.canvas.views.base
import graphlab.canvas.views.sframe

import sys
if sys.version_info.major == 3:
    izip = zip
else:
    from itertools import izip

class SGraphView(graphlab.canvas.views.base.BaseView):

    MAX_VERTICES = 1000
    MAX_EDGES = 1000

    def __init__(self, obj, params):
        super(SGraphView, self).__init__(obj, params)
        self.register_handler('get', 'vertices', self.__get_vertices)
        self.register_handler('get', 'edges', self.__get_edges)

    def get_metadata(self):
        return {
            'descriptives': {
                'vertices': self.obj.summary()['num_vertices'],
                'edges': self.obj.summary()['num_edges']
            },
            'descriptives_links': {
                'vertices': 'vertices',
                'edges': 'edges'
            },
            'view_params': self.params
        }

    def child_views(self):
        # lazily initialize SFrameView objects for children and store them here
        # so that we can share cached sketches with SArrayView
        if len(self.children) != 2:
            self.children = {
                'vertices': graphlab.canvas.views.sframe.SFrameView(self.obj.vertices),
                'edges': graphlab.canvas.views.sframe.SFrameView(self.obj.edges)
            }
        return self.children

    def get_js_file(self):
        return 'sgraph'
    
    def get_js_component(self):
        return 'View'

    def __exceeds_vertex_limit(self):
        ## Error trapping
        num_vertices = self.obj.summary()['num_vertices']
        return num_vertices > SGraphView.MAX_VERTICES

    def __exceeds_edge_limit(self):
        ## Error trapping
        num_edges = self.obj.summary()["num_edges"]
        return num_edges > SGraphView.MAX_EDGES

    def __get_vlabel(self):
        if not 'vlabel' in self.params:
            return None
        vlabel = self.params['vlabel']
        if vlabel is None:
            return None
        if vlabel == 'id':
            vlabel = '__id'
        return list(self.obj.vertices[vlabel])

    def __get_vposition(self):
        if not 'vertex_positions' in self.params:
            return None
        vertex_positions = self.params['vertex_positions']
        if vertex_positions is None:
            return None
        return {
            'names': vertex_positions,
            'x': list(self.obj.vertices[vertex_positions[0]]),
            'y': list(self.obj.vertices[vertex_positions[1]])
        }

    def __get_elabel(self):
        if not 'elabel' in self.params:
            return None
        elabel = self.params['elabel']
        if elabel is None:
            return None
        return list(self.obj.edges[elabel])

    @staticmethod
    def __get_limit_msg():
      return ("GraphLab Canvas cannot display an SGraph with more than %d" % SGraphView.MAX_VERTICES) \
        + (" vertices or %d edges. A subgraph with fewer vertices can be constructed by creating " % SGraphView.MAX_EDGES) \
        + "a new SGraph with a selection of edges and vertices from this SGraph or with the " \
        + "get_neighborhood method."

    def __get_vertices(self, url, handler):
        if self.__exceeds_vertex_limit() or \
           self.__exceeds_edge_limit():
            handler.set_status(413, SGraphView.__get_limit_msg()) # request entity too large
            handler.write({})
            return

        handler.write({
            'values': list(self.obj.vertices['__id']),
            'labels': self.__get_vlabel(),
            'positions': self.__get_vposition()
        })

    def __get_edges(self, url, handler):
        if self.__exceeds_vertex_limit() or \
           self.__exceeds_edge_limit():
            handler.set_status(413, SGraphView.__get_limit_msg()) # request entity too large
            handler.write({})
            return

        edge_sf = self.obj.get_edges()
        handler.write({
            'values': [list(x) for x in izip(edge_sf['__src_id'], edge_sf['__dst_id'])],
            'labels': self.__get_elabel()
        })

    def get_staticdata(self):
        if self.__exceeds_vertex_limit() or \
           self.__exceeds_edge_limit():
            return {
                'error_type' : 413,
                'error_msg' : self.__get_limit_msg() # request entity too large
            }
        
        edge_sf = self.obj.get_edges()
        data = {
            'vertices' : list(self.obj.vertices['__id']),
            'vertices_labels' : self.__get_vlabel(),
            'positions' : self.__get_vposition(),
            'edges' : [list(x) for x in izip(edge_sf['__src_id'], edge_sf['__dst_id'])],
            'edges_labels' : self.__get_elabel(),
            'error_type' : 0,
            'error_msg' : ""
        }
        return data
