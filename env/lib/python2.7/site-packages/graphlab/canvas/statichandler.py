import graphlab
import tornado.web

if graphlab.version == "{{VERSION_STRING}}":
    # static handler for debug builds
    # no caching (means we don't have to do any cache clearing in the browser to pick up changed
    # static files, but it will hurt performance so we shouldn't do this in release builds.
    class Handler(tornado.web.StaticFileHandler):
        def should_return_304(self):
            # don't give 304 -- if the browser asks for a file, always give real contents
            return False
        def set_extra_headers(self, path):
            # don't allow browsers to cache files
            self.set_header("Cache-control", "no-cache")
else:
    # subclass StaticFileHandler but don't change anything for release
    class Handler(tornado.web.StaticFileHandler):
        pass
