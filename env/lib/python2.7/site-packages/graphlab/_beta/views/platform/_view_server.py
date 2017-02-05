# -*- coding: utf-8 -*-
'''
Copyright (C) 2016 Turi
All rights reserved.

View Server
A lightweight service on localhost for hosting Views
@author: znation
'''

import collections as _collections
import copy as _copy
import json as _json
import os as _os
import requests as _requests
import threading as _threading
import tornado as _tornado
import tornado.web as _web
import types as _types
import uuid as _uuid
import six as _six

################################################################################
# DEBUGGING
# In the widgets repo, run `npm run dato-dev`, and use the environment variables
# to enable a "hot-loading" front-end development environment.
# Front-end code changes in the widgets repo will take effect in the live app.
# Alternately, use the `npm run dev` environment within the widgets repo,
# for a separate front-end development experience (decoupled from .show).
#
# For `npm run dato-dev` in widgets, use:
# export TURI_VIEWS_SERVER_PORT_OVERRIDE=3000
# export TURI_VIEWS_JS_URL=http://localhost:3000/widgets.webcomponents.js
#
server_port_override = _os.environ.get('TURI_VIEWS_SERVER_PORT_OVERRIDE', None)
js_url = _os.environ.get('TURI_VIEWS_JS_URL', 'https://static.turi.com/products/views/0.0.16/widgets.webcomponents.js')
################################################################################

from . import _httpresponse

def _ensure_unicode(value):
    # Makes sure that the contents of the structure to be written are
    # encoded in utf-8 before attempting to JSON serialize.

    # recursive structures need to be ensured recursively
    if isinstance(value, (dict,)):
        return {str(k): _ensure_unicode(v) for k,v in value.items()}
    if isinstance(value, (list,)):
        return [_ensure_unicode(v) for v in value]

    # unicode-ify all the strs
    if isinstance(value, (_six.binary_type,)):
        try:
            return value.decode('utf-8')
        except:
            try:
                return value.decode('utf-16')
            except:
                try:
                    return value.decode('utf-32')
                except:
                    try:
                        return value.decode('latin_1')
                    except:
                        raise ValueError('Views only support bytes/str encoded in UTF-8, UTF-16, UTF-32, or Latin-1. Ensure all strings are in one of these encodings.')

    # All other types, just pass through
    return value

class _Endpoints(object):
    def __init__(self):
        self.__endpoints = {}
        self.__lock = _threading.Lock()
    def list(self):
        return list(self.__endpoints.keys())
    def add(self, name, obj):
        with self.__lock:
            if name in self.__endpoints:
                raise ValueError("There is already an endpoint with name '%s'." % name)
            self.__endpoints[name] = obj
    def update(self, name, obj):
        with self.__lock:
            if not(name in self.__endpoints):
                raise ValueError("There is no endpoint with name '%s'." % name)
            self.__endpoints[name] = obj
    def _add_or_update(self, name, obj):
        with self.__lock:
            self.__endpoints[name] = obj
    def has_endpoint(self, name):
        with self.__lock:
            return name in self.__endpoints
    def view(self, name):
        with self.__lock:
            return self.__endpoints[name]._view()
    def query(self, name, **kwargs):
        with self.__lock:
            return self.__endpoints[name](**kwargs)

class _Headers(object):
    def __init__(self):
        self.__headers = {}
        self.__lock = _threading.Lock()
    def set(self, name, value):
        with self.__lock:
            if value is None:
                del self.__headers[name]
            else:
                self.__headers[name] = value
    def get(self):
        with self.__lock:
            return _copy.copy(self.__headers)

class _EndpointHandler(_web.RequestHandler):
    def initialize(self, endpoints, headers):
        self.__endpoints = endpoints
        headers = headers.get()
        for (k,v) in _six.iteritems(headers):
            self.set_header(k,v)


    def write(self, chunk):
        # overloads super class write method, to get custom JSON handling behavior.
        chunk = _ensure_unicode(chunk)
        super(_EndpointHandler, self).write(chunk)

    def _write_error(self, status_code, error_type="Unknown Error", message="No error message specified"):
        self.set_status(status_code)
        self.write({
          'status': "error",
          'error' : {
            'type': error_type,
            'message': message
          }
        })

    def _write(self, chunk):
        if not isinstance(chunk, dict):
            self.write(chunk)
            return
        new_body = chunk.copy()
        new_body.update({
          'status': "ok"
          })
        self.write(new_body)

    def get(self, endpoint):
        if not(self.__endpoints.has_endpoint(endpoint)):
            self._write_error(404, 'UnknownURI', "Endpoint '%s' does not exist" % endpoint)
            return
        response = self.__endpoints.view(endpoint)

        assert isinstance(response, _httpresponse._HTTPResponse), 'Expected _HTTPResponse'
        self.set_header('content-type', response.content_type)
        self.set_status(response.status_code)
        self._write(response.body)

    def post(self, endpoint):
        if not(self.__endpoints.has_endpoint(endpoint)):
            self._write_error(404, 'UnknownURI', "Endpoint '%s' does not exist" % endpoint)
            return

        if self.request.body:
            arguments = _json.loads(self.request.body.decode('utf-8'))
            if 'api_key' not in arguments or 'data' not in arguments:
                raise KeyError("Endpoint expects both 'api_key' and 'data'.")
            api_key = arguments['api_key']
            kwargs = arguments['data']
        else:
            kwargs = {}

        try:
            response = self.__endpoints.query(endpoint, **kwargs)
        except BaseException as e:
            import traceback
            tback = traceback.format_exc()
            print(str(e))
            print(tback)
            self._write_error(500, str(e), tback)
            return

        if isinstance(response, _httpresponse._HTTPResponse):
            self.set_header('content-type', response.content_type)
            self.set_status(response.status_code)
            self._write(response.body)
            return

        self.set_header('content-type', 'application/json')
        response = {
            'from_cache': False,
            'model': endpoint,
            'version': 1,
            'uuid': str(_uuid.uuid4()),
            'response': response
        }
        self._write(response)

class ViewServer(object):
    # Public API
    def __init__(self, port=None):
        self.api_key = ''
        self.port = None
        self.__desired_port = port
        self.__headers = _Headers()
        self.__changes = _collections.deque()
        self.__thread = None # bg thread that runs Tornado
        self.__ioloop = None # Tornado ioloop
        self.__endpoints = _Endpoints()
        self.__base_url = 'http://127.0.0.1'
        if server_port_override is not None:
            self.set_CORS('http://localhost:%s' % server_port_override)
        # this has to come last
        # (the rest of this obj must be initialized before start is called)
        self.__start() # starts Tornado on bg thread

    def list(self):
        return self.__endpoints.list()

    def get_status(self):
        return [
            { 'dns_name': 'localhost' }
        ]

    def __repr__(self):
        s = "_ViewServer instance\n"
        s += '{0: <10}{1}\n'.format('Base url', self.__base_url)
        s += '{0: <10}{1}\n'.format('Port', self.port)

        s += '\nEndpoints:\n'
        for e in self.__endpoints.list():
            s += '\t{}\n'.format(e)
        return s

    ###
    ### Methods to do CRUD operations on deployed CPOs
    ###
    def add(self, name, obj):
        def _change(self):
            self.__endpoints.add(name, obj)
        self.__changes.append(_change)

    def update(self, name, obj):
        def _change(self):
            self.__endpoints.update(name, obj)
        self.__changes.append(_change)

    def _add_or_update(self, name, obj):
        def _change(self):
            self.__endpoints._add_or_update(name, obj)
        self.__changes.append(_change)

    def _uri(self, name):
        return '%s:%d/%s' % (self.__base_url, self.port, name)

    ###
    ### Query methods
    ###
    def query(self, name, **kwargs):
        body = _json.dumps({'data': kwargs, 'api_key': ''})
        uri = '%s:%d/query/%s' % (self.__base_url, self.port, name)
        r = _requests.post(uri,
                           data=body)
        if r.headers['content-type'].startswith('application/json'):
            return r.json()['response']
        else:
            return r.text

    def test_query(self, name, **kwargs):
        # directly calls the endpoint query method
        return self.__endpoints.query(name, **kwargs)

    ###
    ### Control plane (service-related methods)
    ###
    def apply_changes(self):
        # until called, changes queue up in self.__changes
        while len(self.__changes) > 0:
            change = self.__changes.popleft()
            change(self)

    def set_CORS(self, value):
        def _change(self):
            self.__headers.set('Access-Control-Allow-Origin', value)
        self.__changes.append(_change)

    def terminate_service(self):
        assert(self.__ioloop is not None)
        self.__ioloop.add_callback(lambda x: x.stop(), self.__ioloop)
        self.__thread.join()

    ###
    ### Internal/private API
    ###
    def __run(self):
        assert(self.__ioloop is not None)
        self.__ioloop.start()
        # close after start, see
        # http://www.tornadoweb.org/en/stable/ioloop.html#tornado.ioloop.IOLoop.close
        self.__ioloop.close()

    def __start(self):
        if self.__thread is None:
            self.__ioloop = _tornado.ioloop.IOLoop()
            self.__application = _web.Application(
                handlers=(
                    (r"/query/(.*)$", _EndpointHandler, dict(endpoints=self.__endpoints, headers=self.__headers)),
                    (r"/view/(.*)$", _EndpointHandler, dict(endpoints=self.__endpoints, headers=self.__headers))
                )
            )
            if self.__desired_port is None:
                start_port = 32212
                end_port = start_port + 100
                for port in range(start_port, end_port):
                    try:
                        self.__application.listen(port)
                        self.port = port
                        break
                    except IOError as e:
                        pass
                if self.port is None:
                    raise IOError("Could not bind a valid port between {0}-{1}".format(start_port, end_port))
            else:
                self.__application.listen(self.__desired_port)
                self.port = self.__desired_port
            self.__thread = _threading.Thread(target=self.__run, name='view_server')
            self.__thread.daemon = True
            self.__thread.start()

_ViewServerInstance = None
def get_instance():
    global _ViewServerInstance
    if _ViewServerInstance is None:
        _ViewServerInstance = ViewServer()
    return _ViewServerInstance
