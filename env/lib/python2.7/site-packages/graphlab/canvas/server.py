"""
This module provides GraphLab Canvas server functionality on top of Tornado.
The InteractiveView uses this server to send data-structure-specific
visualization information to the browser.
"""

import graphlab.canvas.handlers
import graphlab.canvas.state

import datetime
import time
import threading
import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.web
import uuid

class Server:
    #
    # private variables
    #

    # instance variables
    __thread = None # bg thread that runs Tornado
    __port = None # port Tornado is running on (determined automatically at thread launch)
    __port_lock = threading.RLock()
    __last_ping = None # last ping time from the browser (data check)
    __last_ping_lock = threading.RLock()
    __loop = None
    __application = None
    __server = None

    #
    # public API
    #

    # instance variables
    state = None

    def __init__(self, state, port=None):
        self.state = state
        self.__initialize_server()

        #Will raise exception if port cannot be bound
        self.__bind_socket(port)

    # methods
    def ping(self):
        """
        Updates the timestamp for the alive() calculation.
        """
        with self.__last_ping_lock:
            self.__last_ping = datetime.datetime.now()

    def alive(self):
        """
        Returns True if the browser has communicated with the server recently, false otherwise.
        """
        with self.__last_ping_lock:
            if self.__last_ping is None:
                return False
            # if the browser hasn't made a request for data in the last 3 seconds, consider it dead
            return (datetime.datetime.now() - self.__last_ping) < datetime.timedelta(0, 3)

    def get_port(self):
        with self.__port_lock:
            return self.__port

    def start(self):
        """
        Starts the canvas server if it is not already active
        """

        if self.__thread is None:
            self.__thread = threading.Thread(target=self.__run, name='canvas_server')
            # TODO -- if we want to keep the Python process running until the
            # visualization is closed (a la Matplotlib) we can take this out of daemon
            # mode and do our own thread management
            self.__thread.daemon = True
            self.__thread.start()
        # block the thread and wait for the Tornado bg thread to initialize until the port is set
        port = None
        while port is None:
            port = self.get_port()
            if port is None:
                time.sleep(0.05)

    #
    # private API
    #

    # Raises exception if socket cannot be bound on requested port
    def __bind_socket(self, port=None):
        sockets = tornado.netutil.bind_sockets(port,"localhost")
        self.__server.add_sockets(sockets)
        with self.__port_lock:
            self.__port = sockets[0].getsockname()[:2][1]


    def __initialize_server(self):
        # use our own IOLoop instead of singleton to avoid clashing with IPython/Jupyter Notebook
        self.__loop = tornado.ioloop.IOLoop()
        self.__application = tornado.web.Application(
            graphlab.canvas.handlers.get_handlers(self, self.state),
            io_loop=self.__loop
        )
        self.__server = tornado.httpserver.HTTPServer(self.__application, io_loop=self.__loop)

    def __run(self):
        self.__loop.start()
