"""
 Server Side Event Handler for emitting messages from tailed log files.
"""

# encoding: utf-8
import uuid
import tornado.web
import tornado.escape
import os as _os
import six

import graphlab.deploy

SSE_HEADERS = (
    ('Content-Type','text/event-stream; charset=utf-8'),
    ('Cache-Control','no-cache'),
    ('Connection','keep-alive'),
    ('Access-Control-Allow-Origin', '*'),
)

# SSE "protocol" is described here: http://mzl.la/UPFyxY
class ServerSentEvent(object):

    def __init__(self, data, event=None, id=None):
        self.data = data
        self.event = event
        self.id = id
        self.desc_map = {
            self.data : "data",
            self.event : "event",
            self.id : "id"
        }

    def encode(self):
        if not self.data:
            return ""
        lines = ["%s: %s" % (v, k)
                 for k, v in six.iteritems(self.desc_map) if k]

        return "%s\n\n" % "\n".join(lines)

class SSEHandler(tornado.web.RequestHandler):
    """
    URL route expects an object id to be passed in as the first parameter
    to the get handler

    specified object id is validated before stream connection is started
    """

    _connections = {}
    _source = None

    def __init__(self, application, request, **kwargs):
        super(SSEHandler, self).__init__(application, request, **kwargs)
        self.stream = request.connection.stream
        self._closed = False

    def set_default_headers(self):
        for name, value in SSE_HEADERS:
            self.set_header(name, value)

    def get_class(self):
        return self.__class__

    def get_source(self, *args):
        return None

    def listen(self, source):
        return None

    @tornado.web.asynchronous
    def get(self, obj_id):
        # validate id
        self._source = self.get_source(obj_id)
        # abort connection if invalid id
        if not self._source:
            self.set_status(403)
            self.finish()
        else:
            # Sending the standard headers: open event
            self.on_open()

    def on_open(self, *args, **kwargs):
        """ Invoked for a new connection opened. """
        self.listen(self._source)

    def on_close(self):
        """ Invoked when the connection for this instance is closed. """
        pass

    def on_connection_close(self):
        """ Closes the connection for this instance """
        self.on_close()
        self.stream.close()

    def send_message(self, data):
        """ Sends a message to all live connections """
        msg_id = str(uuid.uuid4())
        event = 'message'

        sse = ServerSentEvent(data, event, msg_id)
        message = sse.encode()

        self.on_message(message)

    def on_message(self, message):
        self.write(message)
        self.flush()


class LogSSEHandler(SSEHandler):
    """
    Stream messages from log file as it is being written to
    """
    line_terminators = ('\r\n', '\n', '\r')

    def __init__(self, application, request, **kwargs):
        super(LogSSEHandler, self).__init__(application, request, **kwargs)
        self._file_handle = None
        self._periodic_task = None

    # return the job log file path, if available
    # otherwise, return None.  Connection will be closed.
    def get_source(self, job_id):
        try:
            job_obj = graphlab.deploy.jobs[job_id]
        except KeyError:
            return None

        job_log_file = job_obj.get_log_file_path()
        return job_log_file


    # close file handle and periodic callbacks
    def on_close(self):
        """ Invoked when the connection for this instance is closed. """
        # close log file if it has been opened
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except:
                pass
        # end periodic callback if it has been created
        if self._periodic_task is not None:
            try:
                self._periodic_task.stop()
            except:
                pass

    # asynchronous writes of log file messages.
    # generator exits when log file is closed and all
    # messages have been sent
    @tornado.web.asynchronous
    def listen(self, log_file_path):
        for line in self.__last_lines_of_file(log_file_path, 1000):
            self.send_message(line)

        self.__open_log_file_at_end(log_file_path)

        #start periodic callback to tail file
        self._periodic_task = tornado.ioloop.PeriodicCallback(self.__follow_file_async, 3000)
        self._periodic_task.start()

    def __open_log_file_at_end(self, log_file_path):
        fsize = _os.stat(log_file_path).st_size
        self._file_handle = open(log_file_path, 'rb')
        self._file_handle.seek(fsize)

    # generator that yields
    #    the last 1000 lines of a log file
    #    all following lines until the file is terminated
    @tornado.web.asynchronous
    def __last_lines_of_file(self, log_file, lines=1000):
        bufsize = 8192
        fsize = _os.stat(log_file).st_size

        with open(log_file, 'rb') as f:
            #limit buffer size to file size
            if bufsize > fsize:
                bufsize = fsize-1
            position = fsize
            data = []
            while True:
                position -= bufsize
                bufsize *= 2
                if position < 0:
                    position = 0
                f.seek(position)
                data = f.readlines()
                # quit reading if we have read enough lines
                #  or we are at the beginning of the file
                if len(data) >= lines or position == 0 or bufsize > 131072:
                    break

            # send out last (up to) 1000 lines
            for line in data[-lines:]:
                yield line

    def __follow_file_async(self):
        ### start tailing log
        trailing = True
        while True:
            where = self._file_handle.tell()
            line = self._file_handle.readline()
            if line:
                if trailing and line in self.line_terminators:
                    # This is just the line terminator added to the end of the file
                    # before a new line, ignore.
                    trailing = False
                    continue

                if line[-1] in self.line_terminators:
                    line = line[:-1]
                    if line[-1:] == '\r\n' and '\r\n' in self.line_terminators:
                        # found crlf
                        line = line[:-1]

                trailing = False
                # send out new lines
                self.send_message(line)
            else:
                # preserve file position and return
                self._file_handle.seek(where)
                break
