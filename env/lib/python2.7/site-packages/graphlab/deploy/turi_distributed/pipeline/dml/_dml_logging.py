from graphlab.util import file_util
import logging
import threading
import time

import sys
if sys.version_info.major == 2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen


__logger__ = logging.getLogger(__name__)


class LogStream(object):
    def __init__(self, server_url, file_url, outstream):
        """
        Parameters
        ----------
        server_url : str
          URL endpoint of the metric server

        file_url : str
          URL of the complete log file
        """
        self.out = outstream
        self.server_url = server_url
        self.file_url = file_url
        self.total_received_message = ""
        self.server_alive = True

    def receive_next(self):
        if (self.server_alive):
            self.server_alive = self.receive_from_server()
            return True
        else:
            self.receive_from_file()
            return False

    def receive_from_server(self):
        try:
            content = urlopen(self.server_url).read()
            if len(content):
                self.out.write(content)
                self.out.flush()
            self.total_received_message += content
            return True
        except:
            pass
        return False

    def receive_from_file(self):
        try:
            if file_util.exists(self.file_url):
                __logger__.debug("Read from %s" % self.file_url)
                content = file_util.read(self.file_url)
                leftover_progress_content = content[len(self.total_received_message):]
                # Final log file incomplete
                if not leftover_progress_content.endswith('$'):
                    return False

                if len(leftover_progress_content):
                    self.out.write(leftover_progress_content[:-1])  # ignore $
                    self.out.flush()
                self.total_received_message += leftover_progress_content
                return True
        except Exception as e:
            __logger__.warning(e)
        return False


class LogPrinter(object):
    """
    The log printer streams log from the dml cluster, and
    print to the console.

    The log stream comes from two sources:
    1. the metric server while the job is running
    2. a final log file in hdfs when the job is done.

    While running, the log printer will first try print from metric
    server first. Content from the metric server will be
    accumulated into local variable: total_received_message.

    If no message from the metric server, we check the final log
    file. If final log file is avaiable, we print the remaining
    message from the log file and be done.
    """
    def __init__(self):
        self.finish = False
        self.streams = []
        self.thread = None

    def add_stream(self, stream):
        self.streams.append(stream)

    def __del__(self):
        try:
            self.stop()
        except:
            pass

    def start(self):
        def thread_fun():
            while (not self.finish):
                time.sleep(.5)
                new_streams = []
                for s in self.streams:
                    if s.receive_next():
                        new_streams.append(s)
                self.streams = new_streams
                if len(self.streams) == 0:
                    break

        self.thread = threading.Thread(target=thread_fun)
        self.thread.start()
        return True

    def stop(self):
        self.finish = True
        if self.thread:
            self.thread.join()


def get_log_metric_server_address(log_server_address_file, timeout=120):
    starttime = time.time()
    try:
        while(not file_util.exists(log_server_address_file)):
            time.sleep(.05)
            if (time.time() - starttime) > timeout:
                __logger__.warning('Unable to get server log (timeout reached)')
                return ""
        ret_str = file_util.read(log_server_address_file)
        if ret_str.endswith('$'):
            return ret_str[:-1]
    except Exception as e:
        __logger__.warning(e)
    return ""
