# Copyright 2012-2013 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import logging
from bcdoc.docstringparser import DocStringParser
from bcdoc.style import ReSTStyle

LOG = logging.getLogger('bcdocs')


class ReSTDocument(object):

    def __init__(self, target='man'):
        self.style = ReSTStyle(self)
        self.target = target
        self.parser = DocStringParser(self)
        self.keep_data = True
        self.do_translation = False
        self.translation_map = {}
        self.hrefs = {}
        self._writes = []
        self._last_doc_string = None

    def _write(self, s):
        if self.keep_data and s is not None:
            self._writes.append(s)

    def write(self, content):
        """
        Write content into the document.
        """
        self._write(content)

    def writeln(self, content):
        """
        Write content on a newline.
        """
        self._write('%s%s\n' % (self.style.spaces(), content))

    def peek_write(self):
        """
        Returns the last content written to the document without
        removing it from the stack.
        """
        return self._writes[-1]

    def pop_write(self):
        """
        Removes and returns the last content written to the stack.
        """
        return self._writes.pop()

    def push_write(self, s):
        """
        Places new content on the stack.
        """
        self._writes.append(s)

    def getvalue(self):
        """
        Returns the current content of the document as a string.
        """
        if self.hrefs:
            self.style.new_paragraph()
            for refname, link in self.hrefs.items():
                self.style.link_target_definition(refname, link)
        return ''.join(self._writes).encode('utf-8')

    def translate_words(self, words):
        return [self.translation_map.get(w, w) for w in words]

    def handle_data(self, data):
        if data and self.keep_data:
            self._write(data)

    def include_doc_string(self, doc_string):
        if doc_string:
            try:
                start = len(self._writes)
                self.parser.feed(doc_string)
                end = len(self._writes)
                self._last_doc_string = (start, end)
            except Exception:
                LOG.debug('Error parsing doc string', exc_info=True)
                LOG.debug(doc_string)

    def remove_last_doc_string(self):
        # Removes all writes inserted by last doc string
        if self._last_doc_string is not None:
            start, end = self._last_doc_string
            del self._writes[start:end]
