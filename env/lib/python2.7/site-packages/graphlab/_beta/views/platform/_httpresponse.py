class _HTTPResponse(object):
    def __init__(self, body, status_code, content_type):
        self.body = body
        self.status_code = status_code
        self.content_type = content_type
