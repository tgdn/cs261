# -*- coding: utf-8 -*-

import pytest
from purple import db

def test_reql_connection():
    # assume rethinkdb is running, otherwise connection will fail
    with db.get_reql_connection() as conn:
        assert conn.is_open() == True

def test_postgres():
    pass
    # drop tables and create new ones
    #db.Base.metadata.drop_all(db.engine)
    #db.Base.metadata.create_all(db.engine)
