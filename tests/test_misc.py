# -*- coding: utf-8 -*-

import pytest
from purple.finance import Trade

TRADE_ROW = '2017-01-13 15:26:41.917266,w.tuffnell@janestreetcap.com,j.newbury@citadel.com,469.74,15952,GBX,AV.L,Financial,469.08,469.74'

def test_trade_parse_err():
    t = Trade(None)
    assert t.parse_err == True

def test_trade_parser():
    t = Trade(TRADE_ROW)
    assert t.parse_err == False

def test_trade_parse_attr():
    # check number is correctly parsed (float)
    t = Trade(TRADE_ROW)
    assert t.price == 469.74
    