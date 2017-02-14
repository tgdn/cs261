# -*- coding: utf-8 -*-

from sqlalchemy import Column, Integer, BigInteger, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Database:

    @staticmethod
    def connect():
        pass


class BaseModel(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True)

class SymbolModel(BaseModel):
    __tablename__ = 'symbols'

    name = Column(String)


class TradeModel(BaseModel):
    __tablename__ = 'trades'

    price = Column(Float)
    size = Column(BigInteger)
