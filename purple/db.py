# -*- coding: utf-8 -*-

from contextlib import closing
from sqlalchemy import (
    MetaData,
    create_engine,
    Column,
    Integer,
    BigInteger,
    Float,
    String,
    Boolean,
    ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_SETTINGS = {
    'drivername': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'username': 'purple',
    'password': 'purple123',
    'database': 'cs261'
}

# create database engine and setup session
engine = create_engine(URL(**DATABASE_SETTINGS))
Base = declarative_base(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()

def create_tables():
    # will create all tables
    Base.metadata.create_all(engine)

def drop_tables():
    # will delete all data in the tables
    Base.metadata.drop_all(engine)

######################################
#               Models               #
######################################

class BaseModel(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True)

class SymbolModel(BaseModel):
    __tablename__ = 'symbols'
    name = Column(String)
    trades = relationship('TradeModel', back_populates='symbol')

    @classmethod
    def get_or_create(cls, name):
        # retrieve symbol or create a new one and return
        obj = session.query(cls).filter_by(name=name).one_or_none()
        if not obj:
            obj = cls(name=name)
            session.add(obj)
            session.commit()
        return obj


class TradeModel(BaseModel):
    '''
    Table that holds data from each trade.
    Add required attributes below.
    '''
    __tablename__ = 'trades'

    price = Column(Float)
    size = Column(BigInteger)
    flagged = Column(Boolean, default=False)
    symbol_id = Column(Integer, ForeignKey('symbols.id'))

    symbol = relationship('SymbolModel', back_populates='trades')

    def flag(truth_value):
        self.flagged = truth_value
        session.commit()
