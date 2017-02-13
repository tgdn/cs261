#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from purple import App

def main():
    parser = argparse.ArgumentParser(description='Purple trade backend')
    parser.add_argument('--init-db', action='store_true')
    parser.add_argument('--reset-db', action='store_true')
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-s', '--stream-url', type=str)
    parser.add_argument('-p', '--port', type=int)

    args = parser.parse_args()
    App(args)

if __name__ == '__main__':
    main()
