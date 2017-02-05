# -*- coding: utf-8 -*-

import os
import sys
import fcntl

from purple.finance import Trade

class App:
    def __init__(self, argv):
        self.argv = argv

        if (len(argv) == 2):
            filename = argv[1]

            with open(filename, 'r') as f:
                fd = f.fileno()
                flag = fcntl.fcntl(fd, fcntl.F_GETFD)
                fcntl.fcntl(fd, fcntl.F_SETFL, flag |  os.O_NONBLOCK)
                lineno = 0

                for line in f:
                    if lineno == 0:
                        lineno = 1
                        continue

                    t = Trade(line)
                    print '{} {} {}'.format(t.currency, t.price, t.sector)
        else:
            sys.stderr.write('Specify a filename: python main.py [filename]')
