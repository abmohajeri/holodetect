import logging
import os
from .constraint import DenialConstraint


class Parser:
    def __init__(self):
        self.dc_strings = []
        self.dcs = []

    def load_denial_constraints(self, attrs, fpath):
        dc_file = open(fpath, 'r')
        for line in dc_file:
            line = line.rstrip()
            if not line or line.startswith('#'):  # Skip empty and comment lines.
                continue
            self.dc_strings.append(line)
            self.dcs.append(DenialConstraint(line, attrs))
        return self.dcs

    def get_dcs(self):
        return self.dcs
