import numpy as np

class Map:
    def __init__(self):
        self.spacemarks = dict()


    def add_spacemark(self, id, spacemark):
        self.spacemarks[id] = spacemark


    def remove_spacemark(self, id):
        del self.spacemarks[id]