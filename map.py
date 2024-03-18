import numpy as np

class Frame:
    def __init__(self):
        self.spacemarks = dict()


    def add_spacemark(self, id, spacemark):
        self.spacemarks[id] = spacemark


    def remove_spacemark(self, id):
        del self.spacemarks[id]


class Map:
    def __init__(self):
        self.frames = []

    
    def add_frame(self, frame):
        self.frames.append(frame)