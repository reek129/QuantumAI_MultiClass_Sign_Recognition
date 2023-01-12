

import pickle

class PickleHelper:
    def __init__(self,path):
        super()
        self.path = path
    def save_pkl(self, data, filename):
        f = open(self.path+filename, 'wb')
        pickle.dump(data, self.path+filename)
        f.close()

    def load_pkl(filename):
        f = open(self.path+filename, 'rb')
        obj = pickle.load(filename)
        f.close()
        return obj

