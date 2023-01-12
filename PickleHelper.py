

import pickle

class PickleHelper:
    def __init__(self,path):
        super()
        self.path = path
    def save_pkl(self, data, filename):
        f = open('store.pckl', 'wb')
        pickle.dump(data, self.path+filename)
        f.close()

    def load_pkl(filename):
        f = open('store.pckl', 'rb')
        obj = pickle.load(filename)
        f.close()
        return obj

