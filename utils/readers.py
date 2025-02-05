

from pyzbar.pyzbar import decode

class zbar_reader:
    def __init__(self):
        self.result = []
    def work(self,img):
        self.result = decode(img)
    def create(self):
        pass
    def destroy(self):
        pass
    def get_decoded_results(self):
        return self.result
    def __len__(self):
        return len(self.result)