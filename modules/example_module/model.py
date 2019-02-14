class LaneDetectionModel(object):

    def predict(self, img):
        return [[True]]

def build():
    return LaneDetectionModel()