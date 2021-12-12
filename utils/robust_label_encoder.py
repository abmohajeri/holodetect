from sklearn.preprocessing import LabelEncoder
import bisect
import logging
import numpy as np


class LabelEncoderRobust(LabelEncoder):
    def __init__(self):
        self.class_type = None

    def fit(self, y):
        super().fit(y)
        le_classes = self.classes_.tolist()
        logging.debug("LER classes: {}".format(le_classes))
        if len(le_classes) > 0:
            self.class_type = type(le_classes[0])
        logging.debug("LER classes type: {}".format(self.class_type))
        if self.class_type == str:
            bisect.insort_left(le_classes, 'UNKNOWN_LBL')
        if self.class_type == int:
            bisect.insort_left(le_classes, -999)
        self.classes_ = np.array(le_classes)

    def transform(self, y):
        for i in range(len(y)):
            item = y[i]
            if item not in self.classes_:
                logging.debug("transform LER classes type: {}".format(self.class_type))
                if self.class_type == str:
                    y[i] = 'UNKNOWN_LBL'
                elif self.class_type == int:
                    y[i] = -999
                else:
                    print(self.class_type)
                    print(item)
                    raise ValueError("list_type in None, cannot transform")
        return super().transform(y)