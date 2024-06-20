from prepare_data import *

labels = convert_bbox2labels((0.0, 0.4765626,0.382812,0.2968749,0.278125))
for label in labels:
    for i in label:
        print(i)