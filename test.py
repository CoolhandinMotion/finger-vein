
import numpy as np
from model import Model
from control import Control

# test_url = r"E:\PycharmProjects\Dr-Kauba-code\02_002_R-DORSAL_02_TI.png"
test_url = r"E:\PycharmProjects\ISMS-github\sample_input_pictures\interesting-people.jpg"

k_array = np.asarray([1, 1, 1])
tile_size = 250

model = Model(image_url=test_url,
              tile_size=tile_size,
              k_adjustments=k_array)

isms_finger_vein = Control(model=model,merge_threshold=5)
isms_finger_vein.run()

