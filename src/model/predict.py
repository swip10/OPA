import numpy as np
import tensorflow as tf
from keras.saving.saving_api import load_model
from pathlib import Path

the_script = Path(__file__)
x = np.array(
    [[i for i in range(0, 59)], [10+i for i in range(0, 59)]]
).T
x = np.expand_dims(x, axis=0)
x = tf.convert_to_tensor(x, np.float32)

model = load_model(the_script.parents[2] / "models" / "001_close_volume" / "keras_next")

y = model.predict(x)
print('fin', y)
