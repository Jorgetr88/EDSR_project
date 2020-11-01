import os
import matplotlib.pyplot as plt
import tensorflow as tf
from model.edsr import edsr

from model import resolve_single
from utils import load_image, plot_sample, plot_sample_no_treatment

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# Number of residual blocks
depth = 16

# Super-resolution factor
scale = 4

# Downgrade operator
downgrade = 'bicubic'

# Location of model weights (needed for demo)
weights_dir = f'weights/article'
weights_file = os.path.join(weights_dir, 'weights.h5')

##Run the demo

def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)
    #plot_sample_no_treatment(lr)
    sr = resolve_single(model,lr)
    plot_sample(lr,sr)

model = edsr(scale=scale, num_res_blocks=depth)
model.load_weights(weights_file)

resolve_and_plot('demo/image_0001.png')
