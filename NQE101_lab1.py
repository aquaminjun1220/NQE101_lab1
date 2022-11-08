SCALE = 2
TRY = 4
TRAIN_LENGTH = 3000
TEST_LENGTH = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CLASSES = 3
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = TEST_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

import tensorflow as tf
import numpy as np
import skimage.draw as draw
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
from IPython.display import clear_output
import copy
import os

def draw_donut(point, radius, array, mask):
  thik = rng.integers(4*SCALE, 8*SCALE)
  disp = rng.integers(0, 2*SCALE)
  theta = rng.random() * 2*np.pi
  disp = np.array([disp * np.cos(theta), disp * np.sin(theta)])
  rro, cco = draw.disk(point+disp, radius+thik, shape=array.shape)
  rri, cci = draw.disk(point, radius, shape=array.shape)
  array[rro, cco] = 1
  mask[rro,cco] = 1
  array[rri, cci] = 0
  mask[rri,cci] = 0

def draw_circle(point, radius, array, mask):
  rr, cc = draw.disk(point, radius, shape=array.shape)
  array[rr, cc] = 1
  mask[rr,cc] = 2

def draw_incircle(point, radius, array, mask):
  rr, cc = draw.disk(point, radius, shape=array.shape)
  for r, c in zip(rr, cc):
    array[r, c] = 1
    if mask[r, c] == 1:
      continue
    mask[r, c] = 2

def draw_poly(r, c, array, mask):
  rr, cc = draw.polygon(r, c, shape=array.shape)
  array[rr, cc] = 1
  mask[rr,cc] = 2

def draw_inpoly(r, c, array, mask):
  rr, cc = draw.polygon(r, c, shape=array.shape)
  for rt, ct in zip(rr, cc):
    array[rt, ct] = 1
    if mask[rt, ct] == 1:
      continue
    mask[rt, ct] = 2

def gen_internal():
  point = rng.integers(60*SCALE, 70*SCALE, 2)
  radius = rng.integers(45*SCALE, 50*SCALE)
  donut = np.zeros((128*SCALE, 128*SCALE))
  mask = np.zeros((128*SCALE, 128*SCALE))
  draw_donut(point, radius, donut, mask)
  
  for _ in range(5):
    crad = rng.integers(5*SCALE, 10*SCALE)
    cpoint = copy.deepcopy(point)
    disp = rng.integers(3*SCALE, radius-crad)
    theta = rng.random() * 2*np.pi
    disp = np.array([disp * np.cos(theta), disp * np.sin(theta)])
    cpoint = cpoint + disp
    draw_circle(cpoint, crad, donut, mask)
    
  for _ in range(5):
    crad = rng.integers(8*SCALE, 10*SCALE)
    n = rng.integers(5, 8)
    cpoint = copy.deepcopy(point)
    disp = rng.integers(3*SCALE, radius-crad)
    theta = rng.random()*2*np.pi
    disp = np.array([disp * np.cos(theta), disp * np.sin(theta)])
    cpoint = cpoint + disp
    r = np.array([cpoint[0]]*n)
    c = np.array([cpoint[1]]*n)
    ccrad = rng.integers(2*SCALE, crad, n)
    ccrad [n-1] = crad
    ctheta = np.sort(rng.random(n)*2*np.pi)
    ctheta[n-1] = 0
    ctheta = ctheta + theta
    r = r + ccrad * np.cos(ctheta)
    c = c + ccrad * np.sin(ctheta)
    draw_poly(r, c, donut, mask)

  for _ in range(2):
    crad = rng.integers(7*SCALE, 13*SCALE)
    cpoint = copy.deepcopy(point)
    disp = radius+1*SCALE-crad
    theta = rng.random() * 2*np.pi
    disp = np.array([disp * np.cos(theta), disp * np.sin(theta)])
    cpoint = cpoint + disp
    draw_incircle(cpoint, crad, donut, mask)
  
  for _ in range(3):
    crad = rng.integers(8*SCALE, 13*SCALE)
    n = rng.integers(5, 8)
    cpoint = copy.deepcopy(point)
    disp = radius+2*SCALE-crad
    theta = rng.random()*2*np.pi
    disp = np.array([disp * np.cos(theta), disp * np.sin(theta)])
    cpoint = cpoint + disp
    r = np.array([cpoint[0]]*n)
    c = np.array([cpoint[1]]*n)
    ccrad = rng.integers(2*SCALE, crad, n)
    ccrad [n-1] = crad
    ctheta = np.sort(rng.random(n)*2*np.pi)
    ctheta[n-1] = 0
    ctheta = ctheta + theta
    r = r + ccrad * np.cos(ctheta)
    c = c + ccrad * np.sin(ctheta)
    draw_inpoly(r, c, donut, mask)

  for _ in range(3):
    crad = rng.integers(8*SCALE, 13*SCALE)
    n = 4
    cpoint = copy.deepcopy(point)
    ctheta = rng.random()*np.pi/8
    disp = radius-crad*np.cos(ctheta)
    theta = rng.random()*2*np.pi
    disp = np.array([disp * np.cos(theta), disp * np.sin(theta)])
    cpoint = cpoint + disp
    r = np.array([cpoint[0]]*n)
    c = np.array([cpoint[1]]*n)
    ccrad = np.array([crad]*4)
    ctheta = np.array([ctheta, np.pi-ctheta, np.pi + ctheta, -ctheta])
    ctheta = ctheta + theta
    r = r + ccrad * np.cos(ctheta)
    c = c + ccrad * np.sin(ctheta)
    draw_inpoly(r, c, donut, mask)

  return donut, mask

def traingen():
  i = 0
  while i < 3000:
    donut, mask = gen_internal()
    yield tf.stack([donut, tf.zeros((128*SCALE, 128*SCALE)), tf.zeros((128*SCALE, 128*SCALE))], axis=2), mask[...,tf.newaxis]

def testgen():
  i = 0
  while i < 1000:
    donut, mask = gen_internal()
    yield tf.stack([donut, tf.zeros((128*SCALE, 128*SCALE)), tf.zeros((128*SCALE, 128*SCALE))], axis=2), mask[...,tf.newaxis]

def init_dataset():
  train_images = tf.data.Dataset.from_generator(
              traingen,
              output_signature = (tf.TensorSpec(shape=(128*SCALE, 128*SCALE, 3), dtype=np.float32), tf.TensorSpec(shape=(128*SCALE, 128*SCALE, 1), dtype=np.float32)))
  test_images = tf.data.Dataset.from_generator(
              testgen,
              output_signature = (tf.TensorSpec(shape=(128*SCALE, 128*SCALE, 3), dtype=np.float32), tf.TensorSpec(shape=(128*SCALE, 128*SCALE, 1), dtype=np.float32)))
  train_batches = (
      train_images
      .cache()
      .batch(BATCH_SIZE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))
  test_batches = test_images.batch(BATCH_SIZE)
  return train_batches, test_batches

def display(display_list):
  plt.figure(figsize=(15, 15))
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
  plt.show()

def unet_model(output_channels:int):
  base_model = tf.keras.applications.MobileNetV2(input_shape=[128*SCALE, 128*SCALE, 3], include_top=False)
  layer_names = [
      'block_1_expand_relu',   # 256
      'block_3_expand_relu',   # 128
      'block_6_expand_relu',   # 64
      'block_13_expand_relu',  # 32
      'block_16_project',      # 16
  ]
  base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
  down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
  down_stack.trainable = False

  up_stack = [
      pix2pix.upsample(512, 3),  # 4x4 -> 8x8
      pix2pix.upsample(256, 3),  # 8x8 -> 16x16
      pix2pix.upsample(128, 3),  # 16x16 -> 32x32
      pix2pix.upsample(64, 3),   # 32x32 -> 64x64
  ]
  inputs = tf.keras.layers.Input(shape=[128*SCALE, 128*SCALE, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def load_latest(save_filepath='./savedmodel/{TRY}{SCALE}'.format(TRY=TRY, SCALE=SCALE)):
  exp = 0
  try:
      latest = tf.train.latest_checkpoint(save_filepath)
      latest_bas = os.path.basename(latest)
      exp = int(latest_bas[3:6]) + int(latest_bas[7:9])
      model.load_weights(latest)
      print("### Loaded from {latest}!!  ###".format(latest=latest))
  except:
      print("### No previous checkpoints found at {save_filepath}!!  ###".format(save_filepath=save_filepath))
  return exp



def create_mask(pred_mask):
  #pred_mask = (batchsize, 128, 128, 3)
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask #pred_mask = (batchsize, 128, 128, 1)

def raw2images(filename="./volume1_Wednesday(512x512x512).raw", shape=None):
  if shape==None:
    bas = os.path.basename(filename)
    bas = bas[-16:-5]
    shape = (int(bas[8:11]), int(bas[0:3]), int(bas[4:7]))
  images = np.fromfile(filename, dtype=np.float32)
  images = np.reshape(images, shape)
  return images

def images2dataset(images):
  shape = images.shape
  images = 0.025 <= images
  images = np.stack([images, np.zeros(shape), np.zeros(shape)], axis=-1)
  images = tf.image.resize(images, (128*SCALE, 128*SCALE), method='nearest')
  masks = tf.zeros((shape[0], 128*SCALE, 128*SCALE, 1))
  dataset = tf.data.Dataset.from_tensor_slices((images, masks))
  dataset = dataset.batch(64)
  return dataset

def take_sample(dataset):
  npiter = dataset.as_numpy_iterator()
  batch1 = next(npiter)
  sample_image = batch1[0][20]
  sample_mask = batch1[1][20]
  return sample_image, sample_mask

def show_predictions(dataset=None, num=1):
  #dataset = numbatch*((batchsize, 128, 128, 3), ( batchsize, 128, 128, 1))
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = create_mask(model.predict(image))
      display([image[0], mask[0], pred_mask[0]])
  else:
    pred_mask = create_mask(model.predict(sample_image[tf.newaxis, ...]))
    display([sample_image, sample_mask, pred_mask[0]])

def my_show(dataset = None, num = 1):
  #dataset = (numbatch, batchsize, 128, 128, 3), (numbatch, batchsize, 128, 128, 1)
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = create_mask(model.predict(image))
      mask = pred_mask == 2
      display([image[0], mask[0], tf.cast(image[0], dtype=bool)&mask[0]])
  else:
    pred_mask = create_mask(model.predict(sample_image[tf.newaxis, ...]))
    mask = pred_mask == 2
    display([sample_image, sample_mask, tf.cast(sample_image, dtype=bool)&mask[0]])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions(test_batches, 1)
    my_show(test_batches, 1)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

if __name__=='__main__':
  rng = np.random.default_rng(557)
  train_batches, test_batches = init_dataset()
  # (num_batch, ((batch_size, 256, 256, 3), (batch_size, 256, 256, 1)))
  """test_batch1 = test_batches.take(1)
  for batch1 in test_batch1:
    images = batch1[0]
    masks = batch1[1]
    for image, mask in zip(images, masks):
      display([image, mask])"""

  model = unet_model(output_channels=OUTPUT_CLASSES)
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  exp = load_latest()
  images = raw2images()
  dataset = images2dataset(images)
  sample_image, sample_mask = take_sample(dataset)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./savedmodel/{TRY}{SCALE}/cp_{exp:03d}+{{epoch:02d}}.ckpt'.format(TRY=TRY, SCALE=SCALE, exp=exp), monitor='loss', verbose=1, save_weights_only=True, save_best_only=False)
  model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_batches,
                            callbacks=[cp_callback])
  show_predictions(dataset, 10)
