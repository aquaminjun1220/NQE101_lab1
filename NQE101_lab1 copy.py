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
  #pred_mask = (batchsize, 128, 128, 3) or (numbatch, batchsize, 128, 128, 3)
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask #pred_mask = (batchsize, 128, 128, 1)

def raw2images(filename="./volume1_Wednesday_512x512x512.raw", shape=None):
  if shape==None:
    bas = os.path.basename(filename)
    bas = bas[-15:-4]
    shape = (int(bas[8:11]), int(bas[0:3]), int(bas[4:7]))
  images = np.fromfile(filename, dtype=np.float32)
  images = np.reshape(images, shape)
  return images

def images2dataset(images):
  shape = images.shape
  images = 0.02 <= images
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

  # Q 2-i
  """print(images.min())
  print(images.max())"""
  # figure 3
  """fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
  im = ax.flat[0].imshow(images[0], vmin=0, vmax=0.26)
  ax.flat[0].set_title("1st image of Volume 1")
  im = ax.flat[1].imshow(images[0,80:130,80:130], vmin=0, vmax=0.26)
  ax.flat[1].set_title("1st image of Volume 1 (magnified)")
  cax = fig.add_axes([ax[1].get_position().x1+0.02,ax[1].get_position().y0,0.02,ax[1].get_position().y1-ax[1].get_position().y0])
  fig.colorbar(im, cax=cax)
  plt.show()"""

  # Q 2-ii
  """air = 0.02
  content = 0.05
  airs = images < air
  contents = (air <= images) & (images < content)
  containers = content <= images
  fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
  im = ax.flat[0].imshow(airs[0])
  ax.flat[0].set_title("1st image of Volume 1 (air)")
  im = ax.flat[1].imshow(contents[0])
  ax.flat[1].set_title("1st image of Volume 1 (content)")
  im = ax.flat[2].imshow(containers[0])
  ax.flat[2].set_title("1st image of Volume 1 (container)")
  plt.show()"""

  # Q 2-ii
  """train_batches, test_batches = init_dataset()
  test_batch1 = test_batches.take(1)
  for batch1 in test_batch1:
    images = batch1[0]
    masks = batch1[1]
  
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
  im = ax.flat[0].imshow(images[0])
  ax.flat[0].set_title("Instance of dataset image")
  im = ax.flat[1].imshow(masks[0])
  ax.flat[1].set_title("Instance of dataset mask")
  plt.show()"""

  # Q 2-ii
  """dataset = images2dataset(images)
  model = unet_model(output_channels=OUTPUT_CLASSES)
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  exp = load_latest()
  predictions = model.predict(dataset)
  pred_masks = create_mask(predictions)
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
  im = ax.flat[0].imshow(images[0], vmin=0, vmax=0.26)
  ax.flat[0].set_title("1st image of Volume 1")
  im = ax.flat[1].imshow(pred_masks[0])
  ax.flat[1].set_title("Predicted mask for 1st image")
  im = ax.flat[2].imshow(images[119], vmin=0, vmax=0.26)
  ax.flat[2].set_title("120th image of Volume 1")
  im = ax.flat[3].imshow(pred_masks[119])
  ax.flat[3].set_title("Predicted mask for 120th image")
  plt.show()"""

  # Q 2-iii
  """# volume of each voxel (in L)
  voxel_vol_l = 0.0194*0.0194*0.035/1000
  # load images into dataset (with shape (num_batch, batch_size, 256, 256, 3))
  dataset = images2dataset(images)
  # initialize model
  model = unet_model(output_channels=OUTPUT_CLASSES)
  model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  # load model weights from checkpoint
  exp = load_latest()
  # produce prediction masks from dataset via model
  predictions = model.predict(dataset)
  pred_masks = create_mask(predictions)
  # count voxels with value 2 (content)
  pred_cont = pred_masks==2
  pred_cont_voxel = np.sum(pred_cont)
  # calculate content volume by multiplying volume pre voxel
  pred_cont_vol_l = pred_cont_voxel * voxel_vol_l * SCALE**2
  print(pred_cont_vol_l)"""

  # Q 2-iii
  """dataset = images2dataset(images)
  model = unet_model(output_channels=OUTPUT_CLASSES)
  model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
  exp = load_latest()
  predictions = model.predict(dataset)
  pred_masks = create_mask(predictions)

  masks = pred_masks == 0
  por = np.empty((64,), np.float32)
  i = 0
  voxel_area = 32 * 55 * 55
  for i in range(16):
    cont_voxel_area1 = np.sum(masks[32*i:32*(i+1),65:120,65:120,:])
    cont_voxel_area2 = np.sum(masks[32*i:32*(i+1),65:120,120:175,:])
    cont_voxel_area3 = np.sum(masks[32*i:32*(i+1),120:175,65:120,:])
    cont_voxel_area4 = np.sum(masks[32*i:32*(i+1),120:175,120:175,:])
    por[4*i] = 100*cont_voxel_area1 / voxel_area
    por[4*i+1] = 100*cont_voxel_area2 / voxel_area
    por[4*i+2] = 100*cont_voxel_area3 / voxel_area
    por[4*i+3] = 100*cont_voxel_area4 / voxel_area

  print(np.average(por))
  plt.hist(por, bins=10)
  plt.xlabel("porosity (%)")
  plt.ylabel("occurence")
  plt.title("Porositiy of different sections")
  plt.show()"""

  # Q 2-iv
  """sizes = np.array([[140, 135] ,[120, 135] ,[88, 156] ,[69, 140] ,[123, 113] ,[102, 140] ,[115, 148] ,[137, 135] ,[129, 145] ,[166, 124] ,[77, 101] ,[166, 100] ,[112, 171] ,[161, 91] ,[167, 174]])"""

  # Q 3
  """images = raw2images(filename="D:/GoogleDrive/KSA/2022/HP/6/NQE/lab1/volume2_768x768x159.raw")
  print(images.min())
  print(images.max())"""

  # Q 4
  images = raw2images(filename="D:/GoogleDrive/KSA/2022/HP/6/NQE/lab1/volume3_512x512x348.raw")
  masks = 200 <= images
  plt.imshow(masks[159])
  plt.title("160th image of Volume 3 (thresholded)")
  plt.show()
  num_cont_voxel = np.sum(masks[130:180, 235:335, 235:335])
  print(num_cont_voxel/(30*100*100))