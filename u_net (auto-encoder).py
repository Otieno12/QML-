import os

# Create lists to hold paths for images and masks
image_paths = []
mask_paths = []

# Traverse through the directory and collect paths
for dirname, _, filenames in os.walk('/content/ordinal_nets_lc/lc_pictures/8cb(segunda ordem)/8CB_1'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        if filename.lower().endswith('.jpeg'):
            if filename[0:4] == 'mask':
                mask_paths.append(path)
            else:
                image_paths.append(path)

# Create directories for images and masks
!mkdir -p images
!mkdir -p masks

# Copy images to 'images' directory
for path in image_paths:
    !cp "{path}" images

# Copy masks to 'masks' directory
for path in mask_paths:
    !cp "{path}" masks

# Verify the contents of 'images' and 'masks' directories
!ls images
!ls masks

number=120

images_dir='images'
masks_dir='images


images_listdir = sorted(os.listdir(images_dir))
masks_listdir = sorted(os.listdir(masks_dir))
N=list(range(9))
random_N = N



image_size=512
input_image_size=(512,512)


def read_image(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(image_size,image_size))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    return img


import matplotlib.pyplot as plt
import cv2

rows = 3
cols = 3
fig, ax = plt.subplots(rows, cols, figsize=(10, 10))

# Assuming random_N is the list you want to iterate over
for i, ax in enumerate(ax.flat):
    if i < min(len(random_N), len(images_listdir)):
        img = read_image(f"{images_dir}/{images_listdir[i]}")
        ax.set_title(f"{images_listdir[i]}")
        ax.imshow(img)
        ax.axis('off') 

print(len(images_listdir))
print(len(masks_listdir))


fig, ax = plt.subplots(rows, cols, figsize = (10,10))
for i, ax in enumerate(ax.flat):
    if i < len(random_N):
        if os.path.exists(os.path.join(masks_dir,masks_listdir[i])):
            img = read_image(f"{masks_dir}/{masks_listdir[i]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ax.set_title(f"{masks_listdir[i]}")
            ax.imshow(img)
            ax.axis('off')
        else:
            print('not exist')



MASKS=np.zeros((1,image_size, image_size, 1), dtype=bool)
IMAGES=np.zeros((1,image_size, image_size, 3),dtype=np.uint8)

for j,file in enumerate(images_listdir):   ##the smaller, the faster
    try:
        image = read_image(f"{images_dir}/{file}")
        image_ex = np.expand_dims(image, axis=0)
        IMAGES = np.vstack([IMAGES, image_ex])
        mask = read_image(f"{masks_dir}/{masks_listdir[j]}") 
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.reshape(512,512,1)
        mask_ex = np.expand_dims(mask, axis=0)    
        MASKS = np.vstack([MASKS, mask_ex])
    except:
        print(file)
        continue


images=np.array(IMAGES)[1:number+1]
masks=np.array(MASKS)[1:number+1]
print(images.shape,masks.shape)


from sklearn.model_selection import train_test_split
images_train, images_test, masks_train, masks_test = train_test_split(
    images, masks, test_size=0.25, random_state=42)


import tensorflow as tf

def conv_block(input, num_filters):
    conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    conv = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation("relu")(conv)
    return conv

def encoder_block(input, num_filters):
    skip = conv_block(input, num_filters)
    pool = tf.keras.layers.MaxPool2D((2,2))(skip)
    return skip, pool

def decoder_block(input, skip, num_filters):
    up_conv = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(input)
    conv = tf.keras.layers.Concatenate()([up_conv, skip])
    conv = conv_block(conv, num_filters)
    return conv

def Unet(input_shape):
    inputs = tf.keras.layers.Input(input_shape)

    skip1, pool1 = encoder_block(inputs, 64)
    skip2, pool2 = encoder_block(pool1, 128)
    skip3, pool3 = encoder_block(pool2, 256)
    skip4, pool4 = encoder_block(pool3, 512)

    bridge = conv_block(pool4, 1024)

    decode1 = decoder_block(bridge, skip4, 512)
    decode2 = decoder_block(decode1, skip3, 256)
    decode3 = decoder_block(decode2, skip2, 128)
    decode4 = decoder_block(decode3, skip1, 64)
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(decode4)
    model = tf.keras.models.Model(inputs, outputs, name="U-Net")
    return model

unet_model = Unet((512,512,3))
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
unet_model.summary()


unet_result = unet_model.fit(
    images_train, masks_train, 
    validation_split = 0.2, batch_size = 4, epochs = 100)


def show_result(idx, og, unet, target, p):
    
    fig, axs = plt.subplots(1, 3, figsize=(12,12))
    axs[0].set_title("Original "+str(idx) )
    axs[0].imshow(og)
    axs[0].axis('off')
    
    axs[1].set_title("U-Net: p>"+str(p))
    axs[1].imshow(unet)
    axs[1].axis('off')
    
    axs[2].set_title("Ground Truth")
    axs[2].imshow(target)
    axs[2].axis('off')

    plt.show()


unet_predict = unet_model.predict(images_test)
len(images_test)

r1,r2,r3,r4=0.7,0.8,0.9,0.99


unet_predict1 = (unet_predict > r1).astype(np.uint8)
unet_predict2 = (unet_predict > r2).astype(np.uint8)
unet_predict3 = (unet_predict > r3).astype(np.uint8)
unet_predict4 = (unet_predict > r4).astype(np.uint8)


show_test_idx = random.sample(range(len(unet_predict)), 3)
for idx in show_test_idx: 
    show_result(idx, images_test[idx], unet_predict1[idx], masks_test[idx], r1)
    show_result(idx, images_test[idx], unet_predict2[idx], masks_test[idx], r2)
    show_result(idx, images_test[idx], unet_predict3[idx], masks_test[idx], r3)
    show_result(idx, images_test[idx], unet_predict4[idx], masks_test[idx], r4)
    print()
