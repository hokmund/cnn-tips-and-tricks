import numpy as np
import seaborn as sns

from skimage import io
from matplotlib import pyplot as plt

from file_utils import get_file, get_files, get_class_dir

def show_image(data_dir, class_id, image_id):
    image_name = get_files(data_dir, class_id)[image_id]
    image_file = get_file(data_dir, class_id, image_name)
    io.imshow(io.imread(image_file))

def plot_class_distribution(classes):
    plt.figure(figsize=(30, 6))
    ax = sns.barplot(x=classes.index, y=classes)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_history(history, offset=0):
    best_acc = np.max(history.history['val_acc'])
    best_loss = np.min(history.history['val_loss'])
    print(f'Best accuracy: {best_acc}')
    print(f'Best logloss: {best_loss}')

    plt.plot(history.history['acc'][offset:])
    plt.plot(history.history['val_acc'][offset:])
    plt.plot(np.argmax(history.history["val_acc"]) - offset, best_acc, 'o', color='orange')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'][offset:])
    plt.plot(history.history['val_loss'][offset:])
    plt.plot(np.argmin(history.history["val_loss"]) - offset, best_loss, 'o', color='orange')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def plot_lr(history, window=None):
    best_loss = np.min(history['loss'])
    best_lr = history['lr'][np.argmin(history['loss'])]
    print(f'Best loss: {best_loss} (at {best_lr})')

    plt.figure(figsize=(16, 5))
    
    if window is None:
        plt.plot(history['lr'], history['loss'])
        plt.plot(best_lr, best_loss, 'o', markersize=8, color='C0')
    else:
        smoothed_losses = []
        avg_loss = 0

        smoothed_losses = pd.Series(history['loss']).rolling(window=window).mean()

        plt.plot(history['lr'], smoothed_losses)
        plt.plot(history['lr'][np.argmin(smoothed_losses)], np.min(smoothed_losses), 'o', markersize=7, color='C0')
        
    plt.title('model loss')
    plt.xlabel('learning rate')
    plt.legend(['train loss', 'best loss'], loc='upper left')
    plt.xscale('log')
    plt.show()    

def show_augmentation(augmentation):    
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    from keras.preprocessing import image

    train_datagen = image.ImageDataGenerator(preprocessing_function=augmentation)

    img = load_img('images/demo_img.jpg')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    images = [x / 127.5 - 1]

    for batch in train_datagen.flow(x, batch_size=1):
        images.append(batch)
        i += 1

        if i == 15:
            break

    max_images = 16
    
    scale = 4
    
    grid_width = 4
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width*scale, grid_height*scale))
    for i, img in enumerate(images):
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img[0])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
    return images