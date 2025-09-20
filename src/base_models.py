import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50


# ---- VGG16 ----
def load_vgg16(input_shape=(224, 224, 3)):
    return VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

# ---- ResNet50 ----
def load_resnet50(input_shape=(224, 224, 3)):
    """
    Carga un modelo ResNet50 pre-entrenado en ImageNet.
    
    El modelo se carga sin la capa superior de clasificación para ser
    utilizado como backbone para fine-tuning.
    """
    return ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)