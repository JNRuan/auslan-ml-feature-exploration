################################################################################
# Efficient Net retriever.
################################################################################
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, \
    EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6,EfficientNetB7

################################################################################

def get_efficientnet(model_num=0, input_size=(224, 224, 3), finetune=False, tune_layers=3):
    model = None
    if model_num == 0:
        model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 1:
        model = EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 2:
        model = EfficientNetB2(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 3:
        model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 4:
        model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 5:
        model = EfficientNetB5(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 6:
        model = EfficientNetB6(include_top=False, weights='imagenet', input_shape=input_size)
    elif model_num == 7:
        model = EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_size)
    else:
        print("Model not found, ensure model number is in range [0, 7].")

    if not finetune:
        model.trainable = False
    else:
        for layer in model.layers[:-tune_layers]:
            layer.trainable = False
    return model