from keras.src.legacy.preprocessing.image import ImageDataGenerator
from src.load_data import train_df, test_df, dataset_path

# Parameters for ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_df["image_id"] = train_df["image_id"].astype(str) + ".jpg"

# Making generator for training and validation images
train_generator = datagen.flow_from_dataframe(
    train_df, directory=dataset_path,
    x_col='image_id', y_col='dx',
    target_size=(224, 224), class_mode='categorical', batch_size=32, subset='training',
)

validation_generator = datagen.flow_from_dataframe(
    train_df, directory=dataset_path,
    x_col='image_id', y_col='dx',
    target_size=(224, 224), class_mode='categorical', batch_size=32, subset='validation',
)
