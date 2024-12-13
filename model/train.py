import boto3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up S3
s3 = boto3.client('s3')
bucket_name = 'mlops-image-dataset'
local_data_dir = '/tmp/dataset'

# Download dataset
def download_dataset():
    objects = s3.list_objects(Bucket=bucket_name)['Contents']
    for obj in objects:
        s3.download_file(bucket_name, obj['Key'], f"{local_data_dir}/{obj['Key']}")

# Create Model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    download_dataset()

    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        f"{local_data_dir}/train",
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    model = create_model()
    model.fit(train_generator, epochs=10)
    model.save('model.h5')

    # Upload the trained model to S3
    s3.upload_file('model.h5', bucket_name, 'model/model.h5')
