import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_dataframe_from_split_dirs(base_dir):
    """
    Creates a DataFrame with filepaths, labels, and set (Training/Testing).
    Assumes structure:
    base_dir/
      ├── Training/
      │   └── class_name/
      └── Testing/
          └── class_name/
    """
    filepaths = []
    sets = []
    classes = []

    for split in ['Training', 'Testing']:
        split_path = os.path.join(base_dir, split)

        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(split, class_folder, img_file))
                    sets.append(split)
                    classes.append(class_folder.lower().strip())

    df = pd.DataFrame({
        'filename': filepaths,
        'set': sets,
        'label': classes
    })

    return df

def prepare_generators(base_dir, input_size=(299, 299), batch_size=32, seed=22):
    """
    Combines dataframe creation, train/val/test split, and generator creation.

    Returns:
    - train_gen, val_gen, test_gen
    - class_dict: label mapping
    """
    df = get_dataframe_from_split_dirs(base_dir)

    train_df = df[df['set'] == 'Training'].copy()
    test_df = df[df['set'] == 'Testing'].copy()

    train_data, val_data = train_test_split(
        train_df, test_size=0.2, stratify=train_df['label'], random_state=seed
    )

    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    datagen_val_test = ImageDataGenerator(rescale=1./255)

    train_gen = datagen_train.flow_from_dataframe(
        train_data, base_dir,
        x_col='filename', y_col='label',
        target_size=input_size, batch_size=batch_size,
        class_mode='categorical', shuffle=True, seed=seed
    )

    val_gen = datagen_val_test.flow_from_dataframe(
        val_data, base_dir,
        x_col='filename', y_col='label',
        target_size=input_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )

    test_gen = datagen_val_test.flow_from_dataframe(
        test_df, base_dir,
        x_col='filename', y_col='label',
        target_size=input_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )

    class_dict = train_gen.class_indices

    return train_gen, val_gen, test_gen, class_dict
