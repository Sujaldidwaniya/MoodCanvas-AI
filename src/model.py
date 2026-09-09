from tensorflow.keras import layers, models, Input
from config import IMG_SIZE, NUM_CLASSES

def build_hybrid_model(feature_dim):
    image_input = Input(shape=(*IMG_SIZE, 3), name="image_input")
    x = layers.Conv2D(32, (3,3), activation='relu')(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', name="last_conv")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    cnn_output = layers.Dropout(0.3)(x)

    feature_input = Input(shape=(feature_dim,), name="feature_input")
    y = layers.Dense(64, activation='relu')(feature_input)
    y = layers.Dense(32, activation='relu')(y)
    ann_output = layers.Dropout(0.3)(y)

    merged = layers.concatenate([cnn_output, ann_output])
    z = layers.Dense(64, activation='relu')(merged)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(NUM_CLASSES, activation='softmax')(z)

    model = models.Model(inputs=[image_input, feature_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model