# 1. 필요한 라이브러리 설치 및 불러오기
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models

# 2. 데이터셋 불러오기 및 증강
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# 데이터 증강을 위한 ImageDataGenerator 설정
datagen = ImageDataGenerator(
    rotation_range=30,        # 최대 30도 범위에서 랜덤 회전
    width_shift_range=0.2,    # 가로로 최대 20% 이동
    height_shift_range=0.2,   # 세로로 최대 20% 이동
    shear_range=0.2,          # 전단 변환 (이미지를 기울임)
    zoom_range=0.2,           # 20% 확대/축소
    horizontal_flip=True,     # 좌우 반전
    fill_mode='nearest'       # 이동 또는 회전 후 빈 공간을 근처 값으로 채움
)

# 데이터셋 준비
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 3. 모델 생성 및 학습
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 증강된 데이터를 사용하여 모델 학습
train_generator = datagen.flow(train_images, train_labels, batch_size=64)
model.fit(train_generator, epochs=10, validation_data=(test_images, test_labels))

# 4. 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')