import io
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage import transform, color


def load_image():
    """Создание формы для загрузки изображения"""
    uploaded_file = st.file_uploader(label='Загрузите снимок КТ')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None


# кэшируем загруженную модель
@st.cache_resource
def load_model():
    """Загружаем модель"""
    model = tf.keras.models.load_model("models/vgg16")
    return model


def preprocess_image(img):
    """Преобразует изображение PIL.Image в формат для распознавания ИНС"""
    img = img.resize((200, 200))
    x = np.array(img).astype('float32') / 255
    x = transform.resize(x, (200, 200, 3))
    x = color.rgb2gray(x)
    return np.expand_dims(x, axis=0)


def print_predictions(preds):
    """Выводит предсказанные данные"""
    if preds[0][0] < preds[0][1]:
        st.error("Наблюдается опухоль с вероятностью {:.3%}".format(preds[0][1]))
    else:
        st.success("Не наблюдается опухоль с вероятностью {:.3%}".format(preds[0][0]))


model = load_model()
# Выводим заголовок страницы
st.title('Классификация снимков КТ с раковой опухолью')
# Выводим форму загрузки изображения и получаем изображение
img = load_image()
# Показывам кнопку для запуска распознавания изображения
result = st.button('Распознать изображение')
# Если кнопка нажата, то запускаем распознавание изображения
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('### Результаты распознавания:')
    print_predictions(preds)
