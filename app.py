import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Display Python version with additional details
st.sidebar.write("Versión de Python:", platform.python_version())

# Load your pre-trained model
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Set a more engaging title with emoji
st.title("📷 Reconocimiento de Imágenes con IA 🤖")

# Display an image header to add visual appeal
header_image = Image.open('OIG5.jpg')
st.image(header_image, caption='Modelo de Reconocimiento', use_column_width=True)

# Add a sidebar for instructions
with st.sidebar:
    st.subheader("📋 Instrucciones")
    st.write("1. Presiona 'Toma una Foto' para capturar una imagen.")
    st.write("2. El modelo identificará la dirección con base en la imagen capturada.")
    st.write("3. Verifica el resultado en la pantalla principal.")
    st.write("4. Disfruta explorando los resultados con este modelo entrenado en Teachable Machine!")

st.markdown("---")  # Add a horizontal line for separation

# Introduce columns for a better layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("Captura una Imagen 📸")
    img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    with col2:
        st.subheader("Imagen Capturada")
        st.image(img_file_buffer, caption="Tu Imagen", use_column_width=True)

    # Prepare the image for prediction
    img = Image.open(img_file_buffer)
    img = img.resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Display loading animation while predicting
    with st.spinner('Analizando la imagen...'):
        prediction = model.predict(data)
    
    # Display results with a progress bar for better visualization
    st.markdown("---")
    st.subheader("Resultados del Reconocimiento")
    if prediction[0][0] > 0.5:
        st.success(f"⬅️ Izquierda, con Probabilidad: {prediction[0][0]:.2f}")
        st.progress(int(prediction[0][0] * 100))
        
    if prediction[0][1] > 0.5:
        st.success(f"⬆️ Arriba, con Probabilidad: {prediction[0][1]:.2f}")
        st.progress(int(prediction[0][1] * 100))
    
    # In case other predictions are needed, they can be easily added below
    # if prediction[0][2] > 0.5:
    #     st.success(f"➡️ Derecha, con Probabilidad: {prediction[0][2]:.2f}")
    #     st.progress(int(prediction[0][2] * 100))

# Add a footer for more engagement
st.markdown("---")
st.markdown("Desarrollado con ❤️ por Tomas")



