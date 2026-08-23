import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = "mnist_cnn.keras"

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🔢 MNIST Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0–9), and the CNN will predict it.")

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

def preprocess_image(image):
    """Convert an uploaded image into MNIST-like shape: (1, 28, 28, 1)."""
    gray = ImageOps.grayscale(image)
    arr = np.array(gray)

    # Make the digit white on a black background, like MNIST.
    if arr.mean() > 127:
        arr = 255 - arr

    # Remove very light background noise.
    mask = arr > 30

    if mask.any():
        ys, xs = np.where(mask)
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        # Pad to a square while preserving the digit's proportions.
        h, w = arr.shape
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=np.uint8)

        y0 = (side - h) // 2
        x0 = (side - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = arr
        arr = canvas

    processed = Image.fromarray(arr).resize((28, 28))
    arr = np.array(processed).astype("float32") / 255.0

    return arr.reshape(1, 28, 28, 1)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    input_tensor = preprocess_image(image)

    if st.button("Predict Digit", type="primary"):
        probabilities = model.predict(input_tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities) * 100)

        st.success(f"Predicted digit: **{predicted_digit}**")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Prediction Probabilities")
        st.bar_chart(
            {
                "Probability": probabilities
            },
            x_label="Digit",
            y_label="Probability"
        )

        st.caption("Tip: For best results, use a clear single handwritten digit with minimal background.")
else:
    st.info("Upload a PNG, JPG, or JPEG image to start.")
