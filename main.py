import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from diffusers import StableDiffusionPipeline
import torch
from training import Generator, latent_dim, num_classes, img_size, channels
import torch.nn as nn

# App title
st.title("Stable Diffusion, CIFAR-10 & MNIST Classifier, Digit Generator")

# Sidebar options
option = st.sidebar.selectbox("Choose an option", ["Generate Image", "Classify CIFAR-10 Image", "Generate Digits", "Classify MNIST Image"])

# Load pre-trained stable diffusion model
@st.cache_data()
def load_stable_diffusion_model():
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    return model

# Load the pre-trained CNN model for CIFAR-10
@st.cache_data()
def load_cifar10_model():
    model = load_model("cifar10_cnn_model.h5")
    return model

# Load the pre-trained CNN model for MNIST
@st.cache_data()
def load_mnist_model():
    model = load_model("mnist_cnn_model.h5")
    return model

# Load the PyTorch model for digit generation
@st.cache_data()
def load_digit_generator_model():
    generator = Generator(latent_dim, num_classes, img_size, channels)
    generator.load_state_dict(torch.load("generator.pth"))
    generator.eval()
    return generator


# Stable Diffusion Image Generation
if option == "Generate Image":
    prompt = st.text_input("Enter prompt for image generation", "A futuristic city")
    if st.button("Generate"):
        model = load_stable_diffusion_model()
        generated_image = model(prompt).images[0]
        st.image(generated_image, caption="Generated Image", use_column_width=True)


# CIFAR-10 Image Classification
if option == "Classify CIFAR-10 Image":
    uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for the CNN model

        # Load the pre-trained CNN model
        model = load_cifar10_model()

        # Predict class probabilities
        predictions = model.predict(img_array)[0]
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Display image and predictions
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Plot confidence for each class
        fig, ax = plt.subplots()
        ax.barh(class_names, predictions)
        ax.set_xlabel("Confidence")
        ax.set_title("Classification Results")
        st.pyplot(fig)

# Digit Generation using PyTorch model
if option == "Generate Digits":
    # Dropdown menu to select a digit from 0-9
    digit = st.selectbox("Select a digit (0-9)", list(range(10)))

    if st.button("Generate Digit Image"):
        # Load the PyTorch model
        model = load_digit_generator_model()
        with torch.inference_mode():
            noise = torch.randn(1, latent_dim)
            label = torch.tensor([digit])
            generated_img: torch.Tensor = model(noise, label.cpu())

            img = generated_img.squeeze().numpy()
            img = (img * 255).astype(np.uint8)

        # Convert to image and display

        # Display generated image
        st.image(img, caption=f"Generated Image of Digit {digit}", use_column_width=True)

# MNIST Image Classification
if option == "Classify MNIST Image":
    uploaded_file = st.file_uploader("Choose an MNIST-like image to classify", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and preprocess the uploaded image
        img = Image.open(uploaded_file).convert('L').resize((28, 28))  # Convert to grayscale and resize to 28x28
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add the channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add the batch dimension

        # Load the pre-trained MNIST model
        model = load_mnist_model()

        # Predict the class
        predictions = model.predict(img_array)[0]
        predicted_digit = np.argmax(predictions)

        # Display the uploaded image and prediction
        st.image(img, caption="Uploaded MNIST Image", use_column_width=True)
        st.write(f"Predicted Digit: {predicted_digit}")

        # Plot confidence for each digit class
        fig, ax = plt.subplots()
        ax.barh(range(10), predictions)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Digit")
        ax.set_title("MNIST Classification Results")
        st.pyplot(fig)
