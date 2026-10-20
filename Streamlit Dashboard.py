import streamlit as st
import os
import time
from generate import generate_music
from train import train_model

st.set_page_config(
    page_title="AI Music Generator Dashboard",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 AI-Powered Music Generation System")
st.markdown("### CodeAlpha Artificial Intelligence Internship - Task 3")
st.write("This application uses a deep learning **LSTM (Long Short-Term Memory)** neural network architecture to study patterns from classical or custom MIDI tracks and compose original instrumental sequences.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Model Training Control")
    st.write("Upload custom `.mid` files to the `data/` folder and trigger training.")
    if st.button("Start Model Training Pipeline"):
        with st.spinner("Training LSTM model on MIDI dataset... Please wait."):
            try:
                train_model()
                st.success("Model training completed successfully!")
            except Exception as e:
                st.error(f"Training encountered an issue: {e}")

with col2:
    st.subheader("2. AI Composition Engine")
    st.write("Generate a brand new audio sequence from trained weights.")
    if st.button("Generate New Composition"):
        with st.spinner("Composing original musical sequence via Deep Learning..."):
            try:
                output_file = generate_music()
                if output_file and os.path.exists(output_file):
                    st.success("Music generation successful!")
                    st.audio(output_file)
                    with open(output_file, "rb") as f:
                        st.download_button(
                            label="Download Generated MIDI File",
                            data=f,
                            file_name="AI_Composition.mid",
                            mime="audio/midi"
                        )
                else:
                    st.warning("Please ensure a trained model checkpoint (`.keras`) exists in the `model/` folder.")
            except Exception as e:
                st.error(f"Generation failed: {e}")

st.markdown("---")
st.markdown("#### Tech Stack Used:")
st.markdown("- **Deep Learning:** TensorFlow / Keras (Multi-layer LSTM Architecture)")
st.markdown("- **Music Processing:** `music21` toolkit")
st.markdown("- **Web UI Framework:** Streamlit")