import streamlit as st
from st_audiorec import st_audiorec
import tempfile
import whisper

# Load Whisper model once
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def record_and_transcribe():
    st.subheader("🎙️ Speak your query")
    audio = st_audiorec()

    if audio is None:
        # Optional: show instructions instead of debugging
        st.info("Click the mic icon above to start recording.")
        return None

    st.success("Voice loaded")

    model = load_whisper_model()
    st.success("Model Loaded")

    with tempfile.NamedTemporaryFile(delete=False, dir="/tmp", suffix=".wav") as tmp_file:
        tmp_file.write(audio)
        tmp_file.flush()
        tmp_path = tmp_file.name
        st.code(tmp_path)

    result = model.transcribe(tmp_path)
    st.success("✅ Transcription complete!")
    st.markdown(f"**🗣️ You said:** `{result['text']}`")
    return result['text']