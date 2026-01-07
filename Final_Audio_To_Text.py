# import streamlit as st
# from st_audiorec import st_audiorec
# import whisper
# from gtts import gTTS
# from transformers import MarianMTModel, MarianTokenizer
# import os
#
# # ----------------------------
# # PAGE CONFIG
# # ----------------------------
# st.set_page_config(page_title="🎙️ Audio Language Converter", layout="centered")
# st.title("🎙️ Audio Language Converter")
# st.info("Record from mic OR upload audio → Convert to English / Hindi audio")
#
# # ----------------------------
# # SAVE DIRECTORY
# # ----------------------------
# SAVE_DIR = r"C:\Users\Admin\Desktop\A_Maulik\Audio_to_Text"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# # ----------------------------
# # SESSION STATE
# # ---------------------------
# if "audio_path" not in st.session_state:
#     st.session_state.audio_path = None
#
# # ----------------------------
# # LOAD MODELS (CACHED)
# # ----------------------------
# @st.cache_resource
# def load_models():
#     whisper_model = whisper.load_model("medium")
#
#     hi_model_name = "Helsinki-NLP/opus-mt-en-hi"
#     hi_tokenizer = MarianTokenizer.from_pretrained(hi_model_name)
#     hi_model = MarianMTModel.from_pretrained(hi_model_name)
#
#     return whisper_model, hi_tokenizer, hi_model
#
# whisper_model, hi_tokenizer, hi_model = load_models()
#
# # ----------------------------
# # AUDIO INPUT MODE
# # ----------------------------
# input_mode = st.radio(
#     "Choose audio input method",
#     ["🎙️ Record from Mic", "📁 Upload Audio File"]
# )
#
# # ----------------------------
# # MIC RECORDING (✅ FIXED)
# # ----------------------------
# if input_mode == "🎙️ Record from Mic":
#     st.subheader("🎙️ Speak Now")
#
#     audio_bytes = st_audiorec()
#
#     if audio_bytes is not None:
#         st.success("✅ Audio recorded")
#
#         if st.button("💾 Save & Use Recording"):
#             audio_path = os.path.join(SAVE_DIR, "mic_audio.wav")
#
#             # ✅ Correct: write bytes directly
#             with open(audio_path, "wb") as f:
#                 f.write(audio_bytes)
#
#             st.session_state.audio_path = audio_path
#             st.audio(audio_path)
#             st.success(f"Recording saved at:\n{audio_path}")
#
# # ----------------------------
# # FILE UPLOAD
# # ----------------------------
# if input_mode == "📁 Upload Audio File":
#     audio_file = st.file_uploader(
#         "Upload audio",
#         type=["wav", "mp3", "m4a", "aac", "ogg"]
#     )
#
#     if audio_file:
#         audio_path = os.path.join(SAVE_DIR, audio_file.name)
#
#         with open(audio_path, "wb") as f:
#             f.write(audio_file.read())
#
#         st.session_state.audio_path = audio_path
#         st.audio(audio_path)
#         st.success(f"Audio saved at:\n{audio_path}")
#
# # ----------------------------
# # OUTPUT LANGUAGE SELECTION
# # ----------------------------
# output_lang = st.multiselect(
#     "Select output audio language(s)",
#     ["English", "Hindi"],
#     default=["English"]
# )
#
# # ----------------------------
# # CONVERT AUDIO
# # ----------------------------
# if st.session_state.audio_path and output_lang and st.button("🔁 Convert Audio"):
#     with st.spinner("Processing audio..."):
#
#         audio_path = st.session_state.audio_path
#
#         # Detect language
#         detected_lang = whisper_model.transcribe(
#             audio_path, task="transcribe",temperature=0.0
#         )["language"]
#
#         # Convert to English text
#         english_text = whisper_model.transcribe(
#             audio_path, task="translate",temperature=0.0
#         )["text"]
#
#         st.subheader("📝 English Text")
#         st.text_area("Output", english_text, height=150)
#
#         output_files = []
#
#         # English Audio
#         if "English" in output_lang:
#             eng_file = os.path.join(SAVE_DIR, "output_english.mp3")
#             gTTS(english_text, lang="en").save(eng_file)
#             output_files.append(("English Audio", eng_file))
#
#         # Hindi Audio
#         if "Hindi" in output_lang:
#             inputs = hi_tokenizer(english_text, return_tensors="pt", padding=True)
#             translated = hi_model.generate(**inputs)
#             hindi_text = hi_tokenizer.decode(
#                 translated[0], skip_special_tokens=True
#             )
#
#             hi_file = os.path.join(SAVE_DIR, "output_hindi.mp3")
#             gTTS(hindi_text, lang="hi").save(hi_file)
#             output_files.append(("Hindi Audio", hi_file))
#
#     # ----------------------------
#     # OUTPUT SECTION
#     # ----------------------------
#     st.success("✅ Conversion completed")
#
#     for label, file in output_files:
#         st.subheader(label)
#         st.audio(file)
#
#         with open(file, "rb") as f:
#             st.download_button(
#                 f"⬇ Download {label}",
#                 data=f,
#                 file_name=os.path.basename(file),
#                 mime="audio/mpeg"
#             )
#
#     st.caption(f"Detected audio language: `{detected_lang}`")





import streamlit as st
from st_audiorec import st_audiorec
import whisper
from gtts import gTTS
from transformers import MarianMTModel, MarianTokenizer
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="🎙️ Audio Language Converter", layout="centered")
st.title("🎙️ Audio Language Converter")
st.info("Record from mic OR upload audio → Convert to English / Hindi audio")

# ----------------------------
# SAVE DIRECTORY
# ----------------------------
SAVE_DIR = r"C:\Users\Admin\Desktop\A_Maulik\Audio_to_Text"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# SESSION STATE
# ----------------------------
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None


# ----------------------------
# LOAD MODELS (CACHED)
# ----------------------------
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("medium")

    hi_model_name = "Helsinki-NLP/opus-mt-en-hi"
    hi_tokenizer = MarianTokenizer.from_pretrained(hi_model_name)
    hi_model = MarianMTModel.from_pretrained(hi_model_name)

    return whisper_model, hi_tokenizer, hi_model


whisper_model, hi_tokenizer, hi_model = load_models()

# ----------------------------
# AUDIO INPUT MODE
# ----------------------------
input_mode = st.radio(
    "Choose audio input method",
    ["🎙️ Record from Mic", "📁 Upload Audio File"]
)

# ----------------------------
# MIC RECORDING (✅ FIXED)
# ----------------------------
if input_mode == "🎙️ Record from Mic":
    st.subheader("🎙️ Speak Now")

    audio_bytes = st_audiorec()

    if audio_bytes is not None:
        st.success("✅ Audio recorded")

        if st.button("💾 Save & Use Recording"):
            audio_path = os.path.join(SAVE_DIR, "mic_audio.wav")

            # ✅ Correct: write bytes directly
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            st.session_state.audio_path = audio_path
            st.audio(audio_path)
            st.success(f"Recording saved at:\n{audio_path}")

# ----------------------------
# FILE UPLOAD
# ----------------------------
if input_mode == "📁 Upload Audio File":
    audio_file = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a", "aac", "ogg"]
    )

    if audio_file:
        audio_path = os.path.join(SAVE_DIR, audio_file.name)

        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        st.session_state.audio_path = audio_path
        st.audio(audio_path)
        st.success(f"Audio saved at:\n{audio_path}")

# ----------------------------
# OUTPUT LANGUAGE SELECTION
# ----------------------------
output_lang = st.multiselect(
    "Select output audio language(s)",
    ["English", "Hindi"],
    default=["English"]
)

# ----------------------------
# CONVERT AUDIO
# ----------------------------
if st.session_state.audio_path and output_lang and st.button("🔁 Convert Audio"):
    with st.spinner("Processing audio..."):

        audio_path = st.session_state.audio_path

        # Detect language
        detected_lang = whisper_model.transcribe(
            audio_path, task="transcribe",temperature=0.0
        )["language"]

        # Convert to English text
        english_text = whisper_model.transcribe(
            audio_path, task="translate",temperature=0.0
        )["text"]

        st.subheader("📝 English Text")
        st.text_area("Output", english_text, height=150)

        output_files = []

        # English Audio
        if "English" in output_lang:
            eng_file = os.path.join(SAVE_DIR, "output_english.mp3")
            gTTS(english_text, lang="en").save(eng_file)
            output_files.append(("English Audio", eng_file))

        # Hindi Audio
        if "Hindi" in output_lang:
            inputs = hi_tokenizer(english_text, return_tensors="pt", padding=True)
            translated = hi_model.generate(**inputs)
            hindi_text = hi_tokenizer.decode(
                translated[0], skip_special_tokens=True
            )

            hi_file = os.path.join(SAVE_DIR, "output_hindi.mp3")
            gTTS(hindi_text, lang="hi").save(hi_file)
            output_files.append(("Hindi Audio", hi_file))

    # ----------------------------
    # OUTPUT SECTION
    # ----------------------------
    st.success("✅ Conversion completed")

    for label, file in output_files:
        st.subheader(label)
        st.audio(file)

        with open(file, "rb") as f:
            st.download_button(
                f"⬇ Download {label}",
                data=f,
                file_name=os.path.basename(file),
                mime="audio/mpeg"
            )

    st.caption(f"Detected audio language: `{detected_lang}`")
