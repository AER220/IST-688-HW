import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import time


#  reading a users PDF file into plain text
def read_pdf(uploaded_file):
    """Extract all text from an uploaded PDF and return it as one string."""
    # PdfReader understands the PDF binary format; the uploaded file
    # from Streamlit is file-like, so it can be passed in directly.
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # we ignore all the images in the pages and focus on the text only
            text += page_text + "\n"
    return text


# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Validate the API key as soon as it's entered.
    try:
        client.models.list()
    except Exception:
        st.error("❌ That API key doesn't look valid. Please check it and try again.")
        st.stop()

    st.success("✅ API key validated. Please upload a document and ask a question.")

    #  let the user choose which model answers
    # Note: the API name for "gpt-3.5" is "gpt-3.5-turbo".
    model_option = st.sidebar.selectbox(
        "Choose a model:",
        ("gpt-3.5-turbo", "gpt-4.1", "gpt-5.6-sol", "gpt-5-nano"),
    )

    #  accepting only .txt and .pdf files
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Ask the user for a question via `st.text_area` same question there.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Is this course hard?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # rewind the file so every rerun re-reads it from the start
        uploaded_file.seek(0)

        #  routing the file to the right reader
        document = None
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")

        # Only call the API if we successfully read the document.
        if document:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the model selected in the sidebar.
            stream = client.chat.completions.create(
                model=model_option,
                messages=messages,
                stream=True,
            )

            # Stream the response and time how long the answer takes.
            start = time.time()
            st.write_stream(stream)
            elapsed = time.time() - start
            st.caption(f"⏱️ {model_option} answered in {elapsed:.1f} seconds")