import streamlit as st
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            print(message.content)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            print(message.content)
    return message.content

def extract_patient_info(text_chunks, conversation_chain):
    # Use the conversation chain to extract patient information
    patient_info = {
        'diagnosis_date': None,
        'diagnosis_time': None,
        'disease': None,
        'height': None,
        'weight': None,
        'medicines': []
    }

    for chunk in text_chunks:
        response = conversation_chain({'question': chunk})
        chat_history = response['chat_history']

        # Extract relevant information from the model's responses
        for message in chat_history:
            # Update the conditions based on the actual responses from your model
            if "diagnosis date" in message.content.lower():
                patient_info['diagnosis_date'] = message.content
            elif "diagnosis time" in message.content.lower():
                patient_info['diagnosis_time'] = message.content
            elif "disease" in message.content.lower():
                patient_info['disease'] = message.content
            elif "height" in message.content.lower():
                patient_info['height'] = message.content
            elif "weight" in message.content.lower():
                patient_info['weight'] = message.content
            elif "medicines" in message.content.lower():
                patient_info['medicines'].append(message.content)

    return patient_info

def main():
    load_dotenv()
    st.set_page_config(page_title="ContextualConnect Pro",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None


    
    main_title = '<h1 style="font-family:Sans-serif; color:Black; font-size: 60px;">Med Record</h1>'
    st.markdown(main_title, unsafe_allow_html=True) 
  

    with st.sidebar:
      st.subheader("Your documents")
      pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

# Processing logic moved to the main content area
    if st.button("Process"):
      with st.spinner("Processing"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)

        # Extract patient information using the language model
        # ...

# Extract patient information using the language model
        patient_info = extract_patient_info(text_chunks, st.session_state.conversation)

# Display patient information in an organized way
        st.write("### Extracted Patient Information:")
        if patient_info['diagnosis_date']:
          st.write(f"**Diagnosis Date:** {patient_info['diagnosis_date']}")

        if patient_info['diagnosis_time']:
          st.write(f"**Diagnosis Time:** {patient_info['diagnosis_time']}")

        if patient_info['disease']:
          st.write(f"**Disease:** {patient_info['disease']}")

        if patient_info['height']:
          st.write(f"**Height:** {patient_info['height']}")

        if patient_info['weight']:
          st.write(f"**Weight:** {patient_info['weight']}")

        if patient_info['medicines']:
          st.write("**Medicines:**")
          for medicine in patient_info['medicines']:
            st.write(f"- {medicine}")

if __name__ == '__main__':
    main()
