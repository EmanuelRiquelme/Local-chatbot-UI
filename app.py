import streamlit as st
import time
import os
from utils import start_db,inference_pipeline,create_vector_db

if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False 
if 'upload_pdf' not in st.session_state:
    st.session_state.upload_pdf = True 
if 'db' not in st.session_state:
    st.session_state.db = False 

placeholder = st.empty()

#change dict
#local llm on cpu
def main():
    with placeholder.container():
        if st.session_state.upload_pdf: 
            st.write("### Upload a PDF")
            pdf_file= st.file_uploader("### Upload PDF", type=([".pdf"]) , label_visibility ='hidden' )

            if pdf_file:
                file_saved_dir = os.path.join(f'{pdf_file.name}')
                if not os.path.exists(file_saved_dir):
                    with open(file_saved_dir, "wb") as f:
                        f.write(pdf_file.getvalue())

                DB_file_name = pdf_file.name.split('/')[-1].split('.')[0]
                if not os.path.exists(DB_file_name):
                    with st.spinner("Thinking..."):
                        create_vector_db(pdf_file.name)
                st.session_state.db = start_db(DB_file_name)
                placeholder.empty()
                st.session_state.show_chat = True
                st.session_state.upload_pdf = False

    if st.session_state.show_chat:
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [{"assistant": "Hi,I'm a deep learning expert I will solve your questions"}]

        for message in st.session_state.messages:
            for role,content in message.items():
                with st.chat_message(role):
                    st.write(content)

        if prompt := st.chat_input():
            st.session_state.messages.append({"user": prompt})
            with st.chat_message("user"):
                st.write(prompt)


        # Generate a new response if last message is not from assistant
        for role,content in st.session_state.messages[-1].items():
            if role != 'assistant':
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        #add custom response
                        chat_history = st.session_state.messages
                        db= st.session_state.db
                        response = inference_pipeline(db,prompt,chat_history)
                        placeholder_message = st.empty()
                        full_response = ''
                        for item in response:
                            full_response += item
                            placeholder_message.markdown(full_response)
                        placeholder_message.markdown(full_response)
                message = {"assistant": full_response}
                st.session_state.messages.append(message)
                print(st.session_state.messages)


if __name__ == '__main__':
    main()
