#pip install streamlit pyppdf2 langchain langcahain-community python-dotenv faiss-cpu openai huggingface_hub InstructorEmbedding sentence_transformers
import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def get_pdf_text(pdf_docs):
   text =""
   for pdf in pdf_docs:
       pdf_reader= PdfReader(pdf, strict= False)
       for page in pdf_reader.pages:
           text += page.extract_text() 
   return text 

def get_text_chunks(text):
    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap= 200)
    chunks = rec_char_splitter.split_text(text)
    return chunks
    
def get_vectorstore(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_texts(texts =text_chunks,embedding = embeddings)
    
    print("google embedding completed.")
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

#app config
load_dotenv()
st.set_page_config(page_title="SEO Tanvir Bd RAG App", page_icon= ":car:")
st.header("Chat With Multyyiple PDFs :books: ")

# session state
if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]


user_question = st.text_input("Ask any question about your documents: ")
if user_question:
                # handle_userinput(user_question)
                
                response = get_response(user_question)
                st.session_state.chat_history.append(HumanMessage(content=user_question))
                st.session_state.chat_history.append(AIMessage(content=response))
                
                 # conversation
                for message in st.session_state.chat_history:
                    if isinstance(message, AIMessage):
                        with st.chat_message("AI"):
                            st.write(message.content)
                    elif isinstance(message, HumanMessage):
                        with st.chat_message("Human"):
                            st.write(message.content)
        
with st.sidebar:
    st.subheader("Youe Documents")
    all_pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Proceed'", accept_multiple_files= True)
    if st.button("Proceed"):
        with st.spinner("Processing.."):
            #get the pdf text
            raw_text = get_pdf_text(all_pdf_docs)
            # st.write(raw_text)
           
            #texts into chunks
            text_chunks = get_text_chunks(raw_text)
            # st.write(text_chunks)
            
            #create vector store to store the chunks
            vectorstore = get_vectorstore(text_chunks)
            st.write("vectorization completed. Now ask your question")
            
            if "vector_store" not in st.session_state:
                st.session_state.vector_store = get_vectorstore(text_chunks)  
           
