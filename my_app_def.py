#pip install streamlit pyppdf2 langchain langcahain-community python-dotenv 
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
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=.2)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
       MessagesPlaceholder("chat_history"), #MessagesPlaceholder(variable_name="chat_history")
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)
  

def get_conversational_rag_chain(history_aware_retriever): 
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=.2)
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder("chat_history"),
      ("user", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
def get_response(user_input):
    history_aware_retriever = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(history_aware_retriever)
    
    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

#app config
load_dotenv()
st.set_page_config(page_title="SEO Tanvir Bd RAG App", page_icon= ":car:")
st.header("Chat With Multiple PDFs :books: ")

with st.sidebar:
    st.subheader("Your Documents")
    #pdf uploader
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
            x = st.write("Vectorization completed. Now you can chat..")
           
            
            if "vector_store" not in st.session_state:
                st.session_state.vector_store = get_vectorstore(text_chunks)
                    
# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I helyou?"),]   
# user input
user_question = st.chat_input("Ask any question about your documents: ")
if user_question:
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

           
