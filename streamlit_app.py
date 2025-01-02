import streamlit as st


from PyPDF2 import PdfReader
#from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval




def main():
    #load_dotenv() lokal
    
    st.set_page_config(page_title="UTM BOT")
    st.title("UTMs BOT")

    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    #main
    #conversationhistory
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

    user_query = st.chat_input("Masukkan Pertanyaan anda disini")

    if user_query is not None and user_query!=" ":
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
        
        with st.chat_message("AI"):
            ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

        st.session_state.chat_history.append(AIMessage(ai_response))



    #sidebar
    with st.sidebar:
        st.subheader("Upload PDF")
        pdf_docs = st.file_uploader(
            "Upload Data PDFs dan tekan tombol 'Proses'", accept_multiple_files=True)
        if st.button("Proses"):
            with st.spinner("M"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = text_splitting(raw_text)

                # create vector store
                vectorstore = text_embeddings(text_chunks)

                


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def text_splitting(text):
    text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks

def text_embeddings(text_chunks):
    #streamlit deployment secrets
    openai_api_key = st.secrets["general"]["openai_api_key"]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=  openai_api_key)

    indexx = "utmvector"
    vector_store = PineconeVectorStore.from_texts(texts=text_chunks, embedding=embeddings, index_name=indexx)
    return vector_store








#get response
def get_response(query, chat_history):
    #streamlit deployment secrets
    openai_api_key = st.secrets["general"]["openai_api_key"]
    pinecone_api_key = st.secrets["general"]["pinecone_api_key"]
    template="""Answer the question as truthfully as possible using the provided context. the question is : {user_question}. Chat history: {chat_history}, answer in bahasa indonesia,All the question should be related to UTM (Universitas Trunojoyo Madura) otherwise say 'Pertanyaan di luar cakupan UTM', 
and if the answer is not contained within the text below and the context, say 'Informasi tidak ditemukan, silahkan Hubungi CS UTM : 089678838234', full context {result} """

    prompt = ChatPromptTemplate.from_template(template)
    model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=  openai_api_key)
    embed_query = model.embed_query(query)
    process_query = PineconeVectorStore(index_name="utmvector", embedding=model, pinecone_api_key=pinecone_api_key)
    result = process_query.similarity_search(query, k=2)
    results = "utm adalah kampus idaman"
    print(result)


    llm = ChatOpenAI(model="gpt-4o-mini", api_key=  openai_api_key)

    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "chat_history": chat_history,
        "result": result,
        "user_question": query
        
    })

if __name__ == '__main__':
    main()