import streamlit as st
import chromadb
from openai import OpenAI

# Use Streamlit secrets for the OpenAI key
api_key = st.secrets["OPENAI_API_KEY"]
client_openai = OpenAI(api_key=api_key)

def get_completion(prompt):
    response = client_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful assistant who answers questions using supplied documents and cites sources. If no documents are relevant, you may use general knowledge."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

# Set up Chroma
client_chroma = chromadb.PersistentClient("./mycollection")
collection = client_chroma.get_or_create_collection(
    name="RAG_Assistant",
    metadata={"hnsw:space": "cosine"}
)

st.title("📚 RAG Assistant")
st.markdown("Ask a question and the assistant will search documents and answer with citations.")

n_results = st.sidebar.number_input("Number of results", min_value=1, max_value=10, value=1)
user_question = st.text_area("Your question")

# Add example documents if empty
if collection.count() == 0:
    collection.add(
        documents=[
            "Wrestling and running are among the oldest known sports, dating back to ancient civilizations such as Mesopotamia and Egypt.",
            "The Mesoamerican ballgame is another ancient sport played over 3000 years ago.",
            "Ancient Olympics in Greece included running, jumping, discus throw, and wrestling."
        ],
        metadatas=[
            {"source": "Ancient Sports Encyclopedia", "year": "2020", "url": "https://ancientsports.example.com"},
            {"source": "Mesoamerican Sports History", "year": "2018", "url": "https://mesosports.example.com"},
            {"source": "Greek Olympics Archive", "year": "2019", "url": "https://greekolympics.example.com"}
        ],
        ids=["doc1", "doc2", "doc3"]
    )

if st.button("Get Answer") and user_question.strip():
    st.write(f"**Question:** {user_question}")
    st.write(f"**Number of Results:** {n_results}")

    results = collection.query(
        query_texts=[user_question],
        n_results=n_results,
        include=["documents", "metadatas"]
    )

    search_results = []
    for doc_list, meta_list in zip(results["documents"], results["metadatas"]):
        for doc, meta in zip(doc_list, meta_list):
            meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
            search_results.append(f"{doc}\nMetadata: {meta_str}")
    search_text = "\n\n".join(search_results) if search_results else "No relevant documents found."

    prompt = f"""
    Answer the following question using the supplied search results.
    At the end of each result is Metadata (source, year, URL). Cite passages and metadata in your answer.

    User Question: {user_question}

    Search Results:
    {search_text}
    """

    response = get_completion(prompt)
    st.markdown("**Answer:**")
    st.write(response)
