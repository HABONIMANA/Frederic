import streamlit as st
import sys
import os
import tempfile
import time
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.document_processor import extract_pages_from_pdf, split_text_into_chunks
from services.vector_store import VectorStore
from services.llm_service import generate_response

load_dotenv()

st.set_page_config(
    page_title="Assistant Web Éducatif",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def get_vector_store():
    return VectorStore()

try:
    vector_store = get_vector_store()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation de la base vectorielle : {e}")
    st.stop()

st.title("Assistant Web Éducatif")

with st.sidebar:
    st.header("Configuration")
    
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if not groq_key and not openai_key and not google_key:
        st.error("Aucune clé API détectée !")
        st.info("Configurez le fichier .env")
    else:
        st.success("API Key active")
        
    available_providers = []
    if groq_key: available_providers.append("groq")
    if openai_key: available_providers.append("openai")
    if google_key: available_providers.append("google")
    
    if available_providers:
        selected_provider = st.selectbox(
            "Fournisseur IA", 
            options=available_providers,
            index=0
        )
        os.environ["LLM_PROVIDER"] = selected_provider
    
    st.divider()
    if st.button("Vider le cache"):
        st.cache_resource.clear()
        st.rerun()

tab1, tab2 = st.tabs(["Chat", "Documents"])

with tab1:
    st.header("Posez votre question")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Votre question (Français, English, Espagnol...)..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔎 Recherche dans les documents...")
            
            try:
                results = vector_store.find_similar_chunks(prompt, n_results=5)
                
                context_text = ""
                sources_display = []
                
                if results and results.get('documents') and results['documents'][0]:
                    documents = results['documents'][0]
                    metadatas = results['metadatas'][0]
                    
                    context_parts = []
                    for i, doc in enumerate(documents):
                        meta = metadatas[i]
                        filename = meta.get('filename', 'Inconnu')
                        page = meta.get('page', '?')
                        
                        context_parts.append(f"--- SOURCE : {filename} (Page {page}) ---\n{doc}")
                      
                        sources_display.append(f" **{filename}** (p.{page})")
                    
                    context_text = "\n\n".join(context_parts)
                
                if not context_text:
                    response = "Je n'ai pas trouvé d'informations pertinentes dans les documents."
                else:
                    message_placeholder.markdown("Génération de la réponse...")
                    response = generate_response(prompt, context_text)
                
                full_response = response
                
                if sources_display:
                    unique_sources = list(set(sources_display))
                    full_response += "\n\n---\n**Sources :**\n" + "\n".join([f"- {s}" for s in unique_sources])
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                message_placeholder.error(f"Erreur : {str(e)}")

with tab2:
    st.header("Gestion des Documents")
    uploaded_file = st.file_uploader("Uploader un PDF", type="pdf")
    
    if uploaded_file:
        if st.button("Traiter le document"):
            with st.spinner("Traitement en cours..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    st.info("Extraction du texte...")
                    pages = extract_pages_from_pdf(tmp_file_path, filename=uploaded_file.name)
                    
                    st.info(f"Découpage en chunks ({len(pages)} pages)...")
                    chunks = split_text_into_chunks(pages)
                    
                    st.info(f"Indexation de {len(chunks)} chunks...")
                    doc_id = int(time.time())
                    vector_store.add_document_chunks(doc_id, chunks)
                    
                    st.success(f"Document **{uploaded_file.name}** indexé avec succès !")
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"Erreur : {e}")