import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama
import tempfile
import os
import matplotlib.pyplot as plt

# Streamlit arayüzü başlığı
st.title("📄 RAG Sistemi ile PDF Soru-Cevap")
st.markdown("""
Bu uygulama, yüklediğiniz bir PDF dosyasını işler ve sorularınıza yanıt verir.
""")

# Sidebar için ince ayarlar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    with st.expander("Metin Parçalama Ayarları"):
        chunk_size = st.slider("Metin Parça Boyutu (Chunk Size)", 500, 2000, 1000)
        chunk_overlap = st.slider("Metin Parça Örtüşmesi (Chunk Overlap)", 0, 500, 200)
    with st.expander("Model Ayarları"):
        model_name = st.selectbox("Model", ["deepseek-r1:1.5b", "llama2", "mistral"])
        temperature = st.slider("Model Sıcaklığı (Temperature)", 0.1, 1.0, 0.7)

# PDF yükleme ve işleme fonksiyonu
def load_and_process_pdf(uploaded_file, chunk_size, chunk_overlap):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts, tmp_file_path
    except Exception as e:
        st.error(f"PDF yüklenirken bir hata oluştu: {e}")
        return None, None

# Vektör veritabanı oluşturma fonksiyonu
def create_vector_store(texts, embeddings):
    try:
        vectorstore = Chroma.from_documents(texts, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulurken bir hata oluştu: {e}")
        return None

# Soruya yanıt oluşturma fonksiyonu
def generate_response(query, context, model, temperature):
    try:
        response = ollama.generate(
            model=model,
            prompt=f"Soru: {query}\n\nBağlam: {context}\n\nYanıt:",
            options={"temperature": temperature}
        )
        return response['response']
    except Exception as e:
        st.error(f"Yanıt oluşturulurken bir hata oluştu: {e}")
        return None

# Görselleştirme fonksiyonu
def visualize_text_chunks(texts):
    chunk_sizes = [len(text.page_content) for text in texts]
    fig, ax = plt.subplots()
    ax.hist(chunk_sizes, bins=20, color='blue', alpha=0.7)
    ax.set_title("Metin Parçalarının Boyut Dağılımı")
    ax.set_xlabel("Parça Boyutu")
    ax.set_ylabel("Frekans")
    st.pyplot(fig)

# Kullanıcıdan PDF dosyası yüklemesi istenir
uploaded_file = st.file_uploader("📤 Lütfen bir PDF dosyası yükleyin", type="pdf")

if uploaded_file is not None:
    # PDF yükleme ve işleme
    with st.spinner("PDF yükleniyor ve işleniyor..."):
        texts, tmp_file_path = load_and_process_pdf(uploaded_file, chunk_size, chunk_overlap)

    if texts:
        st.success("✅ PDF başarıyla yüklendi ve işlendi!")
        st.write(f"Toplam {len(texts)} metin parçası oluşturuldu.")

        # Metin parçalarını görselleştirme (isteğe bağlı)
        with st.expander("Metin Parçalarını Görselleştir"):
            visualize_text_chunks(texts)

        # OllamaEmbeddings kullanılarak metinler vektörleştirilir
        embeddings = OllamaEmbeddings(model=model_name)

        # Vektör veritabanı oluşturulur
        with st.spinner("Vektör veritabanı oluşturuluyor..."):
            vectorstore = create_vector_store(texts, embeddings)

        if vectorstore:
            st.success("✅ Vektör veritabanı başarıyla oluşturuldu!")

            # Kullanıcıdan soru alınır
            query = st.text_input("❓ Lütfen bir soru girin:")

            if query:
                with st.spinner("Soruya en uygun metin parçaları aranıyor..."):
                    # Soruya en uygun metin parçaları bulunur
                    docs = vectorstore.similarity_search_with_score(query, k=3)
                    context = " ".join([doc[0].page_content for doc in docs])

                    # Bağlam metnini göster (isteğe bağlı)
                    with st.expander("Bağlam Metnini Göster"):
                        st.write(context)

                    # Yanıt oluşturulur ve gösterilir
                    with st.spinner("Yanıt oluşturuluyor..."):
                        response = generate_response(query, context, model_name, temperature)
                        if response:
                            st.subheader("📝 Yanıt:")
                            st.write(response)

    # Geçici dosya silinir
    if tmp_file_path:
        os.remove(tmp_file_path)