from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os
from tempfile import TemporaryDirectory
from openpyxl import load_workbook
from langchain_core.documents import Document
import re
import unicodedata

from ..config import Settings
from .google_drive import DriveService 
from .openwebui import OpenWebUIEmbeddings


class VectorDB: 
    def __init__(self, settings: Settings): 
        self.settings = settings
        self.embedder = OpenWebUIEmbeddings(settings) if settings.LLM_PROVIDER == 'openwebui' else GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.GOOGLE_API_KEY
        )

    def query(self, message : str):

        if not os.path.exists(self.settings.FAISS_INDEX_PATH):
            raise ValueError("O índice FAISS não foi encontrado. Certifique-se de que o índice foi criado e salvo corretamente.")
        
        vectorstore = FAISS.load_local(self.settings.FAISS_INDEX_PATH, self.embedder, allow_dangerous_deserialization=True)

        # Fichas de alunos usam formularios parecidos: um nome de arquivo explicito
        # deve restringir a busca antes de comparar os trechos por similaridade.
        def normalize(text):
            text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
            return ' '.join(re.findall(r'[a-z0-9]+', text))

        query = f' {normalize(message)} '
        matches = {}
        # ponytail: varredura linear de titulos; criar catalogo se o indice crescer.
        for doc_id in vectorstore.index_to_docstore_id.values():
            doc = vectorstore.docstore.search(doc_id)
            source = doc.metadata.get('source', '')
            title = normalize(Path(source).stem.split('--')[0])
            if len(title) >= 5 and f' {title} ' in query:
                matches[source] = len(title)
        if matches:
            longest = max(matches.values())
            sources = {source for source, length in matches.items() if length == longest}
            return vectorstore.similarity_search(
                message, k=4, filter=lambda metadata: metadata.get('source') in sources,
                fetch_k=vectorstore.index.ntotal,
            )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        docs = retriever.invoke(message)

        return docs
        
    def create_faiss_index(self, parent_folder : str):

        if not os.path.exists(parent_folder):
            raise ValueError(f"O diretório {parent_folder} não existe. Verifique o caminho e tente novamente.")
        
        folder_path = Path(parent_folder)

        docs = []
        files = [folder_path] if folder_path.is_file() else folder_path.iterdir()
        for file_path in files:
            if not file_path.is_file() or file_path.suffix.lower() not in ('.pdf', '.xlsx'):
                continue
            try:
                if file_path.suffix.lower() == '.xlsx':
                    workbook = load_workbook(file_path, read_only=True, data_only=True)
                    try:
                        for sheet in workbook:
                            rows = []
                            for row in sheet.iter_rows():
                                cells = [f'{cell.coordinate}: {cell.value}' for cell in row
                                         if cell.value is not None and str(cell.value).strip()]
                                if cells:
                                    rows.append(' | '.join(cells))
                            if rows:
                                docs.append(Document(page_content='\n'.join(rows), metadata={
                                    'source': str(file_path), 'sheet': sheet.title,
                                }))
                    finally:
                        workbook.close()
                else:
                    loader = PyMuPDFLoader(str(file_path))
                    docs.extend(loader.load())
                print(f"Carregado com sucesso o arquivo {file_path.name}")

            except Exception as e:
                raise ValueError(f'Erro ao carregar {file_path.name}. O indice nao foi alterado.') from e

        print(f"Total de documentos carregados: {len(docs)}")
        
        if not docs:
            raise ValueError('Nenhum documento com texto foi encontrado. O indice nao foi alterado.')
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        split_docs = splitter.split_documents(docs)
        for doc in split_docs:
            heading = f"Arquivo: {Path(doc.metadata['source']).name}\n"
            if doc.metadata.get('sheet'):
                heading += f"Aba: {doc.metadata['sheet']}\n"
            doc.page_content = heading + doc.page_content

        if not split_docs:
            raise ValueError('Nenhum documento com texto foi encontrado. O indice nao foi alterado.')

        vectorstore = FAISS.from_documents(split_docs, self.embedder)

        vectorstore.save_local(self.settings.FAISS_INDEX_PATH)


    def refresh_knowledge_base(self):
        """
        Método Mestre: Baixa do Drive e recria o índice.
        """
        # 1. Configurar caminhos
        # Vamos assumir que data/ fica na raiz do backend
        if not self.settings.GOOGLE_DRIVE_FOLDER_ID:
            self.create_faiss_index(self.settings.local_data_path)
            return {"status": "success", "message": "PDFs locais indexados com sucesso!"}

        # Cada sincronizacao usa somente os arquivos atuais; PDFs locais ficam intactos.
        with TemporaryDirectory(prefix='itt-drive-') as download_path:
            print("Iniciando download do Google Drive...")
            drive_service = DriveService(
                credentials_path=self.settings.GOOGLE_CREDENTIALS_PATH,
                folder_id=self.settings.GOOGLE_DRIVE_FOLDER_ID,
                download_path=download_path,
                credentials_json=self.settings.GOOGLE_CREDENTIALS_JSON.get_secret_value(),
            )
            drive_service.download_files()
            print("Recriando indice vetorial...")
            self.create_faiss_index(download_path)
        
        return {"status": "success", "message": "Base de conhecimento atualizada com sucesso!"}
