import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

for k in ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "PDF_PATH"):
    if not os.getenv(k):
        raise RuntimeError(f"Variável de ambiente {k} não definida")

pdf_path = Path(os.getenv("PDF_PATH"))
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

print(f"Carregando PDF: {pdf_path}")
docs = PyPDFLoader(str(pdf_path)).load()

splits = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=False,
).split_documents(docs)

if not splits:
    print("Nenhum conteúdo encontrado no PDF.")
    raise SystemExit(0)

enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
    )
    for d in splits
]

ids = [f"doc-{i}" for i in range(len(enriched))]

embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)

print(f"Salvando {len(enriched)} chunks no banco de dados...")
store.add_documents(documents=enriched, ids=ids)
print("Ingestão concluída com sucesso!")


if __name__ == "__main__":
    pass
