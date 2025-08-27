from dllmforge.rag_preprocess_documents_docling import RagPreprocessDocumentsDocling
from langchain_docling.loader import ExportType


class TestRagPreprocessDocumentsDocling:

    def test_preprocess_documents(self):
        pdfs = [
            #"test_pdfs/11207168-025-GEO-0002_v1.0-Interviews International Experts - ondertekend.pdf",
            "tests/test_input/Campos Montero et al. - 2025 - SchemaGAN A conditional Generative Adversarial Network for geotechnical subsurface schematisation.pdf",
            #"test_pdfs/11207168-030-GEO-0001_v1.0-Weak Layer Mapping - signed.pdf"
        ]

        embedder = "sentence-transformers/all-MiniLM-L6-v2"
        rag_processor = RagPreprocessDocumentsDocling(pdfs, embedder)
        documents = rag_processor.convert_documents()
        processed_docs = rag_processor.get_processed_documents()
        assert len(processed_docs) > 0
