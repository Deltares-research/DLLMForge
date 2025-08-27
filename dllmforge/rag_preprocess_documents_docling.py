from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langchain_docling.loader import ExportType
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, )
from docling.document_converter import DocumentConverter, PdfFormatOption


class RagPreprocessDocumentsDocling:

    def __init__(self, pdfs, embedder, export_type=ExportType.DOC_CHUNKS):
        self.pdfs = pdfs
        self.chunker = HybridChunker(
            tokenizer=embedder,
        )
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = False
        pipeline_options.do_formula_enrichment = True
        pipeline_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)

        self.doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
        )
        self.processed_docs = []
        self.export_type = export_type

    def convert_documents(self):
        local_docs = []
        for pdf in self.pdfs:
            result = self.doc_converter.convert(pdf)
            local_docs.extend(result)
        return local_docs

    def preprocess_documents_to_chunks(self):
        for pdf in self.pdfs:
            loader = DoclingLoader(file_path=pdf,
                                   chunker=self.chunker,
                                   export_type=self.export_type,
                                   converter=self.doc_converter)
            docs = loader.load()
            self.processed_docs.extend(docs)

    def get_processed_documents(self):
        return self.processed_docs
