"""
Document Processor module for preprocessing documents into text or images for LLM processing.
"""
import io
import base64
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from PIL import Image
import fitz  # PyMuPDF
from utils.document_loader import DocumentLoader
from IE_agent_config import DocumentConfig

class ProcessedDocument:
    """Class representing processed document content"""
    def __init__(self, 
                 content: Union[str, bytes],
                 content_type: str,
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize processed document
        
        Args:
            content: The document content (text string or image bytes)
            content_type: Type of content ('text' or 'image')
            metadata: Additional metadata about the document
        """
        self.content = content
        self.content_type = content_type
        self.metadata = metadata or {}
        self.content_length = len(content) if isinstance(content, str) else len(content)

class DocumentProcessor:
    """Class for preprocessing documents into text or images"""
    
    def __init__(self, config: DocumentConfig):
        """Initialize document processor
        
        Args:
            config: Document processing configuration
        """
        self.config = config
        self.document_loader = DocumentLoader()

    def process_to_text(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """Process document to text using DocumentLoader"""
        file_path = Path(file_path)
        text = self.document_loader.load_document(file_path)
        
        return ProcessedDocument(
            content=text,
            content_type='text',
            metadata={
                'source_file': str(file_path),
                'file_type': file_path.suffix.lower(),
                'text_length': len(text)
            }
        )

    def process_to_image(self, file_path: Union[str, Path]) -> List[ProcessedDocument]:
        """Process document to list of page images"""
        file_path = Path(file_path)
        processed_images = []
        
        if file_path.suffix.lower() == '.pdf':
            # Convert PDF pages to images
            doc = fitz.open(str(file_path))
            for page_num, page in enumerate(doc):
                # Convert page to high-quality image
                zoom = 4  # Increased zoom for better quality
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                if pix.width == 0 or pix.height == 0:
                    continue
                
                # Convert to PIL Image and then to bytes
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=95)
                img_bytes = img_byte_arr.getvalue()
                
                processed_images.append(ProcessedDocument(
                    content=img_bytes,
                    content_type='image',
                    metadata={
                        'source_file': str(file_path),
                        'page_number': page_num + 1,
                        'image_size_bytes': len(img_bytes)
                    }
                ))
            
            doc.close()
        else:
            # For other image formats, just read and optionally resize
            img = Image.open(file_path)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_bytes = img_byte_arr.getvalue()
            
            processed_images.append(ProcessedDocument(
                content=img_bytes,
                content_type='image',
                metadata={
                    'source_file': str(file_path),
                    'image_size_bytes': len(img_bytes)
                }
            ))
        
        return processed_images

    def encode_image_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')

    def process_document(self, file_path: Union[str, Path]) -> Union[ProcessedDocument, List[ProcessedDocument]]:
        """Process document based on configuration
        
        Args:
            file_path: Path to document
            
        Returns:
            Single ProcessedDocument for text or list of ProcessedDocument for images
        """
        if self.config.output_type == 'text':
            return self.process_to_text(file_path)
        elif self.config.output_type == 'image':
            return self.process_to_image(file_path)
        else:
            raise ValueError(f"Unsupported output type: {self.config.output_type}")

    def process_directory(self) -> List[Union[ProcessedDocument, List[ProcessedDocument]]]:
        """Process all matching files in the configured directory"""
        input_dir = Path(self.config.input_dir)
        
        # Find all matching files
        files = list(input_dir.glob(self.config.file_pattern))
        if not files:
            print(f"No files found matching pattern '{self.config.file_pattern}' in {input_dir}")
            return []
        
        print(f"Found {len(files)} files to process")
        
        # Process each file
        processed_docs = []
        for file_path in files:
            try:
                processed_doc = self.process_document(file_path)
                processed_docs.append(processed_doc)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return processed_docs

if __name__ == "__main__":
    # Example 1: Convert PDF to text
    test_pdf_path = r"c:\Users\deng_jg\work\16centralized_agents\test_data\lstm_low_flow.pdf"
    # Configure document processor for text output
    text_config = DocumentConfig(
        input_dir=Path(test_pdf_path).parent,
        file_pattern="*.pdf",
        output_type="text"
    )
    text_processor = DocumentProcessor(text_config)
    # Process PDF to text
    processed_text = text_processor.process_to_text(test_pdf_path)
    print("\nText Processing Results:")
    print(f"Source file: {processed_text.metadata['source_file']}")
    print(f"Text length: {processed_text.metadata['text_length']} characters")
    print("First 500 characters of extracted text:")
    print(processed_text.content[:500])
    
    # Example 2: Convert PDF to images
    test_pdf_path = r"c:\Users\deng_jg\work\16centralized_agents\test_data\Kratzert2018_Rainfall–runoff modelling using Long Short-Term.pdf"
    # Configure document processor for image output
    image_config = DocumentConfig(
        input_dir=Path(test_pdf_path).parent,
        file_pattern="*.pdf",
        output_type="image"
    )
    image_processor = DocumentProcessor(image_config)
    # Process PDF to images
    processed_images = image_processor.process_to_image(test_pdf_path)
    print("\nImage Processing Results:")
    print(f"Source file: {processed_images[0].metadata['source_file']}")
    print(f"Number of pages processed: {len(processed_images)}")
    for i, img_doc in enumerate(processed_images):
        print(f"Page {img_doc.metadata['page_number']}: {img_doc.metadata['image_size_bytes'] / 1024:.1f} KB")
        
    # Example 3: Process all PDFs in directory using process_directory()
    test_dir = r"c:\Users\deng_jg\work\16centralized_agents\test_data"
    print("\nBatch Processing Results using process_directory():")
    # First process all PDFs to text
    text_config = DocumentConfig(
        input_dir=Path(test_dir),
        file_pattern="*.pdf",
        output_type="text"
    )
    text_processor = DocumentProcessor(text_config)
    print("\nProcessing all PDFs to text:")
    text_results = text_processor.process_directory()
    for result in text_results:
        print(f"File: {result.metadata['source_file']}")
        print(f"Text length: {result.metadata['text_length']} characters")


