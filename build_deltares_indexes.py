#!/usr/bin/env python3
"""
Build Deltares Indexes - Data Pipeline

This script handles the data pipeline:
1. Scrapes Deltares website using BeautifulSoup
2. Applies 3 different chunking strategies (with 10% overlap)
3. Uses 2 different embedding models
4. Creates 6 different indexes in Azure AI Search

Run this once to build all indexes, then use deltares_rag_demo.py for testing.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import DLLMForge modules
from dllmforge.rag_preprocess_documents import TextChunker
from dllmforge.rag_embedding import AzureOpenAIEmbeddingModel
from dllmforge.rag_search_and_response import IndexManager

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Scraping configuration
MAX_PAGES_TO_SCRAPE = None  # No limit - scrape the entire website!

# Chunking strategies (chunk_size, overlap_size)
CHUNKING_STRATEGIES = {
    "chunks_100_overlap_10": (100, 10),    # 100 characters, 10% overlap
    "chunks_300_overlap_30": (300, 30),    # 300 characters, 10% overlap  
    "chunks_500_overlap_50": (500, 50),    # 500 characters, 10% overlap
}

# Embedding models
EMBEDDING_MODELS = {
    "ada002": {
        "deployment": "AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS_ADA002",
        "model_name": "text-embedding-ada-002",
        "dimensions": 1536
    },
    "text_embedding_3_large": {
        "deployment": "AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS", 
        "model_name": "text-embedding-3-large",
        "dimensions": 3072
    }
}

# Base index name
BASE_INDEX_NAME = "deltares_rag_demo"

# Upload configuration
UPLOAD_BATCH_SIZE = 100  # Number of chunks to process per batch

# =============================================================================


class DeltaresWebScraper:
    """Scrapes Deltares website content using BeautifulSoup with automatic page discovery."""
    
    def __init__(self, base_url="https://www.deltares.nl", max_pages=None):
        """Initialize the web scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = base_url
        self.max_pages = max_pages
        self.scraped_urls = set()
        self.discovered_urls = set()
        self.scraped_content = []
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove common web artifacts
        text = text.replace('\xa0', ' ')  # Non-breaking space
        text = text.replace('\u200b', '')  # Zero-width space
        
        return text.strip()
    
    def extract_text_from_element(self, element) -> str:
        """Extract clean text from a BeautifulSoup element."""
        # Remove script and style elements
        for script in element(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text and clean it
        text = element.get_text()
        return self.clean_text(text)
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for scraping."""
        if not url:
            return False
        
        # Must be from Deltares domain
        if not url.startswith(self.base_url):
            return False
        
        # Skip external domains (even if they start with base_url)
        if '//' in url[8:]:  # After https://
            return False
        
        # Skip certain file types (documents, images, archives, etc.)
        skip_extensions = [
            # Documents
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.rtf', '.odt', '.ods', '.odp',
            # Images
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp', '.ico',
            # Archives
            '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
            # Media
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
            # Other
            '.xml', '.json', '.csv', '.sql', '.log', '.bak', '.tmp'
        ]
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
        
        # Skip certain paths
        skip_paths = [
            '/admin', '/login', '/search', '/api', '/wp-admin', '/wp-content',
            '/download', '/uploads', '/files', '/media', '/assets', '/static',
            '/feed', '/rss', '/sitemap', '/robots.txt', '/favicon.ico',
            '/linkedin.com', '/twitter.com', '/facebook.com', '/youtube.com',
            '/instagram.com', '/sibforms.com'
        ]
        if any(skip_path in url.lower() for skip_path in skip_paths):
            return False
        
        # Only allow HTML pages (no explicit extension or .html/.htm)
        if '.' in url.split('/')[-1] and not url.lower().endswith(('.html', '.htm')):
            return False
        
        # Skip already scraped URLs
        if url in self.scraped_urls:
            return False
        
        return True
    
    def extract_links(self, soup, current_url: str) -> set:
        """Extract all valid links from a page."""
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            
            # Skip empty or javascript links
            if not href or href.startswith('javascript:') or href.startswith('mailto:'):
                continue
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                href = self.base_url + href
            elif href.startswith('./'):
                href = self.base_url + href[1:]
            elif href.startswith('../'):
                # Handle parent directory navigation
                href = self.base_url + href[2:]
            elif not href.startswith('http'):
                # Skip relative links that don't start with /
                continue
            
            # Clean URL (remove fragments, query params)
            href = href.split('#')[0].split('?')[0]
            
            # Normalize URL (remove double slashes except for protocol)
            if '//' in href[8:]:  # After https://
                href = href.replace('//', '/', 1)  # Replace first occurrence after protocol
                href = href.replace('//', '/')  # Replace remaining double slashes
            
            if self.is_valid_url(href):
                links.add(href)
        
        return links
    
    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page and extract content."""
        try:
            print(f"  📄 Scraping: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else "No title"
            
            # Extract main content (focus on main, article, or content areas)
            main_content = ""
            
            # Try different content selectors
            content_selectors = [
                'main',
                'article', 
                '.content',
                '.main-content',
                '#content',
                '#main',
                '.page-content',
                '.entry-content',
                '.post-content'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    main_content = self.extract_text_from_element(content_element)
                    break
            
            # If no main content found, use body
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = self.extract_text_from_element(body)
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ""
            
            # Extract links for discovery
            discovered_links = self.extract_links(soup, url)
            self.discovered_urls.update(discovered_links)
            
            return {
                "url": url,
                "title": self.clean_text(title_text),
                "description": self.clean_text(description),
                "content": main_content,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"  ❌ Error scraping {url}: {e}")
            return {
                "url": url,
                "title": "Error",
                "description": "",
                "content": f"Error scraping this page: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def scrape_deltares_website(self) -> List[Dict[str, Any]]:
        """Scrape Deltares website with automatic page discovery."""
        print("🌐 Starting Deltares website scraping with automatic discovery...")
        
        # Start with the homepage
        urls_to_scrape = [self.base_url + "/en/"]
        scraped_content = []
        
        while urls_to_scrape and (self.max_pages is None or len(scraped_content) < self.max_pages):
            url = urls_to_scrape.pop(0)
            
            if url in self.scraped_urls:
                continue
            
            page_num = len(scraped_content) + 1
            limit_text = f"/{self.max_pages}" if self.max_pages else ""
            print(f"\n[{page_num}{limit_text}] Scraping: {url}")
            
            page_content = self.scrape_page(url)
            self.scraped_urls.add(url)
            
            if page_content["content"] and len(page_content["content"]) > 100:
                scraped_content.append(page_content)
                print(f"  ✅ Extracted {len(page_content['content'])} characters")
            else:
                print(f"  ⚠️  No content extracted or too short")
            
            # Add newly discovered URLs to the queue
            for discovered_url in self.discovered_urls:
                if discovered_url not in self.scraped_urls and discovered_url not in urls_to_scrape:
                    urls_to_scrape.append(discovered_url)
            
            # Be respectful - small delay between requests
            time.sleep(1)
        
        print(f"\n🎉 Scraping complete! Extracted {len(scraped_content)} pages")
        print(f"📊 Discovered {len(self.discovered_urls)} total URLs")
        print(f"🔍 Scraped {len(self.scraped_urls)} URLs")
        
        # Show some discovered URLs for debugging
        print(f"\n🔍 Sample discovered URLs:")
        for i, url in enumerate(list(self.discovered_urls)[:10]):
            print(f"  {i+1}. {url}")
        if len(self.discovered_urls) > 10:
            print(f"  ... and {len(self.discovered_urls) - 10} more")
        
        return scraped_content


class DeltaresIndexBuilder:
    """Builds all Deltares indexes with different configurations."""
    
    def __init__(self):
        """Initialize the index builder."""
        # Check environment variables
        self._check_environment()
        
        # Initialize components
        self.scraper = DeltaresWebScraper(max_pages=MAX_PAGES_TO_SCRAPE)
        
        # Azure configuration
        self.search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        self.search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
        
        if not all([self.search_endpoint, self.search_api_key]):
            raise ValueError("Azure Search credentials not found in environment variables")
        
        print("🚀 Deltares Index Builder initialized")
    
    def _check_environment(self):
        """Check that required environment variables are set."""
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_API_BASE',
            'AZURE_SEARCH_ENDPOINT',
            'AZURE_SEARCH_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Check embedding model deployments
        embedding_deployments = []
        for model_config in EMBEDDING_MODELS.values():
            deployment_var = model_config["deployment"]
            if os.getenv(deployment_var):
                embedding_deployments.append(deployment_var)
        
        if not embedding_deployments:
            raise ValueError("No embedding model deployments found in environment variables")
        
        print(f"✅ Found {len(embedding_deployments)} embedding model deployments")
    
    def prepare_content_for_chunking(self, scraped_content: List[Dict[str, Any]], max_docs: int = None) -> List[Dict[str, Any]]:
        """Prepare scraped content for chunking."""
        print("\n📝 Preparing content for chunking...")
        
        # Limit to small subset for testing if specified
        if max_docs:
            scraped_content = scraped_content[:max_docs]
            print(f"🔬 Using subset of {max_docs} documents for testing")
        
        prepared_content = []
        
        for page in scraped_content:
            # Combine title, description, and content
            full_text = f"Title: {page['title']}\n\n"
            if page['description']:
                full_text += f"Description: {page['description']}\n\n"
            full_text += f"Content: {page['content']}"
            
            prepared_content.append({
                "text": full_text,
                "file_name": f"deltares_{page['url'].split('/')[-2] if page['url'].split('/')[-2] else 'home'}",
                "page_number": 1,
                "url": page["url"],
                "title": page['title']
            })
        
        print(f"✅ Prepared {len(prepared_content)} documents for chunking")
        return prepared_content
    
    def create_chunks_with_strategy(self, content: List[Dict[str, Any]], 
                                  strategy_name: str, chunk_size: int, overlap_size: int) -> List[Dict[str, Any]]:
        """Create chunks using a specific strategy."""
        print(f"  ✂️  Creating chunks with strategy: {strategy_name}")
        print(f"     Chunk size: {chunk_size}, Overlap: {overlap_size}")
        
        chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)
        
        all_chunks = []
        for doc in content:
            # Convert to the format expected by TextChunker: (page_number, text)
            pages_with_text = [(doc["page_number"], doc["text"])]
            chunks = chunker.chunk_text(pages_with_text, doc["file_name"])
            all_chunks.extend(chunks)
        
        print(f"     ✅ Created {len(all_chunks)} chunks")
        return all_chunks
    
    def create_embedding_model(self, model_key: str) -> AzureOpenAIEmbeddingModel:
        """Create an embedding model instance."""
        model_config = EMBEDDING_MODELS[model_key]
        deployment_var = model_config["deployment"]
        
        # Set the deployment environment variable temporarily
        original_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS')
        os.environ['AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS'] = os.getenv(deployment_var)
        
        try:
            embedding_model = AzureOpenAIEmbeddingModel()
            return embedding_model
        finally:
            # Restore original deployment
            if original_deployment:
                os.environ['AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS'] = original_deployment
            else:
                os.environ.pop('AZURE_OPENAI_DEPLOYMENT_EMBEDDINGS', None)
    
    def create_index_and_upload(self, chunks: List[Dict[str, Any]], 
                              strategy_name: str, model_key: str) -> bool:
        """Create an index and upload chunks with iterative upload for large datasets."""
        model_config = EMBEDDING_MODELS[model_key]
        index_name = f"{BASE_INDEX_NAME}_{strategy_name}_{model_key}"
        
        print(f"  🗄️  Creating index: {index_name}")
        
        try:
            # Create embedding model
            embedding_model = self.create_embedding_model(model_key)
            
            # Create index manager
            index_manager = IndexManager(
                self.search_endpoint,
                self.search_api_key,
                index_name,
                model_config["dimensions"]
            )
            
            # Create index
            index_manager.create_index()
            
            # Generate embeddings in batches
            print(f"  🧠 Generating embeddings with {model_config['model_name']}...")
            
            # Upload in batches to avoid timeouts/memory issues
            batch_size = UPLOAD_BATCH_SIZE
            total_chunks = len(chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_chunks + batch_size - 1) // batch_size
                
                print(f"    📦 Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
                
                # Generate embeddings for this batch
                batch_embeddings = embedding_model.embed(batch_chunks)
                
                # Upload this batch
                print(f"    📤 Uploading batch {batch_num}...")
                index_manager.upload_documents(batch_embeddings)
                
                print(f"    ✅ Batch {batch_num} uploaded successfully")
            
            print(f"  ✅ Index {index_name} created successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to create index {index_name}: {e}")
            return False
    
    def build_all_indexes(self, scraped_content: List[Dict[str, Any]], max_docs: int = None) -> Dict[str, bool]:
        """Build all indexes with different configurations."""
        print("\n🏗️  Building all indexes...")
        print("=" * 60)
        
        # Prepare content
        prepared_content = self.prepare_content_for_chunking(scraped_content, max_docs)
        
        # Store results
        results = {}
        
        # Create indexes for all combinations
        for strategy_name, (chunk_size, overlap_size) in CHUNKING_STRATEGIES.items():
            print(f"\n📊 Strategy: {strategy_name}")
            print("-" * 40)
            
            # Create chunks for this strategy
            chunks = self.create_chunks_with_strategy(
                prepared_content, strategy_name, chunk_size, overlap_size
            )
            
            # Create indexes for each embedding model
            for model_key in EMBEDDING_MODELS.keys():
                print(f"\n🔧 Model: {model_key}")
                
                success = self.create_index_and_upload(
                    chunks, strategy_name, model_key
                )
                
                index_key = f"{strategy_name}_{model_key}"
                results[index_key] = success
        
        return results
    
    def save_index_info(self, results: Dict[str, bool], scraped_content: List[Dict[str, Any]]):
        """Save index information for the demo script."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create index info
        index_info = {
            "timestamp": datetime.now().isoformat(),
            "base_index_name": BASE_INDEX_NAME,
            "indexes": {},
            "scraped_pages": len(scraped_content),
            "chunking_strategies": CHUNKING_STRATEGIES,
            "embedding_models": EMBEDDING_MODELS
        }
        
        for index_key, success in results.items():
            if success:
                index_name = f"{BASE_INDEX_NAME}_{index_key}"
                # Parse strategy and model from index key
                parts = index_key.split('_')
                if len(parts) >= 2:
                    # For cases like "small_chunks_ada002" (2 parts)
                    if len(parts) == 2:
                        strategy = parts[0]
                        model = parts[1]
                    # For cases like "small_chunks_text_embedding_3_large" (5 parts)
                    else:
                        # Find where the model name starts by checking against known model keys
                        model_keys = list(EMBEDDING_MODELS.keys())
                        for i in range(len(parts) - 1, 0, -1):
                            potential_model = '_'.join(parts[i:])
                            if potential_model in model_keys:
                                strategy = '_'.join(parts[:i])
                                model = potential_model
                                break
                        else:
                            # Fallback: assume last part is model, rest is strategy
                            strategy = '_'.join(parts[:-1])
                            model = parts[-1]
                else:
                    strategy = "unknown"
                    model = "unknown"
                
                index_info["indexes"][index_key] = {
                    "name": index_name,
                    "status": "created",
                    "strategy": strategy,
                    "model": model
                }
        
        # Save to file
        filename = f"deltares_indexes_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(index_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Index information saved to: {filename}")
        return filename


def main():
    """Main function to build Deltares indexes."""
    # Set batch size for uploads (must be before any use of UPLOAD_BATCH_SIZE)
    global UPLOAD_BATCH_SIZE
    
    parser = argparse.ArgumentParser(description="Build Deltares RAG Indexes")
    parser.add_argument("--skip-scraping", action="store_true",
                       help="Skip website scraping (use cached data)")
    parser.add_argument("--scraped-data", type=str,
                       help="Path to cached scraped data JSON file")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_TO_SCRAPE,
                       help="Maximum number of pages to scrape (default: no limit)")
    parser.add_argument("--test-subset", type=int, default=None,
                       help="Use only N documents for testing (e.g., --test-subset 10)")
    parser.add_argument("--batch-size", type=int, default=UPLOAD_BATCH_SIZE,
                       help=f"Number of chunks to process per batch (default: {UPLOAD_BATCH_SIZE})")
    
    args = parser.parse_args()
    
    # Update the global batch size with command line argument
    UPLOAD_BATCH_SIZE = args.batch_size
    
    print("🏗️  DELTARES INDEX BUILDER")
    print("=" * 60)
    print("🎯 Building 6 indexes with different chunking + embedding combinations")
    print("=" * 60)
    
    try:
        # Initialize builder with custom page limit
        builder = DeltaresIndexBuilder()
        builder.scraper.max_pages = args.max_pages
        
        # Get scraped content
        if args.skip_scraping and args.scraped_data:
            print(f"📂 Loading cached data from: {args.scraped_data}")
            with open(args.scraped_data, 'r', encoding='utf-8') as f:
                scraped_content = json.load(f)
        elif args.skip_scraping:
            print("❌ --skip-scraping requires --scraped-data file path")
            return
        else:
            # Scrape website
            scraped_content = builder.scraper.scrape_deltares_website()
            
            # Save scraped content
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scraped_filename = f"deltares_scraped_content_{timestamp}.json"
            with open(scraped_filename, 'w', encoding='utf-8') as f:
                json.dump(scraped_content, f, indent=2, ensure_ascii=False)
            print(f"💾 Scraped content saved to: {scraped_filename}")
        
        if not scraped_content:
            print("❌ No content to process. Exiting.")
            return
        
        # Build all indexes
        results = builder.build_all_indexes(scraped_content, max_docs=args.test_subset)
        
        # Save index information
        index_info_file = builder.save_index_info(results, scraped_content)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\n🎉 Index building complete!")
        print(f"📊 Results: {successful}/{total} indexes created successfully")
        
        if successful == total:
            print("✅ All indexes created! Ready for demo.")
            print(f"📋 Index info: {index_info_file}")
            print("\n🚀 Next steps:")
            print("   1. Run: python deltares_rag_demo.py")
            print("   2. Enjoy the RAG evaluation! 🎪")
        else:
            print("⚠️  Some indexes failed to create. Check the errors above.")
        
    except Exception as e:
        print(f"❌ Index building failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 