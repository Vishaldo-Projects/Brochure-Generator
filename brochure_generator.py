# Enhanced Professional Company Brochure Generator with Improved Logo Detection
# Better logo detection, content formatting, and image integration

import os
import requests
import json
import time
import base64
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
import google.generativeai as genai
from urllib.parse import urljoin, urlparse
import logging
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from io import BytesIO
from PIL import Image as PILImage
import tempfile
import re
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize and constants
load_dotenv(override=True)
api_key = os.getenv('GEMINI_API_KEY')

if api_key and len(api_key) > 10:
    genai.configure(api_key=api_key)
    logger.info("Gemini API key configured")
else:
    logger.error("Gemini API key issue - please check your .env file and set GEMINI_API_KEY")
    
MODEL = 'gemini-2.5-flash'

# Enhanced headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

class EnhancedWebsite:
    """Enhanced utility class for website scraping with improved logo detection"""
    
    def __init__(self, url: str, timeout: int = 15):
        self.url = url
        self.title = ""
        self.text = ""
        self.links = []
        self.images = []
        self.logo_candidates = []  # Multiple logo candidates
        self.success = False
        
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            self.body = response.content
            soup = BeautifulSoup(self.body, 'html.parser')
            
            # Extract title
            self.title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
            
            # Find all potential logos
            self._find_all_logo_candidates(soup, url)
            
            # Extract images with better categorization
            self._extract_images(soup, url)
            
            # Clean and extract text with better structure preservation
            if soup.body:
                for irrelevant in soup.body(["script", "style", "nav", "footer"]):
                    irrelevant.decompose()
                self.text = self._extract_structured_text(soup.body)
            else:
                self.text = ""
            
            # Extract and normalize links
            raw_links = [link.get('href') for link in soup.find_all('a')]
            self.links = []
            for link in raw_links:
                if link:
                    absolute_link = urljoin(url, link)
                    if urlparse(absolute_link).netloc == urlparse(url).netloc:
                        self.links.append(absolute_link)
            
            self.links = list(dict.fromkeys(self.links))
            self.success = True
            logger.info(f"Successfully scraped {url} with {len(self.images)} images and {len(self.logo_candidates)} logo candidates")
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            self.success = False

    def _find_all_logo_candidates(self, soup, base_url):
        """Find all potential company logos with scoring"""
        logo_candidates = []
        
        # Enhanced logo selectors with priorities
        logo_patterns = [
            # High priority - explicit logo references
            ('img[alt*="logo" i]', 10),
            ('img[src*="logo" i]', 9),
            ('img[class*="logo" i]', 9),
            ('.logo img', 8),
            ('#logo img', 8),
            
            # Medium priority - header and branding areas
            ('header img:first-of-type', 6),
            ('nav img:first-of-type', 5),
            ('.navbar img:first-of-type', 5),
            ('.header img:first-of-type', 5),
            ('.brand img', 7),
            ('.branding img', 7),
            
            # Lower priority - common brand selectors
            ('.navbar-brand img', 6),
            ('.site-logo img', 8),
            ('img[alt*="brand" i]', 5),
            ('img[src*="brand" i]', 5),
        ]
        
        for selector, priority in logo_patterns:
            elements = soup.select(selector)
            for img in elements:
                src = img.get('src')
                if src:
                    logo_url = urljoin(base_url, src)
                    alt_text = img.get('alt', '').lower()
                    
                    # Additional scoring based on image properties
                    score = priority
                    
                    # Boost score for explicit logo terms in alt text
                    if any(term in alt_text for term in ['logo', 'brand', 'company']):
                        score += 3
                    
                    # Reduce score for generic terms
                    if any(term in alt_text for term in ['icon', 'button', 'arrow']):
                        score -= 2
                    
                    # Check image URL for logo indicators
                    url_lower = logo_url.lower()
                    if any(term in url_lower for term in ['logo', 'brand']):
                        score += 2
                    
                    logo_candidates.append({
                        'url': logo_url,
                        'alt': alt_text,
                        'score': score,
                        'selector': selector,
                        'width': img.get('width'),
                        'height': img.get('height')
                    })
        
        # Remove duplicates and sort by score
        seen_urls = set()
        unique_candidates = []
        for candidate in logo_candidates:
            if candidate['url'] not in seen_urls:
                seen_urls.add(candidate['url'])
                unique_candidates.append(candidate)
        
        # Sort by score (highest first)
        unique_candidates.sort(key=lambda x: x['score'], reverse=True)
        self.logo_candidates = unique_candidates[:5]  # Keep top 5 candidates

    def get_best_logo(self):
        """Get the best logo candidate"""
        if not self.logo_candidates:
            return None
        
        # Try to validate logos by checking their actual dimensions
        for candidate in self.logo_candidates:
            try:
                response = requests.head(candidate['url'], headers=headers, timeout=5)
                if response.status_code == 200:
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if 'image' in content_type:
                        return candidate
            except:
                continue
        
        # Fallback to highest scored candidate
        return self.logo_candidates[0] if self.logo_candidates else None

    def _extract_structured_text(self, body_soup):
        """Extract text while preserving structure with better spacing"""
        text_parts = []
        
        # Process headings and paragraphs separately
        for element in body_soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'div']):
            text_content = element.get_text(strip=True)
            if not text_content:
                continue
                
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                text_parts.append(f"\n### {text_content}\n")
            elif element.name == 'p':
                # Add proper spacing for paragraphs
                text_parts.append(f"{text_content}\n")
            elif element.name in ['ul', 'ol']:
                items = element.find_all('li')
                for item in items:
                    item_text = item.get_text(strip=True)
                    if item_text:
                        text_parts.append(f"• {item_text}")
                text_parts.append("")  # Add space after lists
            elif element.name == 'div' and len(text_content) > 50:
                # Only include substantial div content
                text_parts.append(f"{text_content}\n")
        
        return '\n'.join(text_parts)

    def _extract_images(self, soup, base_url):
        """Extract and categorize images from the webpage"""
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src')
            if src:
                img_url = urljoin(base_url, src)
                alt_text = img.get('alt', '')
                
                if self._is_content_image(img_url, alt_text):
                    # Get original dimensions and context
                    width = img.get('width')
                    height = img.get('height')
                    
                    # Try to determine image category
                    category = self._categorize_image(img_url, alt_text, img)
                    
                    self.images.append({
                        'url': img_url,
                        'alt': alt_text,
                        'original_width': width,
                        'original_height': height,
                        'category': category,
                        'relevance_score': self._score_image_relevance(img_url, alt_text)
                    })
        
        # Sort images by relevance score
        self.images.sort(key=lambda x: x['relevance_score'], reverse=True)

    def _categorize_image(self, url, alt_text, img_element):
        """Categorize image type for better content matching"""
        url_lower = url.lower()
        alt_lower = alt_text.lower()
        
        # Check for different image categories
        if any(term in url_lower or term in alt_lower for term in ['product', 'service']):
            return 'product'
        elif any(term in url_lower or term in alt_lower for term in ['team', 'staff', 'employee', 'people']):
            return 'team'
        elif any(term in url_lower or term in alt_lower for term in ['office', 'building', 'facility']):
            return 'location'
        elif any(term in url_lower or term in alt_lower for term in ['hero', 'banner', 'main']):
            return 'hero'
        else:
            return 'general'

    def _score_image_relevance(self, url, alt_text):
        """Score image relevance for content"""
        score = 0
        url_lower = url.lower()
        alt_lower = alt_text.lower()
        
        # Positive scoring
        if any(term in url_lower or term in alt_lower for term in ['product', 'service', 'team', 'company']):
            score += 5
        if any(term in url_lower or term in alt_lower for term in ['hero', 'main', 'feature']):
            score += 3
        if len(alt_text) > 10:  # Descriptive alt text is good
            score += 2
            
        # Negative scoring
        if any(term in url_lower for term in ['thumb', 'small', 'icon']):
            score -= 2
        if any(term in url_lower or term in alt_lower for term in ['ad', 'banner', 'advertisement']):
            score -= 3
            
        return score

    def _is_content_image(self, url, alt_text):
        """Determine if an image is content-worthy with enhanced filtering"""
        url_lower = url.lower()
        
        # Skip obvious non-content images
        skip_patterns = [
            'favicon', 'pixel', 'tracking', 'analytics', 'icon-', 'btn-',
            'arrow', 'bullet', 'spacer', '1x1', 'transparent'
        ]
        if any(pattern in url_lower for pattern in skip_patterns):
            return False
            
        # Skip very small images
        if any(size in url_lower for size in ['16x16', '32x32', '24x24', '1x1']):
            return False
        
        # Skip common icon formats in URL
        if re.search(r'icon[s]?[/_-]', url_lower):
            return False
            
        # Must be a reasonable image format
        if not any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']):
            return False
            
        return True

    def get_contents(self) -> str:
        if not self.success:
            return f"Failed to scrape: {self.url}\n\n"
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text[:3000]}...\n\n"

class ProfessionalPDFGenerator:
    """Enhanced PDF brochure generator with improved formatting"""
    
    def __init__(self, model: str = MODEL):
        self.model = model
        self.scraped_cache = {}
        
        try:
            self.gemini_model = genai.GenerativeModel(model)
            logger.info(f"Gemini model {model} initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
        
        # Enhanced content organization prompt
        self.content_analysis_prompt = """Analyze the website content and organize it into well-structured sections for a professional company brochure.

Create a JSON response with this structure:
{
    "company_name": "Company Name",
    "tagline": "Company tagline or brief description",
    "sections": [
        {
            "title": "Section Title",
            "content": "Well-formatted content with proper paragraph breaks and bullet points where appropriate. Use • for bullets. Each sentence should have proper spacing.",
            "priority": 1,
            "content_type": "overview|products|about|culture|careers|contact"
        }
    ]
}

Format content professionally with:
- Clear bullet points using •
- Short, impactful sentences with proper spacing
- Key information highlighted
- Proper paragraph breaks (use double line breaks between paragraphs)
- Each sentence should be well-spaced for readability

Section priorities: 1=most important, 2=important, 3=supplementary"""

        self.image_analysis_prompt = """Analyze these images and provide detailed descriptions for a professional brochure.

Respond in JSON:
{
    "images": [
        {
            "url": "image_url",
            "description": "Professional description of what the image shows and its relevance",
            "section_match": "which section this image best supports",
            "is_logo": false,
            "category": "product|team|office|hero|general"
        }
    ]
}

Focus on: product images, team photos, office spaces, technology demonstrations, company facilities.
Exclude obvious logos, icons, and decorative elements."""

    def get_organized_content(self, url: str) -> Dict:
        """Analyze and organize website content"""
        try:
            details = self.get_all_details(url)
            prompt = f"{self.content_analysis_prompt}\n\nWebsite content:\n{details}"
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            return json.loads(response.text)
            
        except Exception as e:
            logger.error(f"Error organizing content: {e}")
            return {"company_name": "Unknown Company", "tagline": "", "sections": []}

    def analyze_images(self, website: EnhancedWebsite) -> List[Dict]:
        """Analyze images for relevance and descriptions"""
        try:
            if not website.images:
                return []
                
            # Filter to top relevant images
            relevant_images = [img for img in website.images if img['relevance_score'] > 0][:10]
            
            image_info = "\n".join([
                f"URL: {img['url']}, Alt: {img['alt']}, Category: {img['category']}" 
                for img in relevant_images
            ])
            
            prompt = f"{self.image_analysis_prompt}\n\nAvailable images:\n{image_info}"
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            return result.get('images', [])
            
        except Exception as e:
            logger.error(f"Error analyzing images: {e}")
            return []

    def download_and_process_image(self, image_url: str, max_width: float = 3.5*inch, is_logo: bool = False) -> Optional[Tuple[str, float, float, bytes]]:
        """Download image and return temp file path with proper sizing and image data"""
        try:
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Open with PIL to get dimensions
            img = PILImage.open(BytesIO(response.content))
            original_width, original_height = img.size
            
            # Calculate aspect ratio
            aspect_ratio = original_height / original_width
            
            # Different sizing for logos vs content images
            if is_logo:
                # Logo sizing - keep reasonable proportions
                max_logo_width = 2.5*inch
                max_logo_height = 1.5*inch
                
                # Calculate size maintaining aspect ratio
                if aspect_ratio > (max_logo_height / max_logo_width):
                    # Height is the limiting factor
                    final_height = max_logo_height
                    final_width = max_logo_height / aspect_ratio
                else:
                    # Width is the limiting factor
                    final_width = max_logo_width
                    final_height = max_logo_width * aspect_ratio
            else:
                # Content image sizing
                if original_width > max_width * 72:  # Convert inches to pixels (72 DPI)
                    final_width = max_width
                    final_height = max_width * aspect_ratio
                else:
                    # Use original size if reasonable
                    final_width = min(original_width / 72, max_width)
                    final_height = final_width * aspect_ratio
                
                # Ensure minimum size for content images
                if final_width < 1.5*inch:
                    final_width = 1.5*inch
                    final_height = 1.5*inch * aspect_ratio
            
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Save to BytesIO instead of temp file to avoid file system issues
            img_bytes = BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            img_bytes.seek(0)
            
            # Also create a temp file as backup
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file.name, 'JPEG', quality=85)
            temp_file.close()
            
            return temp_file.name, final_width, final_height, img_bytes.getvalue()
            
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {e}")
            return None

    def create_custom_styles(self):
        """Create custom styles with better spacing"""
        styles = getSampleStyleSheet()
        
        # Define custom styles with enhanced spacing
        custom_styles = [
            ParagraphStyle(
                name='CompanyTitle',
                parent=styles['Title'],
                fontSize=32,
                spaceAfter=12,
                spaceBefore=6,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#1a365d'),
                fontName='Helvetica-Bold',
                leading=38
            ),
            
            ParagraphStyle(
                name='Tagline',
                parent=styles['Normal'],
                fontSize=16,
                spaceAfter=24,
                spaceBefore=6,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#4a5568'),
                fontName='Helvetica-Oblique',
                leading=20
            ),
            
            ParagraphStyle(
                name='SectionTitle',
                parent=styles['Heading2'],
                fontSize=20,
                spaceAfter=16,
                spaceBefore=24,
                textColor=colors.HexColor('#2d3748'),
                fontName='Helvetica-Bold',
                leading=24
            ),
            
            ParagraphStyle(
                name='CustomBodyText',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=10,
                spaceBefore=2,
                alignment=TA_JUSTIFY,
                leading=18,  # Increased line spacing
                leftIndent=0,
                rightIndent=0,
                fontName='Helvetica'
            ),
            
            ParagraphStyle(
                name='CustomBulletPoint',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=6,
                spaceBefore=2,
                leftIndent=24,
                bulletIndent=12,
                leading=16,
                fontName='Helvetica'
            ),
            
            ParagraphStyle(
                name='ImageCaption',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=12,
                spaceBefore=4,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#718096'),
                fontName='Helvetica-Oblique',
                leading=12
            ),
            
            ParagraphStyle(
                name='CustomFooter',
                parent=styles['Normal'],
                fontSize=11,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#a0aec0'),
                fontName='Helvetica',
                leading=14
            )
        ]
        
        # Add custom styles to the stylesheet
        for style in custom_styles:
            try:
                existing_style = styles[style.name]
            except KeyError:
                styles.add(style)
        
        return styles

    def create_two_column_content(self, text: str, image_data: bytes, image_caption: str, 
                                image_width: float, image_height: float, image_on_left: bool, styles):
        """Create alternating two-column layout with enhanced spacing"""
        content = []
        
        # Create table for two-column layout
        if image_on_left:
            table_data = [[
                self._create_image_cell(image_data, image_caption, image_width, image_height, styles),
                self._create_text_cell(text, styles)
            ]]
            col_widths = [3.5*inch, 4*inch]
        else:
            table_data = [[
                self._create_text_cell(text, styles),
                self._create_image_cell(image_data, image_caption, image_width, image_height, styles)
            ]]
            col_widths = [4*inch, 3.5*inch]
        
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        content.append(table)
        content.append(Spacer(1, 20))
        
        return content

    def _create_image_cell(self, image_data: bytes, caption: str, width: float, height: float, styles):
        """Create image cell with caption using image bytes"""
        cell_content = []
        
        try:
            # Create image from bytes to avoid file path issues
            img = Image(BytesIO(image_data), width=width, height=height)
            cell_content.append(img)
            cell_content.append(Spacer(1, 8))
            
            if caption:
                cell_content.append(Paragraph(caption, styles['ImageCaption']))
                
        except Exception as e:
            logger.error(f"Error creating image cell: {e}")
            cell_content.append(Paragraph("Image could not be loaded", styles['ImageCaption']))
        
        return cell_content

    def _create_text_cell(self, text: str, styles):
        """Create formatted text cell with proper spacing"""
        cell_content = []
        
        # Split text into paragraphs with better spacing
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            lines = paragraph.split('\n')
            current_paragraph = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if it's a bullet point
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    if current_paragraph:
                        cell_content.append(Paragraph(current_paragraph, styles['CustomBodyText']))
                        current_paragraph = ""
                    
                    bullet_text = line.lstrip('•-* ').strip()
                    cell_content.append(Paragraph(f"• {bullet_text}", styles['CustomBulletPoint']))
                else:
                    if current_paragraph:
                        current_paragraph += " " + line
                    else:
                        current_paragraph = line
            
            # Add remaining paragraph
            if current_paragraph:
                cell_content.append(Paragraph(current_paragraph, styles['CustomBodyText']))
                cell_content.append(Spacer(1, 8))  # Add space between paragraphs
        
        return cell_content

    def format_content_with_bullets(self, content: str) -> str:
        """Enhanced content formatting with better spacing"""
        format_prompt = f"""Format this content for a professional brochure with excellent readability:

        {content}

        Requirements:
        - Add proper line spacing between sentences
        - Use bullet points (•) where appropriate to break up dense text
        - Create clear paragraph breaks
        - Make it engaging and easy to scan
        - Ensure professional tone
        - Each paragraph should be well-spaced and readable

        Return the formatted content with proper spacing and structure."""
        
        try:
            response = self.gemini_model.generate_content(
                format_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.3)
            )
            return response.text
        except Exception as e:
            logger.error(f"Error formatting content: {e}")
            return content

    def create_professional_pdf(self, company_name: str, url: str, output_filename: str = None) -> str:
        """Generate enhanced professional PDF brochure"""
        try:
            # Get organized content
            logger.info("Analyzing website content...")
            main_website = EnhancedWebsite(url)
            if not main_website.success:
                raise Exception(f"Failed to scrape main website: {url}")
            
            organized_content = self.get_organized_content(url)
            company_name = organized_content.get('company_name', company_name)
            tagline = organized_content.get('tagline', '')
            sections = organized_content.get('sections', [])
            
            # Get best logo
            logger.info("Finding best company logo...")
            best_logo = main_website.get_best_logo()
            logo_data = None
            logo_width = None
            logo_height = None
            
            if best_logo:
                logger.info(f"Found logo: {best_logo['url']}")
                logo_result = self.download_and_process_image(best_logo['url'], is_logo=True)
                if logo_result:
                    _, logo_width, logo_height, logo_data = logo_result
                    logger.info("Logo successfully processed")
            
            # Analyze content images
            logger.info("Analyzing content images...")
            analyzed_images = self.analyze_images(main_website)
            
            # Download content images and store as bytes to avoid temp file issues
            section_images = {}
            image_toggle = True
            temp_files_to_cleanup = []  # Track temp files for cleanup
            
            for img_info in analyzed_images[:8]:  # Limit to 8 images
                img_result = self.download_and_process_image(img_info['url'])
                if img_result:
                    temp_path, width, height, img_bytes = img_result
                    temp_files_to_cleanup.append(temp_path)
                    
                    section_images[img_info['section_match']] = {
                        'data': img_bytes,
                        'width': width,
                        'height': height,
                        'description': img_info['description'],
                        'on_left': image_toggle
                    }
                    image_toggle = not image_toggle
                    logger.info(f"Downloaded image for {img_info['section_match']}")
            
            # Create PDF
            if not output_filename:
                output_filename = f"{company_name.replace(' ', '_')}_brochure.pdf"
            
            logger.info(f"Creating enhanced PDF: {output_filename}")
            
            doc = SimpleDocTemplate(
                output_filename, 
                pagesize=A4,
                rightMargin=0.75*inch, 
                leftMargin=0.75*inch,
                topMargin=0.75*inch, 
                bottomMargin=0.75*inch
            )
            
            # Build content
            story = []
            styles = self.create_custom_styles()
            
            # === ENHANCED TITLE PAGE ===
            story.append(Spacer(1, 0.3*inch))
            
            # Company logo (if available)
            # # Company logo (if available)
            if logo_data:
                try:
                    logo_img = Image(BytesIO(logo_data), width=logo_width, height=logo_height)
                    story.append(logo_img)
                    story.append(Spacer(1, 0.4*inch))
                except Exception as e:
                    logger.error(f"Error adding logo: {e}")
                    story.append(Spacer(1, 0.4*inch))
                else:
                    story.append(Spacer(1, 0.4*inch))

            
            # Company name with enhanced styling
            story.append(Paragraph(company_name, styles['CompanyTitle']))
            
            # Tagline with better spacing
            if tagline:
                story.append(Paragraph(tagline, styles['Tagline']))
            else:
                story.append(Spacer(1, 0.4*inch))
            
            # Add decorative separator
            story.append(Spacer(1, 0.3*inch))
            
            # === ENHANCED CONTENT SECTIONS ===
            sections.sort(key=lambda x: x.get('priority', 3))
            
            for i, section in enumerate(sections):
                section_title = section['title']
                section_content = section['content']
                content_type = section.get('content_type', 'general')
                
                # Enhanced content formatting with better spacing
                formatted_content = self.format_content_with_bullets(section_content)
                
                # Add section title with proper spacing
                story.append(Paragraph(section_title, styles['SectionTitle']))
                
                # Find matching image for this section
                matching_image = None
                for section_key, img_data in section_images.items():
                    if (section_key.lower() in section_title.lower() or 
                        section_title.lower() in section_key.lower() or
                        content_type.lower() in section_key.lower()):
                        matching_image = img_data
                        break
                
                # If no exact match, try to match by content type
                if not matching_image:
                    for section_key, img_data in section_images.items():
                        if content_type.lower() in section_key.lower():
                            matching_image = img_data
                            break
                
                if matching_image:
                    # Create two-column layout with image and text using image bytes
                    content_items = self.create_two_column_content(
                        formatted_content,
                        matching_image['data'],
                        matching_image['description'],
                        matching_image['width'],
                        matching_image['height'],
                        matching_image['on_left'],
                        styles
                    )
                    story.extend(content_items)
                    # Remove this image so it's not reused
                    del section_images[section_key]
                else:
                    # Text-only section with better spacing
                    text_content = self._create_text_cell(formatted_content, styles)
                    story.extend(text_content)
                    story.append(Spacer(1, 20))
            
            # === REMAINING IMAGES SECTION ===
            if section_images:
                story.append(Paragraph("Additional Information", styles['SectionTitle']))
                
                # Add remaining images in a nice layout
                for section_key, img_data in list(section_images.items())[:3]:  # Max 3 additional images
                    content_items = self.create_two_column_content(
                        f"Learn more about our {section_key} and discover what makes us unique.",
                        img_data['data'],
                        img_data['description'],
                        img_data['width'],
                        img_data['height'],
                        img_data['on_left'],
                        styles
                    )
                    story.extend(content_items)
            
            # === ENHANCED FOOTER ===
            story.append(Spacer(1, 0.4*inch))
            
            # Add elegant separator line
            line_table = Table([['']],  colWidths=[6.5*inch])
            line_table.setStyle(TableStyle([
                ('LINEABOVE', (0, 0), (-1, -1), 2, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(line_table)
            
            # Website URL in footer with better styling
            story.append(Paragraph(f"<b>Visit us online:</b> {url}", styles['CustomFooter']))
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(f"Generated on: {time.strftime('%B %d, %Y')}", styles['CustomFooter']))
            
            # Build the PDF
            doc.build(story)
            
            # Clean up temp files after PDF is built
            for temp_file in temp_files_to_cleanup:
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Could not delete temp file {temp_file}: {e}")
            
            logger.info(f"Enhanced professional PDF brochure created: {output_filename}")
            return output_filename
            
        except Exception as e:
            logger.error(f"Error creating enhanced PDF brochure: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def get_relevant_links(self, url: str) -> Dict:
        """Get relevant links using enhanced Gemini analysis"""
        try:
            website = EnhancedWebsite(url)
            if not website.success:
                return {"links": []}
            
            link_prompt = """Analyze these website links and select the most valuable ones for a comprehensive company brochure.

JSON response format:
{
    "links": [
        {"type": "about page", "url": "full_url", "priority": 1},
        {"type": "products page", "url": "full_url", "priority": 1},
        {"type": "services page", "url": "full_url", "priority": 1},
        {"type": "team page", "url": "full_url", "priority": 2},
        {"type": "careers page", "url": "full_url", "priority": 2}
    ]
}

Priority levels: 1=essential, 2=important, 3=supplementary
Include: About, Products/Services, Team, Careers, Company Culture, Technology pages.
Exclude: Contact forms, Terms, Privacy, Login, Social media links."""
            
            user_prompt = f"Website: {website.url}\nAvailable links:\n" + "\n".join(website.links[:60])
            prompt = f"{link_prompt}\n\n{user_prompt}"
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            # Sort by priority
            links = result.get('links', [])
            links.sort(key=lambda x: x.get('priority', 3))
            
            return {"links": links}
            
        except Exception as e:
            logger.error(f"Error getting relevant links: {e}")
            return {"links": []}

    def get_all_details(self, url: str) -> str:
        """Enhanced scraping of main page and relevant subpages"""
        result = "=== MAIN PAGE ===\n"
        main_site = EnhancedWebsite(url)
        result += main_site.get_contents()
        
        if not main_site.success:
            return result
            
        links_data = self.get_relevant_links(url)
        relevant_links = links_data['links']
        
        logger.info(f"Found {len(relevant_links)} relevant links for detailed analysis")
        
        # Process links by priority
        for link_info in relevant_links[:6]:  # Limit to top 6 subpages
            try:
                time.sleep(1.5)  # Rate limiting
                
                link_url = link_info["url"]
                link_type = link_info["type"]
                
                if link_url in self.scraped_cache:
                    subpage = self.scraped_cache[link_url]
                else:
                    logger.info(f"Scraping {link_type}: {link_url}")
                    subpage = EnhancedWebsite(link_url)
                    self.scraped_cache[link_url] = subpage
                
                if subpage.success:
                    result += f"\n\n=== {link_type.upper()} ===\n"
                    result += subpage.get_contents()
                    
            except Exception as e:
                logger.error(f"Error scraping {link_info.get('url', 'unknown')}: {e}")
                continue
                
        return result

def create_enhanced_professional_brochure():
    """Enhanced interactive function to create professional PDF brochure"""
    
    print("🚀 === Enhanced Professional PDF Brochure Generator ===")
    print("✨ Features:")
    print("   • Intelligent logo detection and placement")
    print("   • Enhanced content formatting with proper spacing")
    print("   • Smart image integration with content matching")
    print("   • Professional typography with improved readability")
    print("   • Alternating image layouts for visual appeal")
    print("   • Multi-page content analysis")
    print()
    
    # Get URL from user
    website_url = input("Enter the website URL: ").strip()
    
    # Validate and normalize URL
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url
    
    try:
        # Extract company name from URL
        domain = urlparse(website_url).netloc
        company_name = domain.replace('www.', '').split('.')[0].title()
        
        print(f"\n🔍 Analyzing website: {website_url}")
        print(f"🏢 Detected company name: {company_name}")
        
        # Ask for custom company name
        custom_name = input(f"Company name (press Enter for '{company_name}'): ").strip()
        if custom_name:
            company_name = custom_name
        
        # Create enhanced generator
        generator = ProfessionalPDFGenerator()
        
        print(f"\n🚀 Generating enhanced professional brochure for {company_name}...")
        print("📊 Step 1: Advanced website content analysis...")
        print("🖼️  Step 2: Intelligent logo detection...")
        print("🤖 Step 3: AI-powered content organization...")
        print("📸 Step 4: Smart image processing and categorization...")
        print("📄 Step 5: Creating professional PDF with enhanced formatting...")
        
        output_file = generator.create_professional_pdf(company_name, website_url)
        
        print(f"\n✅ Enhanced professional PDF brochure created: {output_file}")
        print(f"🌟 Enhanced features applied:")
        print(f"   • ✅ Improved logo detection and placement")
        print(f"   • ✅ Enhanced line spacing and readability")
        print(f"   • ✅ Smart content-image matching")
        print(f"   • ✅ Professional typography and layout")
        print(f"   • ✅ Multi-section content organization")
        print(f"   • ✅ Elegant formatting with proper spacing")
        print(f"   • ✅ Company branding integration")
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ Error creating enhanced brochure: {e}")
        logger.error(f"Error in create_enhanced_professional_brochure: {e}")
        return None

# Enhanced standalone functions
def generate_enhanced_pdf_brochure(company_name: str, url: str, output_filename: str = None) -> str:
    """Generate enhanced professional PDF brochure programmatically"""
    generator = ProfessionalPDFGenerator()
    return generator.create_professional_pdf(company_name, url, output_filename)

def quick_enhanced_brochure_demo():
    """Enhanced quick demo with popular websites"""
    print("🎯 Enhanced Quick Demo - Professional Brochure Generator")
    print("\nSelect a demo website:")
    print("1. OpenAI (https://openai.com)")
    print("2. Anthropic (https://anthropic.com)")  
    print("3. HuggingFace (https://huggingface.co)")
    print("4. Tesla (https://tesla.com)")
    print("5. Custom URL")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    demo_urls = {
        '1': ('OpenAI', 'https://openai.com'),
        '2': ('Anthropic', 'https://anthropic.com'),
        '3': ('HuggingFace', 'https://huggingface.co'),
        '4': ('Tesla', 'https://tesla.com')
    }
    
    if choice in demo_urls:
        company_name, url = demo_urls[choice]
        print(f"\n🚀 Creating enhanced brochure for {company_name}...")
        
        try:
            generator = ProfessionalPDFGenerator()
            output_file = generator.create_professional_pdf(company_name, url)
            print(f"✅ Enhanced demo brochure created: {output_file}")
            return output_file
        except Exception as e:
            print(f"❌ Enhanced demo failed: {e}")
            return None
    elif choice == '5':
        return create_enhanced_professional_brochure()
    else:
        print("Invalid choice!")
        return None

# Example usage and main execution
if __name__ == "__main__":
    print("🚀 Enhanced Professional PDF Brochure Generator")
    print("=" * 55)
    print("🌟 ENHANCED FEATURES:")
    print("• 🎯 Intelligent logo detection with scoring system")
    print("• 📝 Enhanced content formatting with proper line spacing")
    print("• 🖼️  Smart image-content matching and categorization")
    print("• 🎨 Professional typography with improved readability")
    print("• 📊 Multi-page content analysis and organization")
    print("• ✨ Elegant layouts with alternating image positions")
    print("• 🏢 Company branding integration throughout")
    print("• 📱 Responsive image sizing with aspect ratio preservation")
    print()
    print("📋 Requirements:")
    print("• Gemini API key in .env file (GEMINI_API_KEY=your_key)")
    print("• pip install google-generativeai reportlab pillow beautifulsoup4 requests python-dotenv")
    
    print("\n" + "="*55)
    
    # Choose mode
    print("\nChoose an option:")
    print("1. 🎨 Create enhanced brochure for your website")
    print("2. ⚡ Quick demo with sample websites")
    print("3. 🚪 Exit")
    
    mode = input("\nEnter choice (1-3): ").strip()
    
    if mode == '1':
        create_enhanced_professional_brochure()
    elif mode == '2':
        quick_enhanced_brochure_demo()
    elif mode == '3':
        print("👋 Goodbye!")
    else:
        print("Invalid choice. Running enhanced interactive mode...")
        create_enhanced_professional_brochure()