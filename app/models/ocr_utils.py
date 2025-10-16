import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from dateutil.parser import parse
import os
import logging
from pdf2image import convert_from_path
import tempfile
from utils.category_standardizer import category_standardizer

# Set Tesseract path to your specific installation
pytesseract.pytesseract.tesseract_cmd = r"E:\SLIT\Y3S1\IRWA\Tesseract-OCR\tesseract.exe"

# Add Poppler path (update this to your actual poppler path)
POPPLER_PATH = r"E:\SLIT\Y3S1\IRWA\poppler-25.07.0\Library\bin"

def check_tesseract_available():
    """Check if Tesseract is available at the specified path"""
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR is available")
        return True
    except Exception as e:
        print(f"❌ Tesseract not available: {e}")
        return False

def preprocess_image(image_path):
    """Open image, convert to grayscale, enhance contrast"""
    try:
        img = Image.open(image_path).convert('L')  # grayscale
        img = img.filter(ImageFilter.MedianFilter())  # reduce noise
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)  # increase contrast
        img = img.point(lambda x: 0 if x < 150 else 255)  # simple threshold
        return img
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        # Fallback: return original image
        return Image.open(image_path).convert('L')

def extract_text_from_image(image_path):
    """Extract raw text from the image using Tesseract OCR"""
    try:
        # Check if Tesseract is available
        if not check_tesseract_available():
            return ""
        
        # Check if file is PDF
        if image_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(image_path)
        
        # Handle image files
        img = preprocess_image(image_path)
        
        # Configure Tesseract for better receipt reading
        custom_config = r'--oem 3 --psm 6'
        
        text = pytesseract.image_to_string(img, config=custom_config)
        print(f"📄 Extracted text length: {len(text)} characters")
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files by converting to images"""
    try:
        # Check if Tesseract is available
        if not check_tesseract_available():
            return ""
            
        # Convert PDF to images with poppler path
        if os.path.exists(POPPLER_PATH):
            images = convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)
        else:
            # Try without poppler path (might work if poppler is in PATH)
            images = convert_from_path(pdf_path, dpi=200)
        
        all_text = ""
        for i, image in enumerate(images):
            # Save temporary image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                image.save(temp_path, 'PNG')
            
            # Extract text from the image
            text = extract_text_from_image(temp_path)
            all_text += text + "\n"
            print(f"📄 PDF page {i+1} extracted: {len(text)} characters")
            
            # Clean up temp file
            os.unlink(temp_path)
            
        return all_text.strip()
    except Exception as e:
        print(f"PDF processing failed: {e}")
        return ""

def extract_amount(text):
    """Extract total amount from receipt text"""
    try:
        if not text:
            return 0.0
            
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        total_keywords = ['total', 'amount', 'balance', 'grand total', 'subtotal', 'final', 'due', 'paid']
        
        candidates = []
        for line in lines:
            line_lower = line.lower()
            if any(k in line_lower for k in total_keywords):
                # Extract numbers with currency symbols - more flexible pattern
                amount_patterns = [
                    r'[\$€£₹]?\s*(\d{1,6}[.,]\d{1,2})',  # $123.45 or 123,45
                    r'[\$€£₹]?\s*(\d{1,6})',             # $123 or 123
                    r'(\d{1,6}[.,]\d{1,2})',             # 123.45 or 123,45
                    r'(\d{1,6})'                         # 123
                ]
                
                for pattern in amount_patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        # Take the last match in total lines (usually the actual total)
                        amount_str = matches[-1].replace(',', '.')
                        try:
                            amount = float(amount_str)
                            # Only consider reasonable amounts
                            if 0.1 <= amount <= 100000:
                                candidates.append(amount)
                                print(f"💰 Found amount in total line: {amount}")
                        except ValueError:
                            continue
        
        if candidates:
            final_amount = max(candidates)
            print(f"✅ Using amount: {final_amount}")
            return final_amount
        
        # Fallback: find all numbers in the entire text
        print("🔍 Falling back to general number search...")
        all_amounts = re.findall(r'(\d{1,6}[.,]\d{1,2})', text)
        all_amounts.extend(re.findall(r'(\d{1,6})', text))
        
        if all_amounts:
            amounts = []
            for amt_str in all_amounts:
                try:
                    amount = float(amt_str.replace(',', '.'))
                    if 0.1 <= amount <= 10000:  # Reasonable range for receipts
                        amounts.append(amount)
                except ValueError:
                    continue
            
            if amounts:
                final_amount = max(amounts)
                print(f"✅ Using fallback amount: {final_amount}")
                return final_amount
        
        print("❌ No valid amount found")
        return 0.0
    except Exception as e:
        print(f"Amount extraction failed: {e}")
        return 0.0

def extract_date(text):
    """Extract date from receipt text"""
    try:
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',      # 11/10/2025 or 11-10-25
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',        # 2025-10-11
            r'\d{1,2}\s+\w{3,9}\s+\d{2,4}',        # 11 Oct 2025
            r'\w{3,9}\s+\d{1,2},?\s+\d{4}',        # Oct 11, 2025 or Oct 11 2025
            r'\d{1,2}\s+\w{3,9}\s+\d{2,4}'         # 11 October 25
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    parsed_date = parse(match, fuzzy=True)
                    date_str = parsed_date.strftime('%Y-%m-%d')
                    print(f"📅 Found date: {date_str}")
                    return date_str
                except Exception as e:
                    print(f"Date parsing failed for '{match}': {e}")
                    continue
        
        # If no date found, return today's date
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        print(f"📅 No date found, using today: {today}")
        return today
    except Exception as e:
        print(f"Date extraction failed: {e}")
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d')

def extract_note(text):
    """Extract meaningful note from receipt text"""
    try:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Skip lines that are likely headers or totals
        skip_keywords = ['total', 'subtotal', 'tax', 'amount', 'balance', 'receipt', 'invoice', 'change', 'cash', 'card']
        meaningful_lines = []
        
        for line in lines:
            line_lower = line.lower()
            # Skip if it's a total line, very short, or just numbers/symbols
            if (any(kw in line_lower for kw in skip_keywords) or 
                len(line) < 3 or 
                line.isdigit() or
                re.match(r'^[\d\s.,$€£₹]+$', line) or
                'page' in line_lower or
                'date' in line_lower):
                continue
            meaningful_lines.append(line)
        
        # Take first 2 meaningful lines or first 100 characters
        if meaningful_lines:
            note = " | ".join(meaningful_lines[:3])  # Use up to 3 lines separated by |
            note = note[:100]  # Limit length
            print(f"📝 Extracted note: {note}")
            return note
        else:
            # Fallback: first line that's not empty
            for line in lines:
                if line.strip() and len(line.strip()) > 3:
                    note = line.strip()[:100]
                    print(f"📝 Using fallback note: {note}")
                    return note
        
        note = "Receipt scan"
        print(f"📝 Using default note: {note}")
        return note
    except Exception as e:
        print(f"Note extraction failed: {e}")
        return "Receipt scan"

def extract_category(text):
    """Rule-based categorization from text with standardization"""
    try:
        text_lower = text.lower()
        
        # Use the standardizer
        for standard_category, variations in category_standardizer.standard_categories.items():
            for variation in variations:
                if variation in text_lower:
                    print(f"🏷️ OCR categorized: '{text}' → {standard_category}")
                    return standard_category
        
        print(f"🏷️ OCR categorized: '{text}' → Other")
        return "Other"
    except Exception as e:
        print(f"Category extraction failed: {e}")
        return "Other"

def extract_transaction_type(text):
    """Determine if transaction is income or expense based on text"""
    try:
        text_lower = text.lower()
        
        income_keywords = ['salary', 'bonus', 'refund', 'deposit', 'payment received', 'income', 'stipend', 'invoice paid']
        expense_keywords = ['purchase', 'payment', 'bill', 'fee', 'charge', 'paid', 'spent', 'bought']
        
        if any(keyword in text_lower for keyword in income_keywords):
            print("💵 Transaction type: Income")
            return "Income"
        else:
            # Default to expense for receipts (most receipts are expenses)
            print("💵 Transaction type: Expense (default)")
            return "Expense"
    except Exception as e:
        print(f"Transaction type extraction failed: {e}")
        return "Expense"

def parse_receipt_text(text):
    """Main function to parse receipt text and extract structured data"""
    try:
        print("🔍 Starting receipt text parsing...")
        
        if not text or text.strip() == "":
            print("❌ No text to parse")
            return {
                "amount": 0.0,
                "date": None,
                "note": "No text extracted from receipt",
                "category": "Other",
                "type": "Expense",
                "success": False
            }
        
        amount = extract_amount(text)
        date = extract_date(text)
        note = extract_note(text)
        category = extract_category(text)
        transaction_type = extract_transaction_type(text)
        
        result = {
            "amount": amount,
            "date": date,
            "note": note,
            "category": category,
            "type": transaction_type,
            "success": amount > 0  # Consider successful if we found an amount
        }
        
        print(f"✅ Parsing completed: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Receipt parsing failed: {e}")
        return {
            "amount": 0.0,
            "date": None,
            "note": f"Error processing receipt: {str(e)}",
            "category": "Other",
            "type": "Expense",
            "success": False
        }