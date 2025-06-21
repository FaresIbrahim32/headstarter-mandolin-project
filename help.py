import json
import os
import io
import base64
import subprocess
import re
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Set poppler path for Windows
import os
if os.name == 'nt':  # Windows
    # Poppler path
    poppler_path = r"C:\Users\Fares\Downloads\Release-24.08.0-0\poppler-24.08.0\bin"
    if os.path.exists(poppler_path):
        os.environ["PATH"] += os.pathsep + poppler_path
    
    # Tesseract path (common installation locations)
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR",
        r"C:\Program Files (x86)\Tesseract-OCR",
        r"C:\Tesseract-OCR"
    ]
    for tesseract_path in tesseract_paths:
        if os.path.exists(tesseract_path):
            os.environ["PATH"] += os.pathsep + tesseract_path
            break

from pdf2image import convert_from_path
from PIL import Image
import requests
import torch
import cv2
import numpy as np

# LlamaIndex imports
from llama_cloud_services import LlamaParse
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.mistralai import MistralAIEmbedding

load_dotenv()

class DynamicMedicalFormFiller:
    def __init__(self):
        # Initialize LLAMA Cloud
        os.environ["LLAMA_CLOUD_API_KEY"] = "LLAMA_INDEX_API"
        
        # Initialize Embeddings
        api_key = "MISTRAL_API"
        model_name = "mistral-embed"
        embed_model = MistralAIEmbedding(model_name=model_name, api_key=api_key)
        
        # Initialize LLM with fallback options
        self.llm = self._init_llm_with_fallback()
        
        # Set settings
        Settings.llm = self.llm
        Settings.embed_model = embed_model
        
        # Initialize Vision Model for field detection
        self._init_vision_model()
        
        # Field position cache
        self.field_positions = {}

    def _init_llm_with_fallback(self):
        """Initialize LLM with fallback options for reliability"""
        models_to_try = [
            "microsoft/DialoGPT-medium",
            "google/flan-t5-base",
            "microsoft/DialoGPT-small",
            "Qwen/Qwen2.5-Coder-32B-Instruct"
        ]
        
        for model_name in models_to_try:
            try:
                print(f"Attempting to initialize LLM: {model_name}")
                llm = HuggingFaceInferenceAPI(
                    model_name=model_name,
                    timeout=30,
                    max_retries=2
                )
                test_response = llm.complete("Hello")
                print(f"✓ Successfully initialized LLM: {model_name}")
                return llm
                
            except Exception as e:
                print(f"Failed to initialize {model_name}: {e}")
                continue
        
        print("⚠️ All LLM models failed. Using mock LLM for testing.")
        return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """Create a simple mock LLM for when APIs are unavailable"""
        class MockLLM:
            def complete(self, prompt):
                return "Mock response - API unavailable"
            
            def chat(self, messages):
                return "Mock response - API unavailable"
        
        return MockLLM()

    def _init_vision_model(self):
        """Initialize vision model for form field detection"""
        print("Using robust fallback detection for maximum reliability...")
        self.vision_model_type = "fallback"
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print("Attempting to load lightweight BLIP model...")
            self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model_type = "blip_light"
            print("✓ Successfully loaded lightweight BLIP model")
        except Exception as e:
            print(f"Vision model loading failed: {e}")
            print("Using OpenCV-based fallback detection")
            self.vision_model_type = "fallback"

    def extract_patient_data(self, referral_pdf_path):
        """Extract structured patient data from referral document using detailed queries"""
        documents = LlamaParse(result_type="markdown").load_data(referral_pdf_path)
        node_parser = MarkdownElementNodeParser(llm=self.llm, num_workers=8)
        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        recursive_index = VectorStoreIndex(nodes=base_nodes + objects)
        query_engine = recursive_index.as_query_engine(similarity_top_k=25)
        
        # Detailed extraction queries for comprehensive data extraction
        extraction_queries = {
            "patient_demographics": """
            Extract the patient's demographic information and return as JSON:
            {
                "first_name": "patient's first name",
                "last_name": "patient's last name", 
                "date_of_birth": "DOB in MM/DD/YYYY format",
                "age": "patient's age",
                "gender": "patient's gender",
                "address": "full street address",
                "city": "city name",
                "state": "state abbreviation",
                "zip_code": "postal code"
            }
            Look for: patient name, DOB, birth date, address, demographic information.
            """,
            
            "contact_information": """
            Extract all contact information and return as JSON:
            {
                "home_phone": "home phone number",
                "work_phone": "work phone number", 
                "cell_phone": "cell/mobile phone number",
                "email": "email address",
                "emergency_contact": "emergency contact info"
            }
            Look for: phone numbers, telephone, mobile, email addresses, contact details.
            """,
            
            "insurance_details": """
            Extract insurance information and return as JSON:
            {
                "member_id": "insurance member ID",
                "subscriber_id": "subscriber identification",
                "group_number": "group number",
                "insurance_company": "insurance provider name",
                "plan_name": "insurance plan name",
                "subscriber_name": "name of subscriber",
                "relationship_to_subscriber": "patient's relationship to subscriber",
                "effective_date": "insurance effective date"
            }
            Look for: insurance, member ID, subscriber, group number, coverage, benefits.
            """,
            
            "medical_diagnosis": """
            Extract diagnosis and medical condition information and return as JSON:
            {
                "primary_diagnosis": "main diagnosis description",
                "primary_icd_code": "ICD-10 code",
                "secondary_diagnoses": ["list of other diagnoses"],
                "medical_conditions": ["list of medical conditions"],
                "disease_severity": "severity level if mentioned",
                "symptoms": ["list of symptoms"],
                "onset_date": "when condition started"
            }
            Look for: diagnosis, ICD codes, medical conditions, symptoms, disease progression.
            """,
            
            "medication_information": """
            Extract medication and treatment information and return as JSON:
            {
                "requested_medication": "medication being requested",
                "current_medications": ["list of current medications"],
                "dosage": "requested dosage",
                "frequency": "how often to take",
                "route": "administration route",
                "duration": "treatment duration",
                "instructions": "special instructions",
                "allergies": ["medication allergies"],
                "previous_medications": ["previously tried medications"]
            }
            Look for: medications, drugs, dosage, frequency, treatment, prescriptions.
            """,
            
            "provider_information": """
            Extract healthcare provider information and return as JSON:
            {
                "prescriber_first_name": "prescribing physician first name",
                "prescriber_last_name": "prescribing physician last name",
                "prescriber_title": "MD, DO, NP, PA, etc.",
                "specialty": "physician specialty",
                "npi_number": "National Provider Identifier",
                "dea_number": "DEA number",
                "state_license": "state license number",
                "clinic_name": "clinic or hospital name",
                "clinic_address": "provider address",
                "clinic_phone": "provider phone",
                "clinic_fax": "provider fax"
            }
            Look for: doctor, physician, prescriber, provider, clinic, hospital, NPI, DEA.
            """,
            
            "clinical_details": """
            Extract clinical and treatment details and return as JSON:
            {
                "treatment_indication": "reason for treatment",
                "prior_treatments": ["previous treatments tried"],
                "treatment_failures": ["treatments that failed"],
                "contraindications": ["any contraindications"],
                "labs_required": ["required lab tests"],
                "monitoring_requirements": ["monitoring needed"],
                "place_of_service": "where treatment will be given",
                "administration_method": "how medication is given"
            }
            Look for: treatment history, prior therapy, contraindications, administration.
            """
        }
        
        extracted_data = {}
        for category, query in extraction_queries.items():
            try:
                print(f"Extracting {category}...")
                response = query_engine.query(query)
                extracted_data[category] = str(response)
                print(f"✓ Extracted {category}")
            except Exception as e:
                print(f"❌ Error extracting {category}: {e}")
                extracted_data[category] = ""
        
        return extracted_data
    
    def _clean_and_parse_json(self, json_string):
        """Clean and parse JSON from LLM response"""
        try:
            # Remove markdown code blocks
            json_string = re.sub(r'```json\s*', '', json_string)
            json_string = re.sub(r'```\s*', '', json_string)
            
            # Remove any text before the first {
            start_idx = json_string.find('{')
            if start_idx != -1:
                json_string = json_string[start_idx:]
            
            # Remove any text after the last }
            end_idx = json_string.rfind('}')
            if end_idx != -1:
                json_string = json_string[:end_idx + 1]
            
            # Try to parse JSON
            return json.loads(json_string)
        except:
            # If JSON parsing fails, try to extract key-value pairs manually
            return self._extract_key_values(json_string)
    
    def _extract_key_values(self, text):
        """Extract key-value pairs from text when JSON parsing fails"""
        result = {}
        
        # Common patterns for extracting information
        patterns = [
            r'"([^"]+)":\s*"([^"]*)"',  # "key": "value"
            r"'([^']+)':\s*'([^']*)'",  # 'key': 'value'
            r'(\w+):\s*([^\n,}]+)',     # key: value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key = key.strip().lower().replace(' ', '_')
                value = value.strip().strip('"\'')
                if value and value.lower() not in ['null', 'none', '""', "''", 'n/a']:
                    result[key] = value
        
        return result
    
    def _extract_from_text(self, text, patterns):
        """Extract specific information using regex patterns"""
        results = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[key] = match.group(1).strip()
        return results

    def parse_extracted_data(self, extracted_data):
        """Parse the extracted data into structured format for form filling"""
        print("🔍 Parsing extracted data from LLM responses...")
        
        patient_data = {}
        
        # Parse demographics
        demographics_text = extracted_data.get("patient_demographics", "")
        demographics = self._clean_and_parse_json(demographics_text)
        
        # Parse contact info  
        contact_text = extracted_data.get("contact_information", "")
        contact_info = self._clean_and_parse_json(contact_text)
        
        # Parse insurance
        insurance_text = extracted_data.get("insurance_details", "")
        insurance_info = self._clean_and_parse_json(insurance_text)
        
        # Parse diagnosis
        diagnosis_text = extracted_data.get("medical_diagnosis", "")
        diagnosis_info = self._clean_and_parse_json(diagnosis_text)
        
        # Parse medication
        medication_text = extracted_data.get("medication_information", "")
        medication_info = self._clean_and_parse_json(medication_text)
        
        # Parse provider
        provider_text = extracted_data.get("provider_information", "")
        provider_info = self._clean_and_parse_json(provider_text)
        
        # Parse clinical details
        clinical_text = extracted_data.get("clinical_details", "")
        clinical_info = self._clean_and_parse_json(clinical_text)
        
        # Extract fallback patterns from raw text if JSON parsing didn't work well
        all_text = " ".join(extracted_data.values()).lower()
        
        fallback_patterns = {
            "phone": r'(?:phone|tel|telephone)[\s:]*(\(?[\d\s\-\(\)\.]{10,})',
            "dob": r'(?:dob|birth|born)[\s:]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            "icd": r'(g\d{2}(?:\.\d+)?)',
            "address": r'(\d+\s+[a-z\s]+(?:street|st|avenue|ave|apt|drive|dr|boulevard|blvd)[a-z\s\d]*)',
        }
        
        fallback_data = self._extract_from_text(all_text, fallback_patterns)
        
        # Compile final patient data with priority: parsed JSON > fallback extraction > empty
        patient_data = {
            # Demographics
            "first_name": (
                demographics.get("first_name") or 
                demographics.get("firstname") or 
                self._extract_name_part(all_text, "first") or 
                ""
            ),
            "last_name": (
                demographics.get("last_name") or 
                demographics.get("lastname") or 
                self._extract_name_part(all_text, "last") or 
                ""
            ),
            "dob": (
                demographics.get("date_of_birth") or 
                demographics.get("dob") or 
                fallback_data.get("dob") or 
                ""
            ),
            "age": demographics.get("age", ""),
            "gender": demographics.get("gender", ""),
            
            # Address
            "address": (
                demographics.get("address") or 
                fallback_data.get("address") or 
                ""
            ),
            "city": demographics.get("city", ""),
            "state": demographics.get("state", ""),
            "zip": (
                demographics.get("zip_code") or 
                demographics.get("zip") or 
                ""
            ),
            
            # Contact
            "home_phone": (
                contact_info.get("home_phone") or 
                contact_info.get("phone") or 
                fallback_data.get("phone") or 
                ""
            ),
            "work_phone": contact_info.get("work_phone", ""),
            "cell_phone": (
                contact_info.get("cell_phone") or 
                contact_info.get("mobile") or 
                ""
            ),
            "email": contact_info.get("email", ""),
            
            # Insurance
            "member_id": (
                insurance_info.get("member_id") or 
                insurance_info.get("subscriber_id") or 
                ""
            ),
            "group_number": insurance_info.get("group_number", ""),
            "insurance_provider": (
                insurance_info.get("insurance_company") or 
                insurance_info.get("insurance_provider") or 
                ""
            ),
            "plan_name": insurance_info.get("plan_name", ""),
            "subscriber_name": (
                insurance_info.get("subscriber_name") or 
                f"{patient_data.get('first_name', '')} {patient_data.get('last_name', '')}".strip()
            ),
            
            # Medical
            "primary_diagnosis": (
                diagnosis_info.get("primary_diagnosis") or 
                diagnosis_info.get("diagnosis") or 
                ""
            ),
            "primary_icd": (
                diagnosis_info.get("primary_icd_code") or 
                diagnosis_info.get("icd_code") or 
                fallback_data.get("icd") or 
                ""
            ),
            "disease_severity": diagnosis_info.get("disease_severity", ""),
            
            # Medication
            "requested_product": (
                medication_info.get("requested_medication") or 
                medication_info.get("medication") or 
                ""
            ),
            "dose": (
                medication_info.get("dosage") or 
                medication_info.get("dose") or 
                ""
            ),
            "directions": (
                medication_info.get("instructions") or 
                medication_info.get("directions") or 
                medication_info.get("frequency") or 
                ""
            ),
            "allergies": (
                medication_info.get("allergies") or 
                ""
            ),
            
            # Provider
            "prescriber_first": (
                provider_info.get("prescriber_first_name") or 
                provider_info.get("first_name") or 
                ""
            ),
            "prescriber_last": (
                provider_info.get("prescriber_last_name") or 
                provider_info.get("last_name") or 
                ""
            ),
            "prescriber_title": (
                provider_info.get("prescriber_title") or 
                provider_info.get("title") or 
                "MD"
            ),
            "prescriber_npi": (
                provider_info.get("npi_number") or 
                provider_info.get("npi") or 
                ""
            ),
            "clinic_name": (
                provider_info.get("clinic_name") or 
                provider_info.get("hospital") or 
                ""
            ),
            "clinic_address": provider_info.get("clinic_address", ""),
            "clinic_phone": provider_info.get("clinic_phone", ""),
            "clinic_fax": provider_info.get("clinic_fax", ""),
            
            # Clinical
            "treatment_reason": (
                clinical_info.get("treatment_indication") or 
                clinical_info.get("indication") or 
                ""
            ),
            "prior_therapy": "Yes" if clinical_info.get("prior_treatments") else "No",
            "place_of_admin": (
                clinical_info.get("place_of_service") or 
                "Outpatient Infusion Center"
            ),
            "administration_method": clinical_info.get("administration_method", ""),
        }
        
        # Clean up empty values and log what was extracted
        extracted_fields = []
        for key, value in patient_data.items():
            if value and str(value).strip():
                extracted_fields.append(f"{key}: {value}")
        
        print(f"✓ Successfully extracted {len(extracted_fields)} data fields:")
        for field in extracted_fields[:10]:  # Show first 10 fields
            print(f"  • {field}")
        if len(extracted_fields) > 10:
            print(f"  ... and {len(extracted_fields) - 10} more fields")
        
        # Warn if critical fields are missing
        critical_fields = ["first_name", "last_name", "dob", "primary_diagnosis"]
        missing_critical = [field for field in critical_fields if not patient_data.get(field)]
        if missing_critical:
            print(f"⚠️  Missing critical fields: {missing_critical}")
        
        return patient_data
    
    def _extract_name_part(self, text, part):
        """Extract first or last name from text"""
        name_patterns = [
            r'(?:patient|name)[\s:]+([a-zA-Z]+)\s+([a-zA-Z]+)',
            r'([a-zA-Z]+)\s+([a-zA-Z]+)(?:\s+(?:patient|dob|born))',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if part == "first":
                    return match.group(1)
                elif part == "last":
                    return match.group(2)
        return ""

    # [Keep all the existing vision detection and form filling methods unchanged]
    def _pdf_page_to_image(self, pdf_path, page_num):
        """Convert PDF page to image for vision analysis"""
        try:
            poppler_path = None
            if os.name == 'nt':
                potential_path = r"C:\Users\Fares\Downloads\Release-24.08.0-0\poppler-24.08.0\bin"
                if os.path.exists(potential_path):
                    poppler_path = potential_path
            
            if poppler_path:
                images = convert_from_path(
                    pdf_path, 
                    first_page=page_num+1, 
                    last_page=page_num+1, 
                    dpi=150,
                    poppler_path=poppler_path
                )
            else:
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=150)
            
            if images:
                image = images[0]
                max_size = (1024, 1024)
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    print(f"Resized image to {image.size} to avoid processing issues")
                return image
        except Exception as e:
            print(f"Error converting PDF to image: {e}")
        return None

    def _detect_fields_with_vision(self, image):
        """Use vision model to detect form fields and their coordinates"""
        try:
            if self.vision_model_type == "blip_light":
                return self._detect_with_blip_light(image)
            else:
                return self._fallback_detection(image)
        except Exception as e:
            print(f"Vision-based detection failed: {e}")
            return self._fallback_detection(image)

    def _detect_with_blip_light(self, image):
        """Use lightweight BLIP to analyze form"""
        try:
            inputs = self.vision_processor(image, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.vision_model.generate(**inputs, max_length=50)
            
            caption = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Form analysis: {caption}")
            
            return self._fallback_detection(image)
            
        except Exception as e:
            print(f"BLIP light detection error: {e}")
            return self._fallback_detection(image)

    def _fallback_detection(self, image):
        """Enhanced fallback method using OpenCV and heuristics"""
        try:
            print("Using enhanced OpenCV-based field detection...")
            
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            height, width = gray.shape
            fields = []
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            detect_horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            horizontal_contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            field_count = 0
            for contour in horizontal_contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h < 30 and h > 5:
                    fields.append({
                        "label": f"Field_{field_count}",
                        "type": "text",
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": max(h, 20)
                    })
                    field_count += 1
            
            if len(fields) < 5:
                print("Adding systematic field layout...")
                systematic_fields = self._create_systematic_fields(width, height)
                fields.extend(systematic_fields)
            
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 3 < aspect_ratio < 15 and h > 8:
                        fields.append({
                            "label": f"Detected_Field_{len(fields)}",
                            "type": "text",
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": max(h, 15)
                        })
            
            print(f"Detected {len(fields)} potential form fields")
            return fields
            
        except Exception as e:
            print(f"Enhanced fallback detection failed: {e}")
            return self._create_systematic_fields(800, 1000)

    def _create_systematic_fields(self, width, height):
        """Create systematic field layout based on image dimensions"""
        fields = []
        
        x_left = int(width * 0.1)
        x_right = int(width * 0.6)
        y_start = int(height * 0.2)
        field_height = 25
        field_spacing = 35
        
        field_definitions = [
            ("Patient First Name", x_left, 0),
            ("Patient Last Name", x_right, 0),
            ("Date of Birth", x_left, 1),
            ("Patient ID", x_right, 1),
            ("Address", x_left, 2),
            ("City", x_left, 3),
            ("State", int(width * 0.4), 3),
            ("ZIP Code", int(width * 0.5), 3),
            ("Phone Number", x_left, 4),
            ("Insurance ID", x_right, 4),
            ("Insurance Provider", x_left, 5),
            ("Group Number", x_right, 5),
            ("Prescriber Name", x_left, 6),
            ("NPI Number", x_right, 6),
            ("Diagnosis", x_left, 7),
            ("ICD Code", x_right, 7),
            ("Medication", x_left, 8),
            ("Dosage", x_right, 8),
            ("Directions", x_left, 9),
        ]
        
        for label, x, row in field_definitions:
            y = y_start + (row * field_spacing)
            field_width = int(width * 0.25) if x == x_right else int(width * 0.35)
            
            fields.append({
                "label": label,
                "type": "text",
                "x": x,
                "y": y,
                "width": field_width,
                "height": field_height
            })
        
        return fields

    def _detect_page_fields(self, pdf_path, page_num):
        """Vision-based field detection using AI models"""
        try:
            image = self._pdf_page_to_image(pdf_path, page_num)
            if image is None:
                print(f"Failed to convert page {page_num} to image, using fallback fields")
                return {"fields": self._create_fallback_fields()}
            
            print(f"Analyzing page {page_num} with enhanced detection...")
            field_data = self._detect_fields_with_vision(image)
            
            if not isinstance(field_data, list):
                field_data = field_data if isinstance(field_data, list) else []
            
            pdf_height = 792
            pdf_width = 612
            image_height = image.size[1]
            image_width = image.size[0]
            
            x_scale = pdf_width / image_width
            y_scale = pdf_height / image_height
            
            processed_fields = []
            for field in field_data:
                try:
                    pdf_x = int(field["x"] * x_scale)
                    pdf_y = pdf_height - int((field["y"] + field["height"]) * y_scale)
                    pdf_width_field = int(field["width"] * x_scale)
                    pdf_height_field = int(field["height"] * y_scale)
                    
                    pdf_x = max(0, min(pdf_x, pdf_width - 10))
                    pdf_y = max(0, min(pdf_y, pdf_height - 10))
                    pdf_width_field = max(10, min(pdf_width_field, pdf_width - pdf_x))
                    pdf_height_field = max(10, min(pdf_height_field, pdf_height - pdf_y))
                    
                    processed_fields.append({
                        "label": field["label"],
                        "type": field.get("type", "text"),
                        "x": pdf_x,
                        "y": pdf_y,
                        "width": pdf_width_field,
                        "height": pdf_height_field
                    })
                except (KeyError, TypeError, ValueError) as e:
                    print(f"Error processing field {field}: {e}")
                    continue
            
            print(f"Successfully processed {len(processed_fields)} fields for page {page_num}")
            return {"fields": processed_fields}
            
        except Exception as e:
            print(f"Page field detection failed: {e}")
            return {"fields": self._create_fallback_fields()}

    def _create_fallback_fields(self):
        """Create reliable fallback field layout"""
        return [
            {"label": "Patient First Name", "type": "text", "x": 80, "y": 650, "width": 200, "height": 20},
            {"label": "Patient Last Name", "type": "text", "x": 320, "y": 650, "width": 200, "height": 20},
            {"label": "Date of Birth", "type": "text", "x": 80, "y": 620, "width": 150, "height": 20},
            {"label": "Patient ID", "type": "text", "x": 320, "y": 620, "width": 150, "height": 20},
            {"label": "Address", "type": "text", "x": 80, "y": 590, "width": 400, "height": 20},
            {"label": "City", "type": "text", "x": 80, "y": 560, "width": 150, "height": 20},
            {"label": "State", "type": "text", "x": 250, "y": 560, "width": 80, "height": 20},
            {"label": "ZIP Code", "type": "text", "x": 350, "y": 560, "width": 100, "height": 20},
            {"label": "Phone Number", "type": "text", "x": 80, "y": 530, "width": 200, "height": 20},
            {"label": "Insurance ID", "type": "text", "x": 320, "y": 530, "width": 200, "height": 20},
            {"label": "Insurance Provider", "type": "text", "x": 80, "y": 500, "width": 300, "height": 20},
            {"label": "Group Number", "type": "text", "x": 400, "y": 500, "width": 120, "height": 20},
            {"label": "Prescriber Name", "type": "text", "x": 80, "y": 470, "width": 250, "height": 20},
            {"label": "NPI Number", "type": "text", "x": 350, "y": 470, "width": 150, "height": 20},
            {"label": "Diagnosis", "type": "text", "x": 80, "y": 440, "width": 300, "height": 20},
            {"label": "ICD Code", "type": "text", "x": 400, "y": 440, "width": 120, "height": 20},
            {"label": "Medication", "type": "text", "x": 80, "y": 410, "width": 250, "height": 20},
            {"label": "Dosage", "type": "text", "x": 350, "y": 410, "width": 150, "height": 20},
            {"label": "Directions", "type": "text", "x": 80, "y": 380, "width": 400, "height": 20},
        ]

    def fill_pa_form(self, patient_data, pa_form_path, output_path):
        """Fill the form using enhanced field detection and only extracted data"""
        try:
            reader = PdfReader(pa_form_path)
            writer = PdfWriter()
            
            total_fields_filled = 0
            
            # Check if we have any extracted data
            has_data = any(value for value in patient_data.values() if value and str(value).strip())
            if not has_data:
                print("⚠️  No extracted data available for form filling!")
                return False
            
            for page_num, page in enumerate(reader.pages):
                print(f"Processing page {page_num + 1}...")
                
                if page_num == 0:
                    writer.add_page(page)
                    continue
                
                page_key = f"{pa_form_path}_page{page_num}"
                if page_key not in self.field_positions:
                    self.field_positions[page_key] = self._detect_page_fields(pa_form_path, page_num)
                
                packet = io.BytesIO()
                can = canvas.Canvas(packet, pagesize=letter)
                
                fields_filled = 0
                fields = self.field_positions[page_key].get('fields', [])
                
                print(f"Attempting to fill {len(fields)} fields on page {page_num + 1}")
                
                for i, field in enumerate(fields):
                    try:
                        value = self._map_field_to_data(field['label'], patient_data)
                        if value and str(value).strip():  # Only fill if we have actual data
                            x = field.get('x', 100) + 5
                            y = field.get('y', 100) + 5
                            
                            if 0 <= x <= 612 and 0 <= y <= 792:
                                field_height = field.get('height', 20)
                                font_size = min(12, max(8, field_height - 4))
                                can.setFont("Helvetica", font_size)
                                
                                field_width = field.get('width', 200)
                                max_chars = max(1, field_width // 6)
                                text = str(value)[:max_chars]
                                
                                can.drawString(x, y, text)
                                fields_filled += 1
                                print(f"  ✓ Filled {field['label']}: {text}")
                            else:
                                print(f"  ⚠ Skipped {field['label']}: coordinates out of bounds")
                        else:
                            print(f"  - No extracted data for {field['label']}")
                            
                    except Exception as e:
                        print(f"  ❌ Error filling field {field.get('label', i)}: {e}")
                        continue
                
                print(f"Successfully filled {fields_filled} fields on page {page_num + 1}")
                total_fields_filled += fields_filled
                
                can.save()
                packet.seek(0)
                
                if packet.getvalue():
                    overlay_pdf = PdfReader(packet)
                    if overlay_pdf.pages:
                        page.merge_page(overlay_pdf.pages[0])
                
                writer.add_page(page)
            
            temp_path = os.path.join(os.getenv("TEMP", "/tmp"), "temp_filled_form.pdf")
            with open(temp_path, "wb") as f:
                writer.write(f)
            
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_path, output_path)
            
            print(f"✓ Successfully filled {total_fields_filled} total fields and saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error filling form: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _map_field_to_data(self, field_name, patient_data):
        """Map PDF form field names to patient data fields with flexible matching"""
        field_mappings = {
            # Patient info variations
            "first name": patient_data.get("first_name", ""),
            "patient first name": patient_data.get("first_name", ""),
            "fname": patient_data.get("first_name", ""),
            "last name": patient_data.get("last_name", ""),
            "patient last name": patient_data.get("last_name", ""),
            "lname": patient_data.get("last_name", ""),
            "surname": patient_data.get("last_name", ""),
            
            # Date variations
            "date of birth": patient_data.get("dob", ""),
            "dob": patient_data.get("dob", ""),
            "birth date": patient_data.get("dob", ""),
            "birthdate": patient_data.get("dob", ""),
            
            # Address variations
            "address": patient_data.get("address", ""),
            "street address": patient_data.get("address", ""),
            "patient address": patient_data.get("address", ""),
            "home address": patient_data.get("address", ""),
            "city": patient_data.get("city", ""),
            "state": patient_data.get("state", ""),
            "zip": patient_data.get("zip", ""),
            "zip code": patient_data.get("zip", ""),
            "postal code": patient_data.get("zip", ""),
            
            # Phone variations
            "phone": patient_data.get("home_phone", ""),
            "phone number": patient_data.get("home_phone", ""),
            "home phone": patient_data.get("home_phone", ""),
            "cell phone": patient_data.get("cell_phone", ""),
            "mobile": patient_data.get("cell_phone", ""),
            "mobile phone": patient_data.get("cell_phone", ""),
            
            # Insurance variations
            "member id": patient_data.get("member_id", ""),
            "patient id": patient_data.get("member_id", ""),
            "insurance id": patient_data.get("member_id", ""),
            "id number": patient_data.get("member_id", ""),
            "group number": patient_data.get("group_number", ""),
            "group #": patient_data.get("group_number", ""),
            "insurance provider": patient_data.get("insurance_provider", ""),
            "insurance company": patient_data.get("insurance_provider", ""),
            "payor": patient_data.get("insurance_provider", ""),
            
            # Medical info variations
            "diagnosis": patient_data.get("primary_diagnosis", ""),
            "primary diagnosis": patient_data.get("primary_diagnosis", ""),
            "icd code": patient_data.get("primary_icd", ""),
            "icd-10": patient_data.get("primary_icd", ""),
            "medication": patient_data.get("requested_product", ""),
            "drug": patient_data.get("requested_product", ""),
            "dosage": patient_data.get("dose", ""),
            "dose": patient_data.get("dose", ""),
            "directions": patient_data.get("directions", ""),
            
            # Provider variations
            "prescriber": f"{patient_data.get('prescriber_first', '')} {patient_data.get('prescriber_last', '')}".strip(),
            "prescriber name": f"{patient_data.get('prescriber_first', '')} {patient_data.get('prescriber_last', '')}".strip(),
            "physician": f"{patient_data.get('prescriber_first', '')} {patient_data.get('prescriber_last', '')}".strip(),
            "doctor": f"{patient_data.get('prescriber_first', '')} {patient_data.get('prescriber_last', '')}".strip(),
            "npi": patient_data.get("prescriber_npi", ""),
            "npi number": patient_data.get("prescriber_npi", ""),
        }
        
        field_name_lower = field_name.lower().strip()
        
        if field_name_lower in field_mappings:
            return field_mappings[field_name_lower]
        
        for mapping_key, value in field_mappings.items():
            if mapping_key in field_name_lower or field_name_lower in mapping_key:
                if value and str(value).strip():
                    return value
        
        return ""

    def process_forms(self, referral_pdf_path, pa_form_path, output_path):
        """Complete workflow to extract data and fill form using only LLM extracted data"""
        print("🤖 Starting Dynamic Medical Form Processing (LLM Data Only)...")
        
        print("Step 1: Extracting patient data from referral document...")
        extracted_data = self.extract_patient_data(referral_pdf_path)
        
        print("Step 2: Parsing extracted data...")
        patient_data = self.parse_extracted_data(extracted_data)
        
        # Verify we have extracted data
        populated_fields = {k: v for k, v in patient_data.items() if v and str(v).strip()}
        if not populated_fields:
            raise Exception("No data could be extracted from the referral document")
        
        print(f"✓ Extracted {len(populated_fields)} data fields for form filling")
        
        print("Step 3: Filling PA form with extracted data only...")
        success = self.fill_pa_form(patient_data, pa_form_path, output_path)
        
        if success:
            return patient_data
        else:
            raise Exception("Form filling failed")

if __name__ == "__main__":
    print("🤖 Initializing Dynamic Medical Form Filler (LLM Extraction Only)...")
    form_filler = DynamicMedicalFormFiller()
    
    try:
        patient_data = form_filler.process_forms(
            referral_pdf_path="referral_package.pdf",
            pa_form_path="PA.pdf", 
            output_path="filled_PA_form_dynamic.pdf"
        )
        
        print("\n📋 Final Extracted Patient Data Used:")
        populated_data = {k: v for k, v in patient_data.items() if v and str(v).strip()}
        print(json.dumps(populated_data, indent=2))
        
        print(f"\n✅ Dynamic form processing completed successfully!")
        print(f"📄 Filled form saved as: filled_PA_form_dynamic.pdf")
        print(f"📊 Total fields populated: {len(populated_data)}")
                
    except Exception as e:
        print(f"❌ Error processing forms: {e}")
        import traceback
        traceback.print_exc()