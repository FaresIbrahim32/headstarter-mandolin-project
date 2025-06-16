from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal

for page_layout in extract_pages("PA.pdf"):
    for element in page_layout:
        if isinstance(element, LTTextBoxHorizontal):
            print(element.get_text())