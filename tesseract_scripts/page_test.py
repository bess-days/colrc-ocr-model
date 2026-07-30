from pytesseract import image_to_string
from PIL import Image
from pdf2image import convert_from_path
import tempfile
def ocr_image(pdf_path, output_path, first_page=0, last_page=None):
    """
    Perform OCR on a pdf that is images of a book in Coeur d'Alene + English
    

    Args:
        pdf_path (str): Path to the input PDF file containing images of the book.
        output_path (str): Path to the output text file where the extracted text will be saved.
    """
    with tempfile.TemporaryDirectory() as path:
        # Convert PDF to images and save them in the temporary directory
        images = convert_from_path(pdf_path=pdf_path, output_folder=path, first_page=first_page, last_page=last_page)
    extracted_text = ""
    for i, image in enumerate(images):
        text = image_to_string(image, lang='rcd+eng', config='--tessdata-dir "./model"')
        # Doing CdA + English should in theory allow the English translation and CDA texts to be extracted, but in practice, it can't differentiate one line is CdA and one is English so occasionally merges it
        extracted_text += f"--- Page {i + 1} ---\n"
        extracted_text += text + "\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

ocr_image("./test/pages/CoyoteAndBadger_Typed.pdf", "./test/pages_output/cb.txt", 1, 3)
custom_config = r'--psm 6'
print(image_to_string(Image.open("./sample_page.png"), lang='rcd+eng', config=custom_config))