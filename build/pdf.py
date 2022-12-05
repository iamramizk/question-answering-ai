import PyPDF2
import wordninja
import pdfplumber


# def clean_words(text: str) -> str:
#     output = ""
#     paras = text.split("\n")
#     for para in paras:
#         output += f"{' '.join(wordninja.split(para.replace(' ','')))}\n"
#     return output


# def text_from_pdf(file) -> str:
#     corpus = ""
#     pdfReader = PyPDF2.PdfFileReader(file)

#     page_count = pdfReader.numPages

#     for page_idx in range(page_count):
#         pageOj = pdfReader.getPage(page_idx)
#         corpus += pageOj.extractText() + "\n"

#     if corpus:
#         return clean_words(corpus)


def text_from_pdf(file) -> str:
    """Extracts text from all pages on a pdf"""
    corpus = ""
    with pdfplumber.open(file) as pdf:
        for page_idx in range(len(pdf.pages)):
            corpus += pdf.pages[page_idx].extract_text()
    return corpus
