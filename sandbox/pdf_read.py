from PyPDF2 import PdfReader, PdfWriter

reader = PdfReader("ar.pdf")
writer = PdfWriter()

# Extract pages 2 and 3 (zero-indexed: 1 and 2)
for i in [3, 4]:
    writer.add_page(reader.pages[i])

with open("extracted_pages.pdf", "wb") as f:
    writer.write(f)
