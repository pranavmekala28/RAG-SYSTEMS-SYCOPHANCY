from markitdown import MarkItDown
from pathlib import Path

pdf_folder = Path(".")         # looks in current folder
md_folder  = Path("docs_md")  # saves .md files here
md_folder.mkdir(exist_ok=True)

md = MarkItDown()

pdfs = list(pdf_folder.glob("**/*.pdf"))

if not pdfs:
    print("No PDFs found in this folder. Drop some PDFs in rag-eval and run again.")
else:
    for pdf in pdfs:
        result = md.convert(str(pdf))
        out = md_folder / (pdf.stem + ".md")
        out.write_text(result.text_content, encoding="utf-8")
        print(f"✓ {pdf.name} → {out.name}")
    print(f"\nDone. {len(pdfs)} file(s) converted → docs_md/")