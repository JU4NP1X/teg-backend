import PyPDF2

def extraer_abstract(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_paginas = len(pdf_reader.pages)
        
        for pagina_num in range(num_paginas):
            if pagina_num >= num_paginas:
                break
            
            pagina = pdf_reader.pages[pagina_num]
            contenido = pagina.extract_text()
            
            if 'abstract' in contenido.lower():
                abstract_split = contenido.split('abstract', 1)
                if len(abstract_split) > 1:
                    abstract = abstract_split[1].strip()
                    return abstract

    return None

pdf_path = '/home/juan/projects/teg/backend/Nwogu.pdf'
abstract = extraer_abstract(pdf_path)

if abstract:
    print("Abstract encontrado:")
    print(abstract)
else:
    print("No se encontró el abstract en el PDF.")