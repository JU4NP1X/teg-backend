# VoCo UC Backend Setup

This guide will walk you through the steps to set up and launch your Django backend.

## Prerequisites

- Python 3.10

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/JU4NP1X/teg-backend
   ```

2. Create a virtual environment and activate it:

   ```shell
   python3.10 -m venv myenv
   source myenv/bin/activate
   ```

3. Install the required packages:

   ```shell
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:

   - Ubuntu:

        ```shell
        sudo apt-get install tesseract-ocr
        ```

        Note: If you need additional language support, you can install the corresponding language packages. You must change the following command aplha3 to your language code:

        ```shell
        sudo apt-get install tesseract-ocr-{alpha3}
        ```

        For example:
        
        ```shell
        sudo apt-get install tesseract-ocr-spa
        ```
    - Other operating systems:

        Please refer to the official Tesseract documentation for specific instructions.

## Configuration

1. Set up the database:

   ```shell
   python manage.py migrate
   ```

2. Create a superuser (admin) account:

   ```shell
   python manage.py createsuperuser
   ```

   Follow the prompts to enter a username and password for the admin account.

## Launching the Server

To start the Django development server, run the following command:

```shell
python manage.py runserver
```

The server will start running at `http://localhost:8000/`.

## API Endpoints

- `/documents/`: GET - List all endpoints related to documents.
- `/documents/list/`: GET, POST - List all documents or create a new document.
- `/documents/list/<id>/`: GET, PUT, DELETE - Retrieve, update, or delete a specific document.
- `/documents/text-extractor/`: POST - Extract the title and summary from two images in base64.
  
  ---
- `/datasets/`: GET - List all endpoints related to datasets.
- `/datasets/list/`: GET, POST - List all datasets or create a new dataset.
- `/datasets/list/<id>/`: GET, PUT, DELETE - Retrieve, update, or delete a specific dataset.
- `/datasets/tranlations/`: GET, POST - List all translated datasets or create a new translated dataset.
- `/datasets/tranlations/<id>/`: GET, PUT, DELETE - Retrieve, update, or delete a specific translated dataset.
  
  ---
- `/categories/`: GET - List all endpoints related to categories.
- `/categories/list/`: GET, POST - List all category or create a new category.
- `/categories/list/<id>/`: GET, PUT, DELETE - Retrieve, update, or delete a specific category.
- `/categories/tranlations/`: GET, POST - List all translated categories or create a new translated category.
- `/categories/tranlations/<id>/`: GET, PUT, DELETE - Retrieve, update, or delete a specific translated category.

Refer to the DJango API for more details on the available endpoints and request/response formats.

## License

The following is the text of the license for the program developed as part of the Trabajo Especial de Grado (TEG) of the Universidad de Carabobo:

This program is licensed under the **MIT License**.


    MIT License

    Copyright (c) 2023 Juan Pablo Herrera

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.


---

Version 1.0, July 2023