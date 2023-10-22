from django.core.management.base import BaseCommand
import PyPDF2
import re
from googletrans import Translator as GoogleTranslator
from categories.models import (
    Categories,
    Translations,
    Authorities,
    update_categories_tree,
)
from tqdm import tqdm

translator = GoogleTranslator()


class Command(BaseCommand):
    help = "Scrape PDF and extract bold words"

    def add_arguments(self, parser):
        parser.add_argument("pdf_path", type=str, help="Path to the PDF file")

    def handle(self, *args, **options):
        pdf_path = options["pdf_path"]

        authority, _ = Authorities.objects.update_or_create(
            name="IEEE",
        )

        categories = Categories.objects.filter(name__startswith="(")
        for category in categories:
            category.deprecated = True
            category.save()

        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            progress_bar = tqdm(total=num_pages, desc="Processing Pages")
            page_num = 0
            for page in pdf_reader.pages:
                page_num += 1
                content = page.extract_text()
                sentences = re.split(r"\s{2}|[\n]", content)
                if page_num > 150:
                    for index, text in enumerate(sentences):
                        if text.strip().startswith("BT:"):
                            parent = text.strip()[3:].strip()
                            i = index - 1
                            while i >= 0:
                                if (
                                    ":" not in sentences[i]
                                    and sentences[i].strip() != ""
                                    and not sentences[i].strip().startswith("(")
                                ):
                                    child = sentences[i].strip()
                                    break
                                i -= 1
                            child, created = Categories.objects.update_or_create(
                                name=child,
                                authority=authority,
                            )
                            if created:
                                translation = translator.translate(
                                    child.name, dest="es"
                                ).text
                                Translations.objects.update_or_create(
                                    name=translation, category=child, language="es"
                                )
                            parent, created = Categories.objects.update_or_create(
                                name=parent,
                                authority=authority,
                            )
                            if created:
                                translation = translator.translate(
                                    parent.name, dest="es"
                                ).text
                                Translations.objects.update_or_create(
                                    name=translation, category=parent, language="es"
                                )
                            update_categories_tree(child, parent)
                progress_bar.update(1)
            progress_bar.close()
