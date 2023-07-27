from django.db import models
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
import base64


class Documents(models.Model):
    title = models.CharField(max_length=255)
    summary = models.TextField()
    authors = models.CharField(max_length=255)
    pdf = models.BinaryField()
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="documents_created"
    )
    updated_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="documents_updated"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        self.convert_pdf_to_binary()
        super().save(*args, **kwargs)

    def convert_pdf_to_binary(self):
        try:
            # Check if the pdf field is already binary
            if not isinstance(self.pdf, bytes):
                # Decode the base64 encoded PDF to bytes
                pdf_bytes = base64.b64decode(self.pdf)

                # Create a ContentFile object from the PDF bytes
                content_file = ContentFile(pdf_bytes)

                # Assign the binary content to the pdf field
                self.pdf = content_file.read()

        except Exception as e:
            # Handle any conversion errors
            raise ValueError("Error converting PDF field to binary: {}".format(str(e)))
