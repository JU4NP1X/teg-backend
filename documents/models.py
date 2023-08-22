import base64
from django.db import models
from django.core.files.base import ContentFile
from users.models import User


class Documents(models.Model):
    """
    Model for representing documents.

    Attributes:
        title (CharField): The title of the document.
        summary (TextField): The summary of the document.
        authors (CharField): The authors of the document.
        pdf (BinaryField): The binary content of the PDF document.
        created_by (ForeignKey): The user who created the document.
        updated_by (ForeignKey): The user who last updated the document.
        created_at (DateTimeField): The datetime when the document was created.
        updated_at (DateTimeField): The datetime when the document was last updated.
    """

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
        """
        Return a string representation of the document.

        Returns:
            str: The title of the document.
        """
        return str(self.title)

    def save(self, *args, **kwargs):
        """
        Save the document.

        Converts the PDF field to binary before saving.

        Args:
            *args: Variable length argument list.
            **kwargs: Keyword arguments.
        """
        self.convert_pdf_to_binary()
        super().save(*args, **kwargs)

    def convert_pdf_to_binary(self):
        """
        Convert the base64 encoded PDF to binary.

        Raises:
            ValueError: If there is an error converting the PDF field to binary.
        """
        try:
            # Check if the pdf field is already binary
            if not isinstance(self.pdf, bytes):
                # Decode the base64 encoded PDF to bytes
                pdf_bytes = base64.b64decode(self.pdf)

                # Create a ContentFile object from the PDF bytes
                content_file = ContentFile(pdf_bytes)

                # Assign the binary content to the pdf field
                self.pdf = content_file.read()

        except Exception as error:
            # Handle any conversion errors
            raise ValueError(
                "Error converting PDF field to binary: {}".format(str(error))
            )
