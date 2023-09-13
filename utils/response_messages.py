from rest_framework.response import Response
from rest_framework import status


RESPONSE_MESSAGES = {
    "CIRCULAR_RELATIONSHIP": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Circular relationship detected in the CSV data.",
    },
    "DUPLICATE_NAME": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Duplicate name found in the CSV data.",
    },
    "INVALID_RELATIONSHIP": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Invalid relationship found in the CSV data.",
    },
    "ELEMENT_NOT_FOUND": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Element not found in the CSV data.",
    },
    "CSV_CREATION_FAILED": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Failed to create the CSV with categories.",
    },
    "ERROR_TRAINING": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Failed training the authority.",
    },
    "TEXT_CLASSIFICATION_FAILED": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Failed to classify the text.",
    },
    "TRAINING_MODEL_NOT_EXIST": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "The trained model doesn't exist.",
    },
    "CANNOT_DELETE_NATIVE_AUTHORITY": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Cannot delete a native authority.",
    },
    "CANNOT_MODIFY_NATIVE_AUTHORITY_NAME": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Cannot modify the name of a native authority.",
    },
    "INVALID_CSV_FORMAT": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Invalid CSV format. Unable to decode base64 or parse CSV.",
    },
    "AUTHORITY_ALREADY_EXISTS": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "The authority already exists.",
    },
    "FORBIDDEN_ACTION": {
        "code": status.HTTP_403_FORBIDDEN,
        "message": "Only administrators can execute this action.",
    },
    "ACTION_INITIATED_SUCCESSFULLY": {
        "code": status.HTTP_200_OK,
        "message": "Action initiated successfully.",
    },
    "INVALID_IMAGE_FORMAT": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Invalid image format. Unable to decode base64 or open image.",
    },
    "TEXT_EXTRACTION_SUCCESS": {
        "code": status.HTTP_200_OK,
        "message": "Text extraction successful.",
    },
    "INVALID_CREDENTIALS": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Invalid credentials.",
    },
    "INVALID_TOKEN": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Invalid token.",
    },
    "OK": {
        "code": status.HTTP_200_OK,
        "message": "Success.",
    },
}
