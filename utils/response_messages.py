from rest_framework.response import Response
from rest_framework import status

RESPONSE_MESSAGES = {
    "CIRCULAR_RELATIONSHIP": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Se detectó una relación circular en los datos del CSV",
    },
    "DUPLICATE_NAME": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Se encontró un nombre duplicado en los datos del CSV.",
    },
    "INVALID_RELATIONSHIP": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Se encontró una relación inválida en los datos del CSV.",
    },
    "ELEMENT_NOT_FOUND": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Elemento no encontrado en los datos del CSV.",
    },
    "CSV_CREATION_FAILED": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Error al crear el CSV con las categorías.",
    },
    "ERROR_TRAINING": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Error al entrenar la autoridad.",
    },
    "TEXT_CLASSIFICATION_FAILED": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Error al clasificar el texto.",
    },
    "TRAINING_MODEL_NOT_EXIST": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "El modelo entrenado no existe.",
    },
    "CANNOT_DELETE_NATIVE_AUTHORITY": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "No se puede eliminar una autoridad nativa.",
    },
    "CANNOT_MODIFY_NATIVE_AUTHORITY_NAME": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "No se puede modificar el nombre de una autoridad nativa.",
    },
    "INVALID_CSV_FORMAT": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Formato CSV inválido. No se puede decodificar base64 o analizar el CSV.",
    },
    "AUTHORITY_ALREADY_EXISTS": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "La autoridad ya existe.",
    },
    "FORBIDDEN_ACTION": {
        "code": status.HTTP_403_FORBIDDEN,
        "message": "Solo los administradores pueden ejecutar esta acción.",
    },
    "ACTION_INITIATED_SUCCESSFULLY": {
        "code": status.HTTP_200_OK,
        "message": "Acción iniciada exitosamente.",
    },
    "INVALID_IMAGE_FORMAT": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Formato de imagen inválido. No se puede decodificar base64 o abrir la imagen.",
    },
    "TEXT_EXTRACTION_SUCCESS": {
        "code": status.HTTP_200_OK,
        "message": "Extracción de texto exitosa.",
    },
    "LOGOUT": {
        "code": status.HTTP_200_OK,
        "message": "Cierre de sesión exitoso.",
    },
    "INVALID_CREDENTIALS": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Credenciales inválidas.",
    },
    "INVALID_TOKEN": {
        "code": status.HTTP_400_BAD_REQUEST,
        "message": "Token inválido.",
    },
    "OK": {
        "code": status.HTTP_200_OK,
        "message": "Éxito.",
    },
    "PREDICTOR_LOAD_SUCCESS": {
        "code": status.HTTP_200_OK,
        "message": "Predictor cargado correctamente",
    },
    "UNKNOW_ERROR": {
        "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "Error desconocido en el sistema",
    },
}
