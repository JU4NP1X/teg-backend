import os
import sys
import psutil
from django.core.wsgi import get_wsgi_application


# Aquí puedes agregar tu código de verificación del servicio
def verify_datasets_syncs():
    from categories.models import Authorities

    authorities = Authorities.objects.all()

    for authority in authorities:
        if authority.pid != 0 or (
            authority.pid == 0 and authority.status in ("GETTING_DATA", "TRAINING")
        ):
            try:
                process = psutil.Process(authority.pid)
                if process.name() != "python":
                    authority.pid = 0
                    authority.status = "COMPLETE"
                    authority.save()
            except psutil.NoSuchProcess:
                authority.pid = 0
                authority.status = "COMPLETE"
                authority.save()


def verify_authorities():
    from categories.models import Authorities

    authorities = Authorities.objects.filter(name="UNESCO")
    if not len(authorities):
        Authorities.objects.create(name="UNESCO")


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tu_proyecto.settings")

application = get_wsgi_application()

# Llama a la función de verificación del servicio después de que se inicie la aplicación
verify_datasets_syncs()
verify_authorities()
