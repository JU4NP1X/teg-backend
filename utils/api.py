import psutil
import platform
import torch
from gpuinfo import GPUInfo
from rest_framework import viewsets
from rest_framework.response import Response


class SystemInfoViewSet(viewsets.ViewSet):
    """
    ViewSet for system information.
    """

    def list(self, request):
        """
        Get system information.

        Returns:
            Response: The response object containing the system information.
        """
        # Check if GPU is available
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            gpu_name = gpu_info.name
            gpu_memory_total = gpu_info.total_memory
            gpu_percent, gpu_memory_used = GPUInfo.gpu_usage()
        else:
            gpu_name = "N/A"
            gpu_memory_total = 0
            gpu_memory_used = [0]
            gpu_percent = [0]

        # Get RAM information
        ram = psutil.virtual_memory()
        ram_total = ram.total
        ram_percent = ram.percent

        # Get CPU information
        cpu_percent = psutil.cpu_percent()
        cpu_name = platform.processor()

        # Prepare system information response
        system_info = {
            "platform": platform.system(),
            "ram": {"total": ram_total, "percent": ram_percent},
            "cpu": {"name": cpu_name, "percent": cpu_percent},
            "gpu": {
                "name": gpu_name,
                "memory_total": gpu_memory_total,
                "memory_used": gpu_memory_used[0] * 2**20,
                "percent": gpu_percent[0],
            },
        }

        return Response(system_info)
