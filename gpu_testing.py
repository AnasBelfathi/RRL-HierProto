# test_idr_torch.py
import pandas as pd
try:
    import idr_torch
    print("idr_torch imported")
    import torch.distributed as dist
    print("Module idr_torch imported successfully.")

    # Vérification des informations disponibles
    print(f"idr_torch.rank: {idr_torch.rank}")
    print(f"idr_torch.local_rank: {idr_torch.local_rank}")
    print(f"idr_torch.size: {idr_torch.size}")
    print(f"idr_torch.cpus_per_task: {idr_torch.cpus_per_task}")

    # Initialisation du processus de distribution avec NCCL (recommandé pour Jean Zay)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=idr_torch.size,
        rank=idr_torch.rank
    )

    # Afficher le statut de l'initialisation
    if dist.is_initialized():
        print("Distributed process group initialized successfully.")
    else:
        print("Failed to initialize the distributed process group.")

    # Vérification du GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(idr_torch.local_rank)
        print(f"Using GPU with local rank: {idr_torch.local_rank}")
    else:
        print("No GPU detected. Using CPU.")

except ImportError:
    print("Failed to import idr_torch. Ensure that it is installed and available on this system.")
except Exception as e:
    print(f"An error occurred: {e}")
