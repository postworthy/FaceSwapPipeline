import gc, torch

def _move_pipe(pipe, device: str):
    """Move an SDXL(+ControlNet) pipeline and all heavy submodules to a device."""
    pipe.to(device)
    if hasattr(pipe, "controlnet") and pipe.controlnet is not None:
        cn = pipe.controlnet
        if isinstance(cn, (list, tuple)):
            for c in cn: c.to(device)
        else:
            cn.to(device)
    # keep Diffusers' execution device in sync (avoids surprise cuda:0 usage later)
    try:
        pipe._execution_device = torch.device(device)
    except Exception:
        pass
    return pipe

def _free_cuda(gather_gc: bool = True):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # helps free leftover IPC memory in some drivers
        try: torch.cuda.ipc_collect()
        except Exception: pass
    if gather_gc:
        gc.collect()
