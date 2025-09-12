# gpu_utils.py
import os, torch, gc
from typing import Iterable, List, Optional, Union

# ---------- helpers you started (polished) ----------
def env_preferred_device() -> Optional[int]:
    """Prefer rank hints if you're spawning one process per GPU."""
    for var in ("LOCAL_RANK", "CUDA_DEVICE", "CUDA_IDX"):
        v = os.environ.get(var)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return None

def device_with_most_free() -> int:
    """Return best device index in the *current* CUDA visible set, or -1 for CPU."""
    if not torch.cuda.is_available():
        return -1
    best = (-1, -1)  # (idx, free_bytes)
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            free, _total = torch.cuda.mem_get_info()
        except Exception:
            free = -1
        if free > best[1]:
            best = (i, free)
    return best[0]

def select_device() -> int:
    """Pick device to use: env rank if present, else the most-free GPU, else CPU(-1)."""
    env_idx = env_preferred_device()
    if env_idx is not None and 0 <= env_idx < torch.cuda.device_count():
        return env_idx
    idx = device_with_most_free()
    return idx if idx >= 0 else -1

def providers_for(device_id: int):
    """ONNX Runtime provider config for a specific device."""
    if device_id < 0:
        return ["CPUExecutionProvider"]
    return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]

def free_cuda():
    """Try hard to return VRAM to the system."""
    if torch.cuda.is_available():
        try: torch.cuda.synchronize()
        except Exception: pass
        torch.cuda.empty_cache()
        try: torch.cuda.ipc_collect()
        except Exception: pass
    gc.collect()

# ---------- Diffusers-aware move/offload helpers ----------
def _to_device(pipe, device: Union[str, torch.device]):
    """
    Move a Diffusers pipeline and its known heavy submodules to device.
    Also keeps Diffusers' internal execution device consistent.
    """
    pipe.to(device)
    # Some pipelines carry extra modules not tracked by .to()
    for name in ("unet", "vae", "text_encoder", "text_encoder_2", "transformer"):
        if hasattr(pipe, name) and getattr(pipe, name) is not None:
            try:
                getattr(pipe, name).to(device)
            except Exception:
                pass
    if hasattr(pipe, "controlnet") and getattr(pipe, "controlnet") is not None:
        cn = getattr(pipe, "controlnet")
        if isinstance(cn, (list, tuple)):
            for c in cn:
                try: c.to(device)
                except Exception: pass
        else:
            try: cn.to(device)
            except Exception: pass
    # Keep execution device in sync so schedulers/prepare_latents pick the right device
    try:
        pipe._execution_device = torch.device(device) if isinstance(device, str) else device
    except Exception:
        pass

def _offload_to_cpu(pipe):
    """Offload a pipeline to CPU."""
    _to_device(pipe, "cpu")

# ---------- ModelManager ----------
class ModelManager:
    """
    Tracks which model group is 'live' on GPU and frees VRAM when switching.
    - device_policy: 'env-rank' (default), 'max-free', 'fixed:<idx>'
    - switch_policy: 'offload' (default) or 'destroy'  (destroy only drops our refs; you still own globals)
    - min_free_gb: if free VRAM on target device is below threshold, we free current group first
    """
    def __init__(self,
                 device_policy: str = None,
                 switch_policy: str = None,
                 min_free_gb: float = None):
        self.device_policy = device_policy or os.getenv("DEVICE_POLICY", "env-rank")
        self.switch_policy = switch_policy or os.getenv("MODEL_SWITCH_POLICY", "offload")
        self.min_free_gb = float(min_free_gb if min_free_gb is not None else os.getenv("MIN_FREE_GB", "2"))
        self.current_group: Optional[str] = None
        self.current_device: Optional[int] = None
        self.registry: dict[str, List[object]] = {}   # group -> list[pipelines]

    # -------- device selection --------
    def _pick_device_index(self, explicit: Optional[int] = None) -> Optional[int]:
        if not torch.cuda.is_available():
            return None
        if explicit is not None:
            return explicit
        if self.device_policy.startswith("fixed:"):
            try:
                idx = int(self.device_policy.split(":", 1)[1])
                return idx
            except Exception:
                pass
        if self.device_policy == "env-rank":
            idx = env_preferred_device()
            if idx is not None:
                return idx
        # default: most free
        return device_with_most_free()

    def _device_string(self, idx: Optional[int]) -> str:
        return "cpu" if (idx is None or idx < 0) else f"cuda:{idx}"

    def _free_memory_if_needed(self, target_idx: Optional[int]):
        """If target GPU looks tight, free current group first."""
        if target_idx is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.set_device(target_idx)
            free_b, _ = torch.cuda.mem_get_info()
            free_gb = free_b / (1024**3)
        except Exception:
            free_gb = 0.0

        if free_gb < self.min_free_gb and self.current_group is not None:
            # Free current VRAM first
            self._offload_group(self.current_group)
            free_cuda()

    # -------- public switching API --------
    def before_switch(self, target_group: str, target_device_index: Optional[int] = None) -> str:
        """
        Call before initializing/using a new group.
        Returns the device string to use ('cuda:N' or 'cpu').
        """
        idx = self._pick_device_index(target_device_index)
        self._free_memory_if_needed(idx)
        # If we're switching between groups, optionally free current right now
        if self.current_group and self.current_group != target_group:
            if self.switch_policy == "offload":
                self._offload_group(self.current_group)
            elif self.switch_policy == "destroy":
                self._destroy_group(self.current_group)
            free_cuda()
            self.current_group = None
            self.current_device = None
        return self._device_string(idx)

    def after_switch(self, group: str, objects: Iterable[object], device_index: Optional[int] = None, recompile_unet: bool = False):
        """
        Register objects under a group and move them to the chosen device.
        """
        objs = [o for o in objects if o is not None]
        self.registry[group] = objs
        idx = self._pick_device_index(device_index)
        dev_str = self._device_string(idx)
        # Move the new group to device
        for p in objs:
            _to_device(p, dev_str)
            # optional: recompile unet if you rely on torch.compile across device changes
            if recompile_unet and hasattr(p, "unet") and p.unet is not None:
                try:
                    p.unet = torch.compile(p.unet, mode="reduce-overhead")
                except Exception:
                    pass
        self.current_group = group
        self.current_device = idx

    def move_current_to(self, device_index: Optional[int]):
        """Explicitly move the current group to a specific device index."""
        if self.current_group is None:
            return
        idx = self._pick_device_index(device_index)
        dev_str = self._device_string(idx)
        for p in self.registry.get(self.current_group, []):
            _to_device(p, dev_str)
        self.current_device = idx
        free_cuda()

    # -------- free/destroy internals --------
    def _offload_group(self, group: str):
        """Move all pipelines of 'group' to CPU."""
        for p in self.registry.get(group, []):
            _offload_to_cpu(p)

    def _destroy_group(self, group: str):
        """Drop our references to pipelines in 'group' (your code still owns globals)."""
        self.registry.pop(group, None)

    # -------- status (for logging/debug) --------
    def status(self) -> dict:
        return {
            "current_group": self.current_group,
            "current_device": self.current_device,
            "device_policy": self.device_policy,
            "switch_policy": self.switch_policy,
        }
