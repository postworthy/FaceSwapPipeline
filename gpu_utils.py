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

    Robustness for Diffusers/ControlNet/SDXL:
    - Remembers the device selected in `before_switch` and uses the SAME device in `after_switch`.
    - Flattens tuples/lists and unwraps to pipeline-like objects automatically.
    - Forces ALL submodules of every registered pipeline onto one device and sets `_execution_device`.
    - Detects and heals accidental multi-device splits; optional strict failure via env `MM_STRICT_DEVICE=1`.
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
        self.strict_device = os.getenv("MM_STRICT_DEVICE", "0") == "1"
        # remember the device chosen in before_switch so after_switch won't pick a different one
        self._pending_device_index: Optional[int] = None

    # -------- device selection --------
    def _pick_device_index(self, explicit: Optional[int] = None) -> Optional[int]:
        if not torch.cuda.is_available():
            return None
        if explicit is not None:
            return explicit
        if self.device_policy.startswith("fixed:"):
            try:
                return int(self.device_policy.split(":", 1)[1])
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

    # -------- internal normalization/helpers (only used inside ModelManager) --------
    def _flatten(self, items: Iterable[object]) -> List[object]:
        out: List[object] = []
        for x in items:
            if x is None:
                continue
            if isinstance(x, (list, tuple)):
                out.extend(self._flatten(x))
            else:
                out.append(x)
        return out

    def _is_pipeline_like(self, obj) -> bool:
        return hasattr(obj, "config") or hasattr(obj, "unet") or hasattr(obj, "vae")

    def _unwrap_pipeline_like(self, obj):
        if self._is_pipeline_like(obj):
            return obj
        if isinstance(obj, (list, tuple)):
            for it in obj:
                if self._is_pipeline_like(it):
                    return it
        return obj  # allow non-pipeline objects to be tracked too

    def _collect_devices(self, pipe) -> List[torch.device]:
        devs = set()
        for name in ("unet","vae","text_encoder","text_encoder_2","text_encoder_3","transformer","image_encoder"):
            m = getattr(pipe, name, None)
            if m is not None:
                try:
                    for p in m.parameters():
                        devs.add(p.device)
                        break
                except Exception:
                    pass
        cn = getattr(pipe, "controlnet", None)
        if cn is not None:
            items = cn if isinstance(cn, (list, tuple)) else [cn]
            for c in items:
                try:
                    for p in c.parameters():
                        devs.add(p.device)
                        break
                except Exception:
                    pass
        exec_dev = getattr(pipe, "_execution_device", None)
        if isinstance(exec_dev, torch.device):
            devs.add(exec_dev)
        return list(devs)

    def _heal_split(self, pipe, dev_str: str):
        """Force all submodules (incl. controlnet(s)) onto dev_str and sync execution device."""
        _to_device(pipe, dev_str)
        cn = getattr(pipe, "controlnet", None)
        if cn is not None:
            items = cn if isinstance(cn, (list, tuple)) else [cn]
            for c in items:
                try: c.to(dev_str)
                except Exception: pass
        try:
            pipe._execution_device = torch.device(dev_str) if isinstance(dev_str, str) else dev_str
        except Exception:
            pass

    # -------- public switching API --------
    def before_switch(self, target_group: str, target_device_index: Optional[int] = None) -> str:
        """
        Call before initializing/using a new group.
        Returns the device string to use ('cuda:N' or 'cpu').
        IMPORTANT: The chosen device is remembered and reused in `after_switch`.
        """
        idx = self._pick_device_index(target_device_index)
        self._pending_device_index = idx  # remember for after_switch
        self._free_memory_if_needed(idx)

        # If switching groups, optionally free current now
        if self.current_group and self.current_group != target_group:
            if self.switch_policy == "offload":
                self._offload_group(self.current_group)
            elif self.switch_policy == "destroy":
                self._destroy_group(self.current_group)
            free_cuda()
            self.current_group = None
            self.current_device = None

        dev_str = self._device_string(idx)
        # Set CUDA current device early to reduce chances of tensors landing on a different GPU
        if torch.cuda.is_available() and idx is not None and idx >= 0:
            try: torch.cuda.set_device(idx)
            except Exception: pass
        return dev_str

    def after_switch(self, group: str, objects: Iterable[object], device_index: Optional[int] = None, recompile_unet: bool = False):
        """
        Register objects under a group and move them to the chosen device.
        This method now *reuses* the device selected in `before_switch` unless an explicit index is provided.
        """
        # Normalize and unwrap
        objs_in = self._flatten(list(objects))
        objs = [self._unwrap_pipeline_like(o) for o in objs_in if o is not None]
        self.registry[group] = objs

        # Use explicit device if provided, else the one chosen in before_switch
        idx = device_index if device_index is not None else self._pending_device_index
        if idx is None:
            idx = self._pick_device_index(None)  # final fallback

        dev_str = self._device_string(idx)

        # Move all pipelines/modules to the same device and set execution device
        for p in objs:
            if self._is_pipeline_like(p):
                _to_device(p, dev_str)
                if recompile_unet and hasattr(p, "unet") and p.unet is not None:
                    try:
                        p.unet = torch.compile(p.unet, mode="reduce-overhead")
                    except Exception:
                        pass

        # Heal/verify: ensure no multi-device split remains
        for p in objs:
            if not self._is_pipeline_like(p):
                continue
            devs = {str(d) for d in self._collect_devices(p) if isinstance(d, torch.device)}
            if len(devs) > 1:
                self._heal_split(p, dev_str)
                devs2 = {str(d) for d in self._collect_devices(p) if isinstance(d, torch.device)}
                if len(devs2) > 1:
                    msg = f"ModelManager.after_switch: pipeline spans multiple devices: {sorted(devs2)} (target {dev_str})"
                    if self.strict_device:
                        raise RuntimeError(msg)
                    else:
                        print(msg)

        # Finalize state
        self.current_group = group
        self.current_device = idx
        self._pending_device_index = None  # consumed

        # Set CUDA device so subsequent tensor creations default here
        if torch.cuda.is_available() and idx is not None and idx >= 0:
            try: torch.cuda.set_device(idx)
            except Exception: pass

    def move_current_to(self, device_index: Optional[int]):
        """Explicitly move the current group to a specific device index."""
        if self.current_group is None:
            return
        idx = self._pick_device_index(device_index)
        dev_str = self._device_string(idx)
        for p in self.registry.get(self.current_group, []):
            _to_device(p, dev_str)
        # Verify & heal if needed
        for p in self.registry.get(self.current_group, []):
            if self._is_pipeline_like(p):
                devs = {str(d) for d in self._collect_devices(p) if isinstance(d, torch.device)}
                if len(devs) > 1:
                    self._heal_split(p, dev_str)
        self.current_device = idx
        if torch.cuda.is_available() and idx is not None and idx >= 0:
            try: torch.cuda.set_device(idx)
            except Exception: pass
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
