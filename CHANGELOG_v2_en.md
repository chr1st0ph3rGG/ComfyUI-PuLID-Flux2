# PuLID-Flux2 v2.0 - CHANGELOG & INSTALLATION

## 🎯 Main improvements

### 1. **Auto dtype detection per GPU** ✅
**Problem solved:** RTX 3090 (Ampere) has no native bf16 support → emulation via fp32 = 10x+ slower

**Solution:**
- Automatic GPU architecture detection via `torch.cuda.get_device_capability()`
- **Ampere (RTX 30xx, A100):** dtype = `fp16` (optimal)
- **Ada (RTX 40xx):** dtype = `bf16` (native)
- **Hopper (H100):** dtype = `bf16` (native + Tensor Cores)

**Before:**
```python
dtype = torch.bfloat16  # Hardcoded → slow on RTX 3090
```

**After:**
```python
dtype = get_optimal_dtype(device)  # Auto fp16 on RTX 3090
```

---

### 2. **Aggressive VRAM cleanup** 🗑️
**Problem solved:** Memory leak after each generation → run 2+ becomes slow

**Solution:**
- Automatic garbage collection after each application
- Option `cleanup_vram` (default: True) in the Apply PuLID node
- Function `clear_model_cache()` to clear the global cache

**Impact:**
- Run 1: 2s/it
- Run 2+: 2s/it (stable, no slowdown)

---

### 3. **Configurable cache** 🎛️
**Problem solved:** Global cache can cause conflicts with some workflows

**Solution:**
- Option `use_cache` in the EVA-CLIP and InsightFace loaders
- New node **PuLIDCacheCleaner** to manually clear the cache
- Cache can be disabled if needed

**Usage:**
```
PuLIDEVACLIPLoader → use_cache=False (reload every time)
PuLIDCacheCleaner → clear=True (clears the cache)
```

---

### 4. **Patch conflict detection** ⚠️
**Problem solved:** Silent collisions with Consistency Models, LoRAs, etc.

**Solution:**
- Automatic scan for patch markers (`_lora_patched`, `_consistency_patched`, etc.)
- Warning if conflict detected
- Detailed logs in debug mode

**Example output:**
```
[PuLID] ⚠️  3 patch conflict(s) detected - may cause instability
  - double_block_2 has _consistency_patched
  - single_block_6 has _lora_patched
```

---

### 5. **Configurable sigma range parameters** 📊
**New parameter:** `sigma_start` and `sigma_end` in Apply PuLID

**Usage:**
- Default: `sigma_start=0.0, sigma_end=1.0` (full denoising)
- Custom: `sigma_start=0.3, sigma_end=0.8` (injects PuLID only in middle of denoising)

**Use cases:**
- Light preservation: `sigma_start=0.0, sigma_end=0.5`
- Strong preservation: `sigma_start=0.0, sigma_end=1.0`

---

### 6. **Extended debug mode** 🔍
**Additional information:**
- GPU capability (compute version)
- Automatically selected dtype
- Patched blocks (indices)
- Patch conflict detection
- Embedding shapes at each step

**Activation:**
```
Apply PuLID → debug_mode=True
```

---

## 📋 New nodes

### **PuLIDCacheCleaner** 🗑️
Clears the global cache for EVA-CLIP and InsightFace models

**Usage:**
1. Add the node to your workflow
2. Set `clear=True`
3. Execute → cache cleared + VRAM freed

**Use cases:**
- Workflow switch
- Suspected memory leak
- Before training/fine-tuning

---

## 🚀 Installation

### Method 1: Direct replacement
```bash
# Backup old file
copy C:\AI\ComfyUI\custom_nodes\ComfyUI-PuLID-Flux2\pulid_flux2.py C:\AI\ComfyUI\custom_nodes\ComfyUI-PuLID-Flux2\pulid_flux2.py.backup

# Replace with v2
copy pulid_flux2_v2_complete.py C:\AI\ComfyUI\custom_nodes\ComfyUI-PuLID-Flux2\pulid_flux2.py
```

### Method 2: Fresh install
```bash
# Clone the repo (once published on GitHub)
cd C:\AI\ComfyUI\custom_nodes
git clone https://github.com/iFayens/ComfyUI-PuLID-Flux2.git
cd ComfyUI-PuLID-Flux2
git checkout v2.0
```

---

## 🔧 Recommended configuration

### For RTX 3090 (24GB VRAM)
```batch
# Launch script
python main.py --force-fp16 --normalvram --preview-method auto
```

**Node settings:**
- `use_cache=True` (saves loading time)
- `cleanup_vram=True` (avoids memory leak)
- `debug_mode=False` (unless debugging)

**VRAM budget:**
- Klein 4B + PuLID: ~16GB → comfortable
- Klein 9B + PuLID: ~21GB → OK
- Klein 9B + PuLID + Consistency: ~24GB → limit (disable LoRAs)

---

### For RTX 4090 (24GB VRAM)
```batch
# Launch script
python main.py --highvram --preview-method auto
```

**Difference vs RTX 3090:**
- Native bf16 → can use `--highvram` without issues
- Same VRAM (24GB) but better management

---

## 🐛 Troubleshooting

### "Still slow on the 2nd run"
**Solution:**
1. Set `cleanup_vram=True` in Apply PuLID
2. Use `--normalvram` instead of `--highvram`
3. Add PuLIDCacheCleaner node between generations

### "Conflict with Consistency Model"
**Solution:**
1. Disable Consistency OR PuLID (not both)
2. OR switch to Klein 4B (saves ~4.5GB)
3. OR rollback NVIDIA drivers 595.97 → 560.94

### "dtype bf16 still used on RTX 3090"
**Check:**
```python
# In ComfyUI console after loading
import torch
print(torch.cuda.get_device_capability())  # Should be (8, 6) for RTX 3090
```

If dtype is still bf16 → verify that v2 is properly loaded (check console log on startup)

---

## 📊 Benchmarks

### RTX 3090 + Klein 9B + PuLID

**Before v2 (with PyTorch 2.5.1 + xformers 0.0.35 + drivers 595.97):**
- Run 1: ~40s/it (forced bf16)
- Run 2+: CRASH or 12s/it (memory leak + swap)

**After v2 (with PyTorch 2.4.0 + xformers 0.0.27 + drivers 560.94 + --normalvram):**
- Run 1: ~2s/it (auto fp16)
- Run 2+: ~2s/it stable (VRAM cleanup)

**Gain:** ~20x faster + full stability

---

## 🎓 Migration from v1

### Breaking changes
**None!** v2 is **100% compatible** with v1 workflows.

### New parameters (optional)
- `sigma_start` / `sigma_end`: default = `0.0` / `1.0` (v1 behavior)
- `cleanup_vram`: default = `True` (improves stability)
- `use_cache`: default = `True` (v1 behavior)

### New nodes (optional)
- **PuLIDCacheCleaner**: for advanced workflows
- Display names changed (suffix " v2") for easy distinction

---

## 📝 Technical notes

### get_optimal_dtype() logic
```python
capability = torch.cuda.get_device_capability(device)
major, minor = capability

if major >= 9:           # Hopper (H100)
    return torch.bfloat16
elif major == 8 and minor >= 9:  # Ada (RTX 40xx)
    return torch.bfloat16
elif major == 8:         # Ampere (RTX 30xx, A100)
    return torch.float16
else:                    # Older
    return torch.float16
```

### Patch conflict detection
Scans all blocks for these markers:
- `_lora_patched`
- `_consistency_patched`
- `_ipadapter_patched`
- `_custom_forward`

If found → warning (continues anyway, but notifies the user)

---

## 🤝 Contributing

To contribute or report bugs:
- GitHub: https://github.com/iFayens/ComfyUI-PuLID-Flux2
- Issues: Describe your GPU, drivers, PyTorch version, full log

---

## 📜 License

Same license as v1 (to be defined based on the original repo)

---

## 🙏 Credits

**Developed by:** Mems (iFayens)  
**Based on:** PuLID paper + Flux.2 architecture  
**Debugged with:** Claude (Anthropic) — marathon session March 27, 2026 🚀

---

**Version:** 2.0.0  
**Date:** March 27, 2026  
**Tested on:** RTX 3090, PyTorch 2.4.0, ComfyUI 0.18.1
