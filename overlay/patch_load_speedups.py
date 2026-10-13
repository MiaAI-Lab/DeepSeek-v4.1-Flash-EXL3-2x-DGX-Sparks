#!/usr/bin/env python3
"""Cold-load speedups for DeepSeek V4.1 on two GB10 Sparks (measured 2026-10-12, see README "Cold-load time").

1. [dsv41-draft-shards]  the DSpark draft opens only the shards that hold mtp.* (2 of 39) instead of get_tensor()-ing all 39.
2. [dsv41-fast-st]       safetensors_weights_iterator reads each tensor with one header parse + torch.frombuffer.
3. [dsv41-pycache]       compiled bytecode kept across boots under the host-mounted vLLM cache.

Idempotent; each part prints applied / already applied / SKIPPED and never fails the boot.
"""
import glob as _g

# ---------------------------------------------------------------------------------------------------------------
# [dsv41-draft-shards] The DSpark draft keeps only mtp.* tensors (_remap_dspark_name drops
# the rest), but DefaultModelLoader handed it all 39 shards and get_tensor()'d every tensor in them: py-spy on
# 2026-10-11 put the 17-30 s draft pass in safetensors_weights_iterator. The index puts every mtp.* tensor in
# shards 38-39, so the class tells the loader to open only the shards that hold mtp.* (the stock
# allow_patterns_overrides hook). No index found -> None -> the stock all-shards behaviour.
for _f in _g.glob("/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/*/dspark.py"):
    _s = open(_f).read()
    _old = '    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n        """Load the ``mtp.{0,1,2}.*`` draft weights'
    if "[dsv41-draft-shards]" in _s:
        print("draft shards: already applied (%s)" % _f); continue
    if _s.count(_old) != 1:
        print("draft shards: SKIPPED, anchor not found once (%s)" % _f); continue
    _new = '''    @property
    def allow_patterns_overrides(self):  # [dsv41-draft-shards]
        import json as _j, os as _o
        _p = _o.path.join(getattr(self.config, "_name_or_path", "") or "", "model.safetensors.index.json")
        if not _o.path.isfile(_p):
            return None
        _files = sorted({f for k, f in _j.load(open(_p))["weight_map"].items() if k.startswith("mtp.")})
        # _prepare_weights stops at the FIRST pattern that matches, so hand it one glob covering every needed shard
        # (a per-position character class; over-matching only costs reads, never correctness).
        if not _files or len({len(f) for f in _files}) != 1:
            return None
        return ["".join(c[0] if len(set(c)) == 1 else "[" + "".join(sorted(set(c))) + "]" for c in zip(*_files))]

''' + _old
    _s = _s.replace(_old, _new)
    open(_f, "w").write(_s); print("draft shards: applied (%s)" % _f)

# [dsv41-draft-shards] part 2: filter_duplicate_safetensors_files raises when an index-listed file is absent from the
# load set, which any explicit allow_patterns_overrides subset always is. Explicit overrides name their files on purpose,
# so skip the duplicate filter for them (the only other override user, the engram path, discards its tensors anyway).
_f = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/default_loader.py"
_s = open(_f).read()
_old = "            hf_weights_files = filter_duplicate_safetensors_files(\n                hf_weights_files, hf_folder, index_file\n            )\n"
_new = "            if allow_patterns_overrides is None:  # [dsv41-draft-shards]\n                hf_weights_files = filter_duplicate_safetensors_files(\n                    hf_weights_files, hf_folder, index_file\n                )\n"
if "[dsv41-draft-shards]" in _s:
    print("draft shards loader: already applied")
elif _s.count(_old) != 1:
    print("draft shards loader: SKIPPED, anchor not found once")
else:
    open(_f, "w").write(_s.replace(_old, _new)); print("draft shards loader: applied")

# ---------------------------------------------------------------------------------------------------------------
# [dsv41-fast-st] py-spy: most of the main load's non-io time sat in safetensors get_tensor
# (torch storage slicing, ~0.65 s per 4.9 GiB shard x 39 per rank). This reader parses the header once, maps the file
# MAP_PRIVATE (same as safetensors: the madv release logic still finds the *.safetensors mapping) and makes each tensor
# with torch.frombuffer(offset, count) -> a storage covering exactly that tensor, like safetensors. Keys in safetensors'
# order (sorted) so the pack replay sequence is unchanged. Measured: byte-identical, 6-24x faster per shard.
_f = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"
_s = open(_f).read()
_old = '''        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
'''
_new = '''        else:
            yield from _dsv41_fast_st(st_file, local_expert_ids)  # [dsv41-fast-st]
'''
_helper = '''

# [dsv41-fast-st] see overlay/patch_load_speedups.py
def _dsv41_fast_st(st_file, local_expert_ids):
    import json as _j, mmap as _m, struct as _st
    _DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
           "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "U16": torch.uint16, "I32": torch.int32,
           "U32": torch.uint32, "I64": torch.int64, "U64": torch.uint64, "BOOL": torch.bool,
           "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
    with open(st_file, "rb") as fh:
        n = _st.unpack("<Q", fh.read(8))[0]
        header = _j.loads(fh.read(n))
        if any(k != "__metadata__" and v["dtype"] not in _DT for k, v in header.items()):
            mm = None
        else:
            mm = _m.mmap(fh.fileno(), 0, access=_m.ACCESS_COPY)
    if mm is None:  # a dtype this reader does not know: the stock path
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                if should_skip_weight(name, local_expert_ids):
                    continue
                yield name, f.get_tensor(name)
        return
    base = 8 + n
    for name in sorted(k for k in header if k != "__metadata__"):
        if should_skip_weight(name, local_expert_ids):
            continue
        v = header[name]; dt = _DT[v["dtype"]]; a, b = v["data_offsets"]
        if b == a:
            yield name, torch.empty(v["shape"], dtype=dt)
        else:
            yield name, torch.frombuffer(mm, dtype=dt, count=(b - a) // dt.itemsize, offset=base + a).view(v["shape"])
'''
if "[dsv41-fast-st]" in _s:
    print("fast st: already applied")
elif _s.count(_old) != 1:
    print("fast st: SKIPPED, anchor found %d times" % _s.count(_old))
else:
    open(_f, "w").write(_s.replace(_old, _new) + _helper); print("fast st: applied")

# ---------------------------------------------------------------------------------------------------------------
# [dsv41-pycache] The image ships .pyc for only 649 of vLLM's 2,601 modules (and ~half of torch), and every boot
# is a fresh container, so each of the head's and worker's processes recompiled them and threw the result away (py-spy:
# _compile_bytecode/_write_atomic during the imports; measured cold 5.3 s vs warm 4.0 s per import set). A .pth line points
# sys.pycache_prefix at the vLLM cache dir both containers already mount from the host (/root/.cache/vllm), so the compiled
# files survive across boots. Python still checks each .pyc against its source (a patched file just recompiles).
import os as _o
_pth = "/usr/local/lib/python3.12/dist-packages/zz_dsv41_pycache.pth"
if _o.path.isdir("/root/.cache/vllm"):
    open(_pth, "w").write('import sys; sys.pycache_prefix = "/root/.cache/vllm/pycache"\n')
    print("pycache prefix: /root/.cache/vllm/pycache")
else:
    print("pycache prefix: SKIPPED (/root/.cache/vllm not mounted)")
