"""Decision model: causal LM backbone + block-causal branch mask + pointer readout."""
import copy, math, os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Reuse existing rarely-used Qwen special tokens as delimiters (state, q, opt, /opt, decide) so no
# embedding rows need to be added/trained; LoRA adapts their meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
# Gemma 4 has no such tokens to borrow. Its first reserved tokens carry distinct pretrained embeddings (pairwise cosine
# <= 0.35 on gemma-4-26B-A4B), unlike the high reserved range (<unused89>...), which sits on <unk> (cosine 0.99). They are
# not added tokens, so caller text cannot produce them.
GEMMA_SPECIAL = ["<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>"]
# training context: state tokens, tokens per question branch, and the whole packed record. Frozen suites are admitted with
# this rule (kev.suite) and training applies it to records built on the fly, so train and eval see the same population.
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048
# serving context (kev.serve): per-branch cap mirrors Jev's ~32k, bounded by the base model window; longer than training, so untested there
SERVE_MAX_STATE, SERVE_MAX_BRANCH = 8192, 8192
SERVE_MAX_PACKED = SERVE_MAX_STATE + SERVE_MAX_BRANCH   # one row at most: a packed request longer than this runs in the row form (the block-causal mask is L x L)
# the longest state a checkpoint may be trained on (kev.train --max_state) and still leave every question its training
# branch budget when served: serving's row limit is SERVE_MAX_BRANCH = state + branch
MAX_TRAIN_STATE = SERVE_MAX_BRANCH - (MAX_BRANCH - MAX_STATE)


def training_context(max_state=MAX_STATE):
    """The encoder limits for training with the state limit lifted to `max_state`: the row (state + one branch) and
    packed limits grow by the same amount, so every question keeps its token budget. training_context() is the default
    training context (kev.suite.CONTEXT without `truncate`)."""
    if not MAX_STATE <= max_state <= MAX_TRAIN_STATE:
        raise ValueError(f"max_state must be in [{MAX_STATE}, {MAX_TRAIN_STATE}]")
    extra = max_state - MAX_STATE
    return {"max_state": max_state, "max_branch": MAX_BRANCH + extra, "max_packed": MAX_PACKED + extra}


def rows_per_pass(rows, prefix_len=0, budget=SERVE_MAX_PACKED):
    """How many causal rows one inference forward pass takes: as many as fit `budget` tokens counting the cached state
    each row carries (prefix_len) plus its own tokens, at least one. Memory per pass is bounded by one maximal row however
    many questions a request has, and the rows are independent, so the answers do not depend on the split. Kev-4B on MLX,
    a 4.8k-token state with 64 questions: 24.7 GB peak in one pass, 8.9 GB one row at a time (and faster: the cost was
    64 copies of the state cache)."""
    return max(1, budget // (prefix_len + max(len(r) for r in rows)))


class ContextOverflow(ValueError):
    """A record does not encode within its context (state, branch or packed limit). Serving turns it into a 422; the
    benchmark counts it as a rejected record for suites scored as published (skip_overlong)."""


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


def pad_id(tok):
    """The id used to right-pad token rows (never attended to); Qwen tokenizers define one, others fall back to 0."""
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def family(tok):
    """What encode() needs from a tokenizer, computed once per tokenizer: `delims` (the five delimiter ids: state, q, opt,
    /opt, decide), `bos` (the ids the tokenizer puts before plain text: [] for Qwen, [<bos>] for Gemma, which expects it
    as token 0) and `escape` (None for Qwen, whose control tokens are all `<|name|>`; otherwise a pattern over the added
    special tokens, so caller text cannot produce <bos>, <|turn> or <|image|>)."""
    fam = getattr(tok, "_kev_family", None)
    if fam is None:
        unk = getattr(tok, "unk_token_id", None)   # what a tokenizer returns for a token it does not have (Qwen: None)
        found = ((n, [tok.convert_tokens_to_ids(t) for t in n]) for n in (SPECIAL, GEMMA_SPECIAL))
        names, delims = next(((n, ids) for n, ids in found if all(i is not None and i != unk for i in ids)), (None, None))
        if names is None: raise ValueError(f"{type(tok).__name__} has none of Kev's delimiter token sets")
        plain, full = tok("a", add_special_tokens=False).input_ids, tok("a").input_ids
        escape = None
        if names is not SPECIAL:
            added = sorted({t.content for t in tok.added_tokens_decoder.values() if t.special}, key=len, reverse=True)
            escape = re.compile("|".join(map(re.escape, added)))
        fam = tok._kev_family = {"delims": delims, "bos": full[: len(full) - len(plain)] if full[-len(plain):] == plain else [], "escape": escape}
    return fam


def text_backbone(lm):
    """The decoder stack of a loaded *ForCausalLM: `.model`. Multimodal checkpoints (Gemma 4 loads as
    Gemma4ForConditionalGeneration) nest it one level deeper as `.language_model`; the vision and audio towers are dropped."""
    m = lm.model
    return getattr(m, "language_model", m)


def sliding_window(config):
    """The window of the local attention layers (Gemma 4: 1,024), or None when every layer is global. The packed mask
    applies it by position (kev.model.branch_mask_batch); caches are kept full length so the state prefix can be cropped."""
    return getattr(config, "sliding_window", None) if "sliding_attention" in set(getattr(config, "layer_types", None) or []) else None


def is_hybrid(config):
    """Whether a (text) config has Gated DeltaNet layers (Qwen3.5). Such backbones cannot honour the block-causal mask and
    run the row form; on Apple Silicon they are what the MLX backend is for."""
    return "linear_attention" in set(getattr(config, "layer_types", None) or [])


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """Tokenize caller-supplied text so it can never produce delimiter/control tokens (option boundaries are unforgeable).
    The fast tokenizer ignores split_special_tokens, so `<|name|>` is rewritten to `<¦name¦>` before tokenizing."""
    text = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    if (escape := family(tok)["escape"]) is not None: text = escape.sub(lambda m: "<¦" + m.group(0)[1:], text)
    return tok(text, add_special_tokens=False).input_ids


OPT_NONE, OPT_DECIDE = -1, -2   # values of enc["opt"]: instruction/state tokens, and the <decide> token


MEDIA_TYPES = ("image", "video", "audio")


def prepare_media(processor, items):
    """Run the base's own processor on each media item of a record ({"type": "image" | "video" | "audio", "data": ...,
    video: optional "metadata" {"fps", "total_num_frames"} and "num_frames"}). Returns one dict per item: `ids`, the item's
    tokens exactly as Gemma lays them out (<|image> placeholders <image|>; video: one "MM:SS <|image> ... <image|>" per
    sampled frame; audio: <|audio> placeholders <audio|>), `types` (Gemma's mm_token_type_ids: 1 image, 2 video frame,
    3 audio, 0 text) and `inputs`, the tensors the multimodal forward takes (pixel values, patch positions, audio features)."""
    bos = family(processor.tokenizer)["bos"]
    out = []
    for it in items:
        kind = it["type"]
        if kind == "image": kw = {"text": processor.image_token, "images": [it["data"]]}
        elif kind == "audio": kw = {"text": processor.audio_token, "audio": [it["data"]]}
        elif kind == "video":
            kw = {"text": processor.video_token, "videos": [it["data"]]}
            if "metadata" in it: kw["video_metadata"] = [it["metadata"]]
            if "num_frames" in it: kw["videos_kwargs"] = {"num_frames": it["num_frames"]}
        else: raise ValueError(f"media type must be one of {MEDIA_TYPES}, got {kind!r}")
        enc = dict(processor(**kw, return_tensors="pt"))
        ids, types = enc.pop("input_ids")[0].tolist(), enc.pop("mm_token_type_ids")[0].tolist(); enc.pop("attention_mask", None)
        if ids[: len(bos)] == bos: ids, types = ids[len(bos):], types[len(bos):]
        out.append({"type": kind, "ids": ids, "types": types, "inputs": enc})
    return out


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False, option_isolation=False, media=None):
    """Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt>... <decide>].

    Returns ids, seg (0 = state, k = question k), pos (branch positions restart after state),
    decide_idx [Q], opt_idx [Q][K] (index of </opt> token for each option), opt (per-token option index within its
    question: OPT_NONE for state/instruction, 0..K-1 for option spans, OPT_DECIDE for <decide>).

    option_isolation=True: every option span is its own sub-branch (it sees state + instruction + itself only), all
    option spans share the same position ids, and <decide> sits at one fixed position after the longest span. Then the
    per-option representations and <decide>'s attention over them are permutation-invariant by construction.

    media (prepare_media output): the items open the state, in order, after <state>; the state text follows. Truncation
    only ever cuts text. The encoding then also carries `media` (the items) and `blk`, a block id per token: every image
    and every video frame is one block, which attends to itself in both directions (Gemma's rule); -1 elsewhere.
    """
    fam = family(tok); head = fam["bos"] + fam["delims"][:1]          # [<bos>] <state>: the state's fixed tokens
    for m in media or []: head = head + m["ids"]
    state_tokens = user_tokens(tok, rec["state"])
    if len(head) > max_state or (strict and len(state_tokens) + len(head) > max_state):
        raise ContextOverflow(f"state exceeds {max_state} tokens: {len(state_tokens) + len(head)}")
    S = head + state_tokens[: max_state - len(head)]
    ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
    q_id, o_id, c_id, d_id = fam["delims"][1:]
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ContextOverflow(f"branch too long: {len(br)} tokens with a {len(S)}-token state (row limit {max_branch})")
        base = len(ids); p0 = len(S)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]
        if option_isolation:
            longest = max(len(sp) for sp in spans)
            br_pos = list(range(p0, p0 + len(instr))) + [p0 + len(instr) + i for sp in spans for i in range(len(sp))] + [p0 + len(instr) + longest]
        else:
            br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp); ends.append(cursor - 1)
        ids += br; seg += [k] * len(br); pos += br_pos; opt += br_opt
        decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in ends])
    enc = {"ids": ids, "seg": seg, "pos": pos, "opt": opt, "option_isolation": option_isolation, "decide_idx": decide_idx, "opt_idx": opt_idx,
           "labels": [q["label"] for q in rec["questions"]], "state_truncated": len(state_tokens) + len(head) > max_state}
    if media:
        types = [0] * (len(fam["bos"]) + 1) + [t for m in media for t in m["types"]]
        blk, n = [-1] * len(ids), -1
        for i, t in enumerate(types):
            if t in (1, 2):                                         # image / video-frame placeholders; audio stays causal
                if i == 0 or types[i - 1] not in (1, 2): n += 1
                blk[i] = n
        enc.update(media=media, blk=blk)
    return enc


def media_key(enc):
    """A digest of a record's media inputs, for anything keyed on the state (the serving prefix cache): the placeholder
    tokens of two images of the same size are identical, so the state's token ids alone do not identify it."""
    import hashlib
    h = hashlib.sha256()
    for m in enc.get("media") or []:
        for k in sorted(m["inputs"]):
            v = m["inputs"][k]; h.update(k.encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.cpu().numpy().tobytes())
    return h.hexdigest() if enc.get("media") else None


def collate_media(items, device, dtype):
    """One batch of multimodal inputs for the items of several records, in placeholder order (record by record), which
    is the order Gemma scatters features in. Items differ in patch, frame and audio-frame counts: they are padded with
    patch position -1 and audio mask False, which Gemma treats as padding and drops."""
    by_key = {}
    for m in items:
        for k, v in m["inputs"].items(): by_key.setdefault(k, []).append(v)
    out = {}
    for k, vs in by_key.items():
        shape = [max(v.shape[d] for v in vs) for d in range(1, vs[0].dim())]
        fill = -1 if k.endswith("position_ids") else (False if vs[0].dtype == torch.bool else 0)
        padded = []
        for v in vs:
            t = torch.full((v.shape[0], *shape), fill, dtype=v.dtype)
            t[(slice(None), *(slice(0, n) for n in v.shape[1:]))] = v
            padded.append(t)
        t = torch.cat(padded).to(device)
        out[k] = t.to(dtype) if t.is_floating_point() else t
    return out


def fits(rec, *tokenizers, max_state=MAX_STATE, max_branch=MAX_BRANCH, max_packed=MAX_PACKED):
    """True when the internal record encodes strictly (no truncation) within the training context under every tokenizer
    given (frozen suites are admitted against the tokenizers of all their pinned bases)."""
    try:
        return all(len(encode(tok, rec, max_state=max_state, max_branch=max_branch, strict=True)["ids"]) <= max_packed for tok in tokenizers)
    except ValueError:
        return False


def branch_mask(seg, device, dtype=torch.float32):
    """attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i]). Returns additive [1,1,L,L]."""
    return branch_mask_batch([seg], device, dtype)


def branch_mask_batch(segs, device, dtype=torch.float32, opts=None, length=None, pos=None, window=None, blocks=None, window_clips_blocks=False):
    """Batched block-causal mask, additive [B,1,L,L], right-padded to the longest sequence.

    Padded key positions are masked for every query; padded query rows keep the diagonal so no row is fully
    masked (finfo.min, not -inf, so softmax stays finite either way). Real tokens never see pads because pads sit
    after them (causal) and belong to no segment (-1).

    opts (option isolation): within a question, an option-span token may attend to state, the instruction, and its own
    span only; <decide> attends to everything in its question. Instruction tokens never see option spans (causal).

    window (sliding-window layers, with `pos`, the position ids): a query also drops keys `window` or more positions
    behind it. Distance is by position, not index, so a branch sees the same state tokens as its causal row does.

    blocks (media, one list per record or None): tokens with the same block id >= 0 also attend to each other in both
    directions (Gemma's rule for an image or a video frame). The two Gemma 4 families differ on the window:
    gemma4_unified (12B) OR-s the blocks onto the windowed mask; gemma4 (E2B/E4B/26B-A4B/31B) applies the window after
    the OR (window_clips_blocks) and keeps blocks off global layers, which the caller does by passing no blocks there."""
    L = max(max(len(s) for s in segs), length or 0)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = causal[None] & same & valid_key
    if opts is not None:
        o = torch.full((len(segs), L), OPT_NONE, device=device)
        for b, op in enumerate(opts):
            o[b, : len(op)] = torch.tensor(op, device=device)
        key_is_option = (o[:, None, :] >= 0)
        query_is_decide = (o[:, :, None] == OPT_DECIDE)
        same_option = o[:, None, :] == o[:, :, None]
        allow = allow & (~key_is_option | query_is_decide | same_option)
    same_block = None
    if blocks is not None:
        g = torch.full((len(segs), L), -1, device=device)
        for b, bl in enumerate(blocks):
            if bl is not None: g[b, : len(bl)] = torch.tensor(bl, device=device)
        same_block = (g[:, :, None] == g[:, None, :]) & (g[:, :, None] >= 0)
        if window_clips_blocks: allow = allow | same_block
    if window is not None:
        p = torch.zeros((len(segs), L), dtype=torch.long, device=device)
        for b, ps in enumerate(pos):
            p[b, : len(ps)] = torch.tensor(ps, device=device)
        allow = allow & ((p[:, :, None] - p[:, None, :]) < window)
    if same_block is not None and not window_clips_blocks:
        allow = allow | same_block
    allow = allow | torch.eye(L, dtype=torch.bool, device=device)[None]
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]


def rows_of(enc):
    """Split a packed encoding into its state and per-question branch rows.

    Returns (state_ids, state_pos, rows) with rows[k] = {"ids", "pos", "decide", "opts"}: the branch tokens of question
    k with their (already state-continuing) positions, and the readout offsets *within the branch*. Feeding
    state + rows[k] as one causal row is equivalent to the packed block-causal form for that question, on any
    architecture: the row contains exactly the tokens question k may attend to, in the same positions."""
    seg = enc["seg"]; Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1                                    # <decide> is the last token of its branch
        if seg[start] != k or seg[end - 1] != k: raise ValueError("branch layout mismatch")
        rows.append({"ids": enc["ids"][start:end], "pos": enc["pos"][start:end], "decide": d - start, "opts": [o - start for o in oi]})
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


def per_clip(get_audio_features):
    """Wrap a Gemma 4 audio encoder (E2B/E4B's conformer) to run one clip at a time, each trimmed to its own frames, with
    the outputs padded back. The encoder is not padding-invariant: a clip's features change with the longest clip it is
    batched with (max |delta| 0.8 on a tiny E4B, even with transformers' own feature-extractor padding), and an answer must
    not depend on what else is in the batch. The 12B has no audio encoder (samples are projected directly) and needs none."""
    def run(input_features, input_features_mask, **kw):
        outs = [get_audio_features(f[None, : int(m.sum())], m[None, : int(m.sum())], **kw) for f, m in zip(input_features, input_features_mask)]
        T = max(o.pooler_output.shape[1] for o in outs); ref = outs[0].pooler_output
        pooled = torch.zeros(len(outs), T, ref.shape[-1], dtype=ref.dtype, device=ref.device)
        mask = torch.zeros(len(outs), T, dtype=torch.bool, device=ref.device)
        for i, o in enumerate(outs):
            n = o.pooler_output.shape[1]; pooled[i, :n] = o.pooler_output[0]; mask[i, :n] = o.attention_mask[0]
        outs[0].pooler_output, outs[0].attention_mask = pooled, mask
        return outs[0]
    return run


class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        """dp = pointer dimension (head capacity knob)."""
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        # calibration: logits are divided by this at inference (eval mode) only. 1.0 = raw. A checkpoint carries the value fitted on
        # its in-distribution development rows (scripts/calibrate_checkpoint.py -> head.pt["temperature"]); training always sees T=1 so
        # a fitted value stays meaningful, and the argmax is unchanged by construction.
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature


# What a loaded model exposes to kev.serve, kev.predictors and the Space: the scoring interface both DecisionModel (torch)
# and kev.mlx_model.MLXDecisionModel implement. tests/test_mlx.py checks the MLX class against this list.
SCORING_INTERFACE = ("encode", "forward", "probs", "probs_and_prefix", "probs_with_prefix", "eval",
                     "head", "backend", "dtype", "device", "hybrid", "option_isolation", "prefix_min_tokens")


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None, head_dim=256, option_isolation=False, special_embeddings=False, lora_targets="all", dtype=torch.float32):
        super().__init__()
        # backbone only (no vocab head): we never generate text.
        # eager on MPS/CPU (known-good with our float 4D mask); SDPA on CUDA (accepts arbitrary additive masks).
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        # dtype: fp32 for training and exact evaluation; bf16 is a serving option for large backbones (8B on a 32 GB Mac)
        full = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn)
        self.lm = text_backbone(full)
        # multimodal bases (Gemma 4): the wrapper around self.lm that embeds images, video frames and audio and scatters them
        # into the placeholder positions. Its encoders stay frozen; records without media never go through it.
        self.mm = full.model if full.model is not self.lm else None
        if self.mm is not None:
            for prm in self.mm.parameters(): prm.requires_grad_(False)
            if getattr(getattr(self.mm.config, "audio_config", None), "model_type", None) == "gemma4_audio":
                self.mm.get_audio_features = per_clip(self.mm.get_audio_features)
        self.name, self.revision, self._processor = name, revision, None
        self.pad_id = pad_id(tok)
        # hybrid backbones (Qwen3.5: Gated DeltaNet layers, recurrent) cannot honour the block-causal mask, so every
        # question runs as its own causal row continuing from the state (rows_of). Attention-only backbones keep the
        # packed form; the two agree to fp32 noise (tests/test_model.py::test_rows_match_packed).
        self.hybrid = is_hybrid(self.lm.config)
        if self.hybrid and option_isolation: raise ValueError("option_isolation needs the packed mask; not available on hybrid backbones")
        self.option_isolation = option_isolation
        # attention-only backbones with local layers (Gemma 4) keep the packed form; the mask gets a per-layer-type variant
        self.window = sliding_window(self.lm.config)
        if lora:
            from peft import LoraConfig, get_peft_model
            extra = {"trainable_token_indices": {"embed_tokens": family(tok)["delims"]}} if special_embeddings else {}
            targets = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       "dense": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   # "all" minus the DeltaNet projections on hybrids (retention ablation)
                       "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "qv": ["q_proj", "v_proj"]}[lora_targets]
            if self.hybrid and lora_targets in ("all", "attn"):
                # Gated DeltaNet projections (transformers 5 names, verified on Qwen3_5TextModel); the mixer's out_proj too
                targets = targets + ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
            # MoE backbones: routed experts are fused 3D parameters (Gemma 4, Qwen3.5-MoE), which these names do not reach;
            # the adapter covers attention and the dense MLP that runs beside the experts (Gemma 4's `mlp`)
            cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora, lora_dropout=0.05, target_modules=targets, **extra)
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    backend = "torch"           # kev.mlx_model.MLXDecisionModel is the other implementation of this scoring interface

    @property
    def prefix_min_tokens(self):
        """kev.serve caches the state prefix from this many state tokens. Attention-only backbones: 384, below which the
        branch-only pass is not faster than one packed pass on MPS (per-op overhead). Hybrid backbones: always, because
        their miss path otherwise recomputes the state once per question (Kev-0.8B bf16 on MPS, 5 questions: 1011 -> 413 ms)."""
        return 0 if self.hybrid else 384

    @property
    def dtype(self):
        return str(next(self.lm.parameters()).dtype).removeprefix("torch.")

    def encode(self, tok, rec, **kw):
        """encode() with this model's option-isolation setting and, for records with "media", the base's processor; use
        this from serving/eval/training code."""
        media = None
        if rec.get("media"):
            if self.mm is None: raise ValueError(f"{self.name} takes text only; this record has media")
            kinds = {m["type"] for m in rec["media"]}
            for kind, cfg in (("audio", "audio_config"), ("image", "vision_config"), ("video", "vision_config")):
                if kind in kinds and getattr(self.mm.config, cfg, None) is None: raise ValueError(f"{self.name} has no {kind} input")
            from .media import load
            items = [load(m) if isinstance(m.get("data"), str) or "path" in m or "url" in m else m for m in rec["media"]]
            media = prepare_media(self.processor, items)
        return encode(tok, rec, option_isolation=self.option_isolation, media=media, **kw)

    @property
    def processor(self):
        """The base's multimodal processor (images, video, audio -> placeholder tokens + tensors), loaded on first use: it
        needs the `media` extra (torchvision, Pillow), which text-only use does not."""
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.name, revision=self.revision)
        return self._processor

    def _backbone(self, encs, ids, pos, mask, **kw):
        """Final hidden states of the backbone for these rows. Records with media run through the multimodal wrapper, which
        embeds their items into the placeholder positions and then runs the same text model under the same mask (a dict
        mask passes through it unchanged)."""
        items = [m for e in encs for m in e.get("media") or []]
        if not items:
            return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask, **kw)
        media = collate_media(items, self.device, next(self.mm.parameters()).dtype)
        return self.mm(input_ids=ids, position_ids=pos, attention_mask=mask, **media, **kw)

    def hidden(self, enc):
        return self.hidden_batch([enc])[0, : len(enc["ids"])]

    SHAPE_BUCKET = int(os.environ.get("KEV_SHAPE_BUCKET", "64"))   # MPS: pad the sequence to a multiple of this (per-shape kernel warm-up); 1 disables

    def _pad_rows(self, rows):
        """Right-pad (ids, pos) token rows into [N, L] id / position tensors and a [N, L] attention mask (1 = real token).
        Pads sit after every real token and are masked keys, so they never change a real token's hidden state (parity
        measured exact). On MPS in eval mode L is rounded up to a SHAPE_BUCKET multiple so kernels are warmed per bucket."""
        L = max(len(ids) for ids, _ in rows)
        if str(self.device) == "mps" and not self.training: L = -(-L // self.SHAPE_BUCKET) * self.SHAPE_BUCKET
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos) in enumerate(rows):
            ids[i, : len(rid)] = torch.tensor(rid, device=self.device); pos[i, : len(rpos)] = torch.tensor(rpos, device=self.device); att[i, : len(rid)] = 1
        return ids, pos, att

    def hidden_batch(self, encs):
        """[B, L_max, d] hidden states for a right-padded batch of encoded records under the packed block-causal mask."""
        ids, pos, _ = self._pad_rows([(e["ids"], e["pos"]) for e in encs])
        isolate = any(e.get("option_isolation") for e in encs)
        if isolate and not all(e.get("option_isolation") for e in encs):
            raise ValueError("cannot mix option-isolated and plain encodings in one batch")
        mask = self._mask(encs, length=ids.shape[1])
        return self._backbone(encs, ids, pos, mask).last_hidden_state.float()   # head stays fp32

    def _mask(self, encs, length=None, queries_from=0):
        """The packed block-causal mask for these records in the backbone's dtype, keeping query rows from `queries_from` on
        (the branches, when the state is cached). Backbones with sliding-window layers get the per-layer-type dict
        transformers accepts in place of one mask: the same mask for global layers, the window applied for local ones."""
        dt = next(self.lm.parameters()).dtype
        opts = [e["opt"] for e in encs] if any(e.get("option_isolation") for e in encs) else None
        segs, pos = [e["seg"] for e in encs], [e["pos"] for e in encs]
        blocks = [e.get("blk") for e in encs] if any("blk" in e for e in encs) else None
        # image blocks (see branch_mask_batch) only where the base was trained with them: use_bidirectional_attention ==
        # "vision" (12B, 26B-A4B, 31B; E2B/E4B keep images causal), on every layer for gemma4_unified, local layers for gemma4
        if getattr(self.lm.config, "use_bidirectional_attention", None) != "vision": blocks = None
        unified = self.mm is not None and self.mm.config.model_type == "gemma4_unified"
        full = branch_mask_batch(segs, self.device, dtype=dt, opts=opts, length=length, blocks=blocks if unified else None)[:, :, queries_from:, :]
        if self.window is None: return full
        local = branch_mask_batch(segs, self.device, dtype=dt, opts=opts, length=length, pos=pos, window=self.window, blocks=blocks,
                                  window_clips_blocks=not unified)[:, :, queries_from:, :]
        return {"full_attention": full, "sliding_attention": local}

    def _new_cache(self):
        """An empty cache for a prefix pass. Hybrid layers need the config (recurrent + conv state per DeltaNet layer).
        Sliding-window layers must not get it: a window-sized cache layer cannot be cropped back to the state once it
        is full, so they keep full-length keys and the mask applies the window."""
        return DynamicCache() if self.window is not None else DynamicCache(config=self.lm.config)

    def _readout(self, h, enc):
        return [self.head(h[d], h[torch.tensor(oi, device=self.device)]) for d, oi in zip(enc["decide_idx"], enc["opt_idx"])]

    def rows_form(self, encs):
        """Whether these records run as causal rows: hybrid backbones always (the recurrent layers cannot honour the packed
        mask), attention-only ones when the packed sequence would exceed one serving row (its L x L mask grows without
        bound with the number of questions). The two forms agree (tests/test_model.py::test_rows_match_packed)."""
        return self.hybrid or any(len(e["ids"]) > SERVE_MAX_PACKED for e in encs)

    def _rows_hidden(self, rows, cache=None, prefix_len=0):
        """Hidden states of causal token rows, one [L_i, d] tensor per row. In eval mode the rows go through the backbone
        rows_per_pass at a time; training keeps one batch (its batches are small and autograd needs the whole graph anyway).
        With `cache`, the rows are branches continuing the cached state: the cache is replicated once per chunk (a copy,
        so the caller's prefix stays pristine) and the cached tokens are marked real in the attention mask."""
        chunk = len(rows) if self.training else rows_per_pass([ids for ids, _ in rows], prefix_len)
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, att = self._pad_rows(part)
            past = {}
            if cache is not None:
                replica = copy.deepcopy(cache); replica.reorder_cache(torch.zeros(len(part), dtype=torch.long, device=self.device))
                att = torch.cat([torch.ones((len(part), prefix_len), dtype=torch.long, device=self.device), att], 1)
                past = {"past_key_values": replica, "use_cache": True}
            h = self.lm(input_ids=ids, position_ids=pos, attention_mask=att, **past).last_hidden_state.float()
            out += [h[i, : len(row_ids)] for i, (row_ids, _) in enumerate(part)]
        return out

    def forward_rows_batch(self, encs):
        """Row form: every question of every record is one causal row = state tokens + its branch tokens. Returns the same
        nested logits as forward_batch. Exact isolation by construction (rows are independent); the state is recomputed
        per row (Q x state tokens), which training accepts; serving uses the prefix cache instead. Records with media run the
        state once (prefix) and the branches as rows from its cache: a causal row cannot carry Gemma's image blocks."""
        if any(e.get("media") for e in encs):
            if self.training: raise ValueError("records with media train in the packed form (rows_form is for over-long serving requests)")
            return [self._branch_rows_logits(e, self.prefix(e)[1]) for e in encs]
        rows, readouts = [], []   # one causal row per question; readouts[i] = (record, <decide> offset, option offsets)
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for r in brs:
                rows.append((S + r["ids"], Sp + r["pos"])); readouts.append((b, len(S) + r["decide"], [len(S) + o for o in r["opts"]]))
        out = [[] for _ in encs]
        for h, (b, d, oi) in zip(self._rows_hidden(rows), readouts):
            out[b].append(self.head(h[d], h[torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        """Returns list of logits tensors, one per question."""
        return self.forward_batch([enc])[0]

    def forward_batch(self, encs):
        """List (per record) of lists (per question) of logits. Row form or packed block-causal mask, see rows_form."""
        if self.rows_form(encs): return self.forward_rows_batch(encs)
        hs = self.hidden_batch(encs)
        return [self._readout(hs[b], e) for b, e in enumerate(encs)]

    @torch.no_grad()
    def probs(self, enc):
        return [F.softmax(z, -1).cpu() for z in self.forward(enc)]

    # --- state-prefix reuse (serving): the state is encoded once, question branches attend to its cached keys/values.
    # Exact by construction: branch tokens never attend to each other across questions (block-causal mask) and the state
    # never sees the branches (causal), so the state's hidden states and KV are identical with or without the branches.

    def _branch_rows_logits(self, enc, cache):
        """Row-form serving: the branches run as causal rows continuing the cached state (exactly the forward_rows_batch
        layout, minus the recomputed state). Logits per question."""
        S, _, rows = rows_of(enc)
        hs = self._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=cache, prefix_len=len(S))
        return [self.head(h[r["decide"]], h[torch.tensor(r["opts"], device=self.device)]) for h, r in zip(hs, rows)]

    def _branch_rows_from_prefix(self, enc, cache):
        return [F.softmax(z, -1).cpu() for z in self._branch_rows_logits(enc, cache)]

    @torch.no_grad()
    def prefix(self, enc):
        """Run the state tokens only. Returns (n_state_tokens, kv cache, state hidden states [Ls, d])."""
        Ls = enc["seg"].count(0)
        ids = torch.tensor([enc["ids"][:Ls]], device=self.device); pos = torch.tensor([enc["pos"][:Ls]], device=self.device)
        # the cache must know the layer types (hybrid backbones keep recurrent + conv states per DeltaNet layer)
        if enc.get("media"):   # the state's own mask, for its image blocks; the items embed through the multimodal wrapper
            mask = self._mask([{"seg": enc["seg"][:Ls], "pos": enc["pos"][:Ls], "blk": enc["blk"][:Ls]}])
            out = self._backbone([enc], ids, pos, mask, past_key_values=self._new_cache(), use_cache=True)
        else:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=self._new_cache(), use_cache=True)
        return Ls, out.past_key_values, out.last_hidden_state[0].float()

    @torch.no_grad()
    def probs_and_prefix(self, enc):
        """One full pass that also returns the state prefix (KV cropped to the state, state hidden states): a cache miss
        costs a single forward pass, not two."""
        Ls = enc["seg"].count(0)
        if self.rows_form([enc]):
            # recurrent layers cannot be cropped back to the state (and an over-long packed pass is what the row form avoids),
            # so a miss here is a state pass (kept as the prefix) plus the branch rows
            Ls, cache, h_state = self.prefix(enc)
            return self._branch_rows_from_prefix(enc, cache), (Ls, cache, h_state)
        ids = torch.tensor([enc["ids"]], device=self.device); pos = torch.tensor([enc["pos"]], device=self.device)
        out = self._backbone([enc], ids, pos, self._mask([enc]), past_key_values=self._new_cache(), use_cache=True)
        h = out.last_hidden_state[0].float()
        out.past_key_values.crop(-(len(enc["ids"]) - Ls))     # keep the state only (negative = drop that many trailing tokens; positive form deprecated in transformers 5)
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)], (Ls, out.past_key_values, h[:Ls].clone())

    @torch.no_grad()
    def probs_with_prefix(self, enc, prefix):
        """probs() for a record whose state tokens equal the cached prefix's; only the branches run. The cache is cropped
        back to the state afterwards so it can be reused."""
        Ls, cache, h_state = prefix
        if enc["seg"].count(0) != Ls: raise ValueError("prefix does not match this record's state")
        if self.rows_form([enc]):
            return self._branch_rows_from_prefix(enc, cache)
        ids = torch.tensor([enc["ids"][Ls:]], device=self.device); pos = torch.tensor([enc["pos"][Ls:]], device=self.device)
        mask = self._mask([enc], queries_from=Ls)
        try:
            out = self.lm(input_ids=ids, position_ids=pos, past_key_values=cache, attention_mask=mask, use_cache=True)
            h = torch.cat([h_state, out.last_hidden_state[0].float()], 0)
        finally:
            cache.crop(-(len(enc["ids"]) - Ls))
        return [F.softmax(z, -1).cpu() for z in self._readout(h, enc)]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
