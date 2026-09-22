"""Gemma 4 backbones (attention-only, sliding-window + global layers; 26B-A4B, E4B and 12B architectures): encoding, the
per-layer-type packed mask, prefix cache and isolation, on a tiny random-weight model with the real 26B-A4B architecture
flags and tokenizer. Downloads the Gemma 4 tokenizer and config only (no weights); not run in CI.
Run: uv run --extra serve python -m pytest tests/test_gemma4.py -q
"""
import pytest
import torch

# (base, revision, tiny text layers, shared-KV layers): the three Gemma 4 architectures, shrunk. 26B-A4B: MoE beside a dense
# MLP, vision tower. E4B: per-layer embeddings, shared-KV layers, vision + audio towers. 12B: gemma4_unified, encoder-free
# vision/audio embedders. Local layers use WINDOW, far below the states and rows below, so every local layer drops keys.
WINDOW = 16
BASES = {"26b-a4b": ("google/gemma-4-26B-A4B", "24548b62aa021d562695c04aaf7758a1ea47990b", ["sliding_attention"] * 3 + ["full_attention"], 0),
         "e4b": ("google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a", ["sliding_attention", "sliding_attention", "full_attention"] * 2, 2),
         "12b": ("google/gemma-4-12B", "023679ed352de9bb66cc873c9009ce3482585c08", ["sliding_attention"] * 3 + ["full_attention"], 0)}
BASE, REVISION = BASES["26b-a4b"][:2]


def shrink(d, layer_types, kv_shared):
    """A real Gemma 4 config dict cut down to a few layers and tiny widths, keeping every architecture flag."""
    t = d["text_config"]; t.pop("per_layer_config", None)                # derived from layer_types; rebuilt for the new depth
    t.update(hidden_size=64, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
             head_dim=16, global_head_dim=32, num_hidden_layers=len(layer_types), layer_types=layer_types, sliding_window=WINDOW,
             num_kv_shared_layers=kv_shared)
    if t.get("enable_moe_block"): t.update(moe_intermediate_size=32, num_experts=4, top_k_experts=2)
    if t.get("hidden_size_per_layer_input"): t.update(hidden_size_per_layer_input=8)
    for key in ("vision_config", "audio_config"):
        m = d.get(key)
        if not m or m.get("model_type") == "gemma4_unified_audio": continue   # 12B audio: raw samples (640 per token), no layers
        if m.get("model_type") == "gemma4_audio": m["num_hidden_layers"] = 1; continue   # E4B conformer: widths are tied to the mel bins
        for k, v in dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                         head_dim=16, global_head_dim=16, mm_embed_dim=32, output_proj_dims=32, audio_embed_dim=32).items():
            if k in m: m[k] = v
    return d


@pytest.fixture(scope="module", params=list(BASES))
def tiny(request, tmp_path_factory):
    """A tiny random-weight copy of one Gemma 4 architecture, saved the way the Hub stores the real one (the multimodal
    *ForConditionalGeneration class), with the real tokenizer."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    base, revision, layer_types, kv_shared = BASES[request.param]
    cfg = AutoConfig.from_pretrained(base, revision=revision)
    cfg = type(cfg)(**shrink(cfg.to_dict(), layer_types, kv_shared))
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp(f"gemma4-{request.param}")
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    AutoTokenizer.from_pretrained(base, revision=revision).save_pretrained(path)
    try:                                                                 # the media extra (torchvision) brings the processor
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(base, revision=revision).save_pretrained(path)
    except (ImportError, ModuleNotFoundError):
        pass
    return str(path)


REC = {"state": "Order 4411 arrived two weeks late and the box was crushed. Two charges of $38.20 appear on the card, "
                "and the customer has written three times without a reply from support.",
       "questions": [{"instr": "Is there a billing problem?", "options": ["yes", "no"], "label": 0},
                     {"instr": "Which team should handle this?", "options": ["returns", "shipping", "billing", "other"], "label": 2},
                     {"instr": "How upset is the customer?", "options": ["calm", "annoyed", "furious"], "label": 1}]}


def close(xs, ys, tol=1e-5):
    return all((a - b).abs().max() < tol for a, b in zip(xs, ys, strict=True))


def test_encode_starts_with_bos_and_escapes_control_tokens(tiny):
    from kev.model import GEMMA_SPECIAL, encode, family, load_tokenizer, user_tokens
    tok = load_tokenizer(tiny); fam = family(tok)
    assert fam["bos"] == [tok.bos_token_id] and fam["delims"] == tok.convert_tokens_to_ids(GEMMA_SPECIAL)
    enc = encode(tok, REC)
    assert enc["ids"][:2] == [tok.bos_token_id, fam["delims"][0]] and all(enc["ids"][d] == fam["delims"][4] for d in enc["decide_idx"])
    forged = user_tokens(tok, "a <bos> b <|turn>user <|image|> <unused3> <|fim_prefix|>")
    assert not set(forged) & (set(tok.all_special_ids) | set(fam["delims"]))
    assert encode(tok, {**REC, "state": "x " * 1000})["seg"].count(0) == 384          # <bos> counts toward the state limit


def test_packed_mask_matches_rows_and_isolates_questions(tiny):
    """The packed form (per-layer-type dict mask, window by position) must reproduce the causal rows, where transformers
    applies the window itself, and a question's answer must not depend on the other questions."""
    from kev.model import DecisionModel, load_tokenizer
    tok = load_tokenizer(tiny); m = DecisionModel(tiny, tok, "cpu").eval()
    assert m.window == WINDOW and not m.hybrid and not m.rows_form([m.encode(tok, REC)])
    enc = m.encode(tok, REC); assert enc["seg"].count(0) > 2 * WINDOW
    with torch.no_grad():
        packed = [torch.softmax(z, -1) for z in m.forward(enc)]
        rows = [torch.softmax(z, -1) for z in m.forward_rows_batch([enc])[0]]
        alone = [m.probs(m.encode(tok, {"state": REC["state"], "questions": [q]}))[0] for q in REC["questions"]]
        batched = m.forward_batch([enc, m.encode(tok, {**REC, "questions": REC["questions"][:1]})])[0]   # right padding
    assert close(packed, rows) and close(packed, alone) and close(packed, [torch.softmax(z, -1) for z in batched])


def test_window_is_needed_for_parity(tiny):
    """Guards the test above: without the window the packed pass differs from the rows, so the dict mask is doing work."""
    from kev.model import DecisionModel, load_tokenizer
    tok = load_tokenizer(tiny); m = DecisionModel(tiny, tok, "cpu").eval(); enc = m.encode(tok, REC)
    with torch.no_grad():
        rows = [torch.softmax(z, -1) for z in m.forward_rows_batch([enc])[0]]
        m.window = None; unwindowed = [torch.softmax(z, -1) for z in m.forward(enc)]
    assert not close(rows, unwindowed)


def test_prefix_cache_matches_full_pass(tiny, monkeypatch):
    """Serving: the state is cached once (full-length keys, cropped back after each use) and reused, packed or as rows."""
    from kev import model as M
    tok = M.load_tokenizer(tiny); m = M.DecisionModel(tiny, tok, "cpu").eval(); enc = m.encode(tok, REC)
    with torch.no_grad():
        full = m.probs(enc)
        miss, prefix = m.probs_and_prefix(enc)
        hit, hit2 = m.probs_with_prefix(enc, prefix), m.probs_with_prefix(enc, prefix)   # twice: crop() must restore the cache
        fresh = m.prefix(enc); hit3 = m.probs_with_prefix(enc, fresh)
        fewer = {**REC, "questions": REC["questions"][1:]}; e2 = m.encode(tok, fewer)
        other = m.probs_with_prefix(e2, prefix)
        monkeypatch.setattr(M, "SERVE_MAX_PACKED", len(enc["ids"]) - 1)                 # force the row form
        assert m.rows_form([enc])
        rows_hit = m.probs_with_prefix(enc, prefix)
        monkeypatch.setattr(M, "rows_per_pass", lambda rows, prefix_len=0, budget=0: 1)
        one_at_a_time = m.probs_with_prefix(enc, prefix)
    assert close(full, miss) and close(full, hit, 1e-4) and close(full, hit2, 1e-4) and close(full, hit3, 1e-4)
    assert close(m.probs(e2), other, 1e-4) and close(full, rows_hit, 1e-4) and close(full, one_at_a_time, 1e-4)


def test_lora_reaches_attention_and_dense_mlp_and_trains(tiny):
    """The adapter hits q/k/v/o (v only where a layer has one: k=v global layers and shared-KV layers do not) and the dense
    MLP; fused experts, the router and per-layer-embedding projections stay frozen. One checkpointed bf16-weights step
    reaches the adapter and the head."""
    from kev.model import DecisionModel, load_tokenizer
    tok = load_tokenizer(tiny); m = DecisionModel(tiny, tok, "cpu", lora=4, dtype=torch.bfloat16)
    names = {n for n, p in m.lm.named_parameters() if p.requires_grad}
    hit = {n.split(".lora_")[0].split(".")[-1] for n in names}
    assert hit == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert not any(".experts." in n or ".router." in n or "per_layer" in n for n in names)
    m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.train()
    enc = m.encode(tok, REC)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = sum(torch.nn.functional.cross_entropy(z.float()[None], torch.tensor([q["label"]])) for z, q in zip(m.forward(enc), REC["questions"]))
    loss.backward()
    grads = [p.grad for n, p in m.lm.named_parameters() if p.requires_grad and "lora_B" in n]
    assert torch.isfinite(loss) and m.head.q.weight.grad.abs().sum() > 0 and all(g is not None for g in grads)


# --- media: images, video frames and audio in the state ------------------------------------------------------------------

def _long():
    from kev.model import training_context      # media alone is hundreds of tokens; training lifts the limit the same way (--max_state)
    return {k: v for k, v in training_context(3000).items() if k != "max_packed"}


LONG = _long()


def media_rec(seed=0, image_hw=(120, 160), audio_s=1.0, kinds=("image", "video", "audio")):
    import numpy as np
    from PIL import Image
    r = np.random.default_rng(seed)
    items = {"image": {"type": "image", "data": Image.fromarray((r.random((*image_hw, 3)) * 255).astype("uint8"))},
             "video": {"type": "video", "data": (r.random((8, 64, 96, 3)) * 255).astype("uint8"), "metadata": {"fps": 4, "total_num_frames": 8}, "num_frames": 4},
             "audio": {"type": "audio", "data": (r.standard_normal(int(16000 * audio_s)) * 0.1).astype("float32")}}
    return {**REC, "media": [items[k] for k in kinds]}


def media_model(tiny):
    pytest.importorskip("torchvision")
    from kev.model import DecisionModel, load_tokenizer
    tok = load_tokenizer(tiny); m = DecisionModel(tiny, tok, "cpu").eval()
    kinds = ("image", "video") + (("audio",) if m.mm.config.audio_config is not None else ())
    return tok, m, kinds


def test_media_state_matches_gemma_reference(tiny):
    """The state with media, under Rev's mask (image and frame blocks where the base uses them, window by position), gives
    the same hidden states as Gemma's own multimodal forward on the same tokens, where transformers builds the masks from
    mm_token_type_ids with each family's rule."""
    tok, m, kinds = media_model(tiny)
    enc = m.encode(tok, media_rec(kinds=kinds), **LONG)
    Ls = enc["seg"].count(0); items = enc["media"]
    assert enc["ids"][2] == tok.convert_tokens_to_ids("<|image>") and max(enc["blk"]) >= 4 and sum(b >= 0 for b in enc["blk"]) > 2 * WINDOW
    from kev.model import collate_media
    types = [0, 0] + [t for it in items for t in it["types"]]; types += [0] * (Ls - len(types))
    with torch.no_grad():
        ours = m.hidden(enc)[:Ls]
        ref = m.mm(input_ids=torch.tensor([enc["ids"][:Ls]]), mm_token_type_ids=torch.tensor([types]),
                   **collate_media(items, "cpu", torch.float32)).last_hidden_state[0]
        no_blocks = m.hidden({**enc, "blk": [-1] * len(enc["blk"])})[:Ls]
    assert (ours - ref).abs().max() < 1e-4 * ref.abs().max()
    if m.lm.config.use_bidirectional_attention == "vision":                # E2B/E4B keep images causal: no blocks to test
        assert (no_blocks - ref).abs().max() > 1e-3 * ref.abs().max()  # the blocks are doing work


def test_media_paths_agree_and_isolate(tiny, monkeypatch):
    """Packed pass, prefix miss/hit, forced row form, one question alone, and a batch of two records whose items differ in
    size (collate padding) all give the same answers."""
    from kev import model as M
    tok, m, kinds = media_model(tiny)
    rec = media_rec(kinds=kinds); enc = m.encode(tok, rec, **LONG)
    other = media_rec(seed=1, image_hw=(200, 120), audio_s=0.6, kinds=kinds); enc2 = m.encode(tok, other, **LONG)
    with torch.no_grad():
        full = m.probs(enc)
        miss, prefix = m.probs_and_prefix(enc); hit = m.probs_with_prefix(enc, prefix)
        alone = [m.probs(m.encode(tok, {**rec, "questions": [q]}, **LONG))[0] for q in rec["questions"]]
        batched = m.forward_batch([enc2, enc])
        full2 = m.probs(enc2)
        monkeypatch.setattr(M, "SERVE_MAX_PACKED", len(enc["ids"]) - 1)
        rows = m.probs(enc)
    assert close(full, miss, 1e-4) and close(full, hit, 1e-4) and close(full, alone, 1e-4) and close(full, rows, 1e-4)
    soft = lambda zs: [torch.softmax(z, -1) for z in zs]
    assert close(soft(batched[1]), full, 1e-4) and close(soft(batched[0]), full2, 1e-4)
    assert not close(full, full2, 1e-3)                                    # the media change the answer


def test_media_trains_with_frozen_encoders(tiny):
    tok, m, kinds = media_model(tiny)
    from kev.model import DecisionModel
    m = DecisionModel(tiny, tok, "cpu", lora=4); m.train()
    enc = m.encode(tok, media_rec(kinds=kinds), **LONG)
    loss = sum(torch.nn.functional.cross_entropy(z[None], torch.tensor([q["label"]])) for z, q in zip(m.forward(enc), REC["questions"]))
    loss.backward()
    lm_ids = {id(p) for p in m.lm.parameters()}
    towers = [p for p in m.mm.parameters() if id(p) not in lm_ids]
    assert towers and all(p.grad is None and not p.requires_grad for p in towers)
    assert all(p.grad is not None for n, p in m.lm.named_parameters() if p.requires_grad and "lora_B" in n)


def test_media_key_tells_same_sized_images_apart(tiny):
    """Two images of the same size have identical placeholder tokens; the serving prefix cache keys on media_key too."""
    from kev.model import media_key
    tok, m, _ = media_model(tiny)
    a, b = (m.encode(tok, media_rec(seed=s, kinds=("image",)), **LONG) for s in (0, 1))
    assert a["ids"] == b["ids"] and media_key(a) != media_key(b) and media_key(a) == media_key(m.encode(tok, media_rec(seed=0, kinds=("image",)), **LONG))
    assert media_key(m.encode(tok, REC)) is None
