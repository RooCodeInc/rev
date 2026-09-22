"""Gemma 4 backbones (attention-only, sliding-window + global layers, MoE block beside a dense MLP): encoding, the
per-layer-type packed mask, prefix cache and isolation, on a tiny random-weight model with the real 26B-A4B architecture
flags and tokenizer. Downloads the Gemma 4 tokenizer and config only (no weights); not run in CI.
Run: uv run --extra serve python -m pytest tests/test_gemma4.py -q
"""
import pytest
import torch

BASE, REVISION = "google/gemma-4-26B-A4B", "24548b62aa021d562695c04aaf7758a1ea47990b"
WINDOW = 16   # far below the states and rows below, so every local layer drops keys


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A 4-layer Gemma 4 (3 sliding, 1 global with k=v attention, 4 experts top-2) saved as a multimodal checkpoint,
    the way the Hub stores the real one."""
    from transformers import AutoConfig, AutoTokenizer, Gemma4Config, Gemma4ForConditionalGeneration
    d = AutoConfig.from_pretrained(BASE, revision=REVISION).to_dict()
    d["text_config"].pop("per_layer_config", None)                        # derived from layer_types; rebuilt for 4 layers
    d["text_config"].update(hidden_size=64, intermediate_size=64, moe_intermediate_size=32, num_attention_heads=4, num_key_value_heads=2,
                            num_global_key_value_heads=1, head_dim=16, global_head_dim=32, num_hidden_layers=4, num_experts=4, top_k_experts=2,
                            layer_types=["sliding_attention"] * 3 + ["full_attention"], sliding_window=WINDOW)
    d["vision_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16, global_head_dim=16)
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("gemma4-tiny")
    Gemma4ForConditionalGeneration(Gemma4Config(**d)).save_pretrained(path)
    AutoTokenizer.from_pretrained(BASE, revision=REVISION).save_pretrained(path)
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
    """The adapter hits q/k/v/o (v only on local layers: global ones share k and v) and the dense MLP; the fused experts
    and the router stay frozen. One checkpointed bf16-weights step reaches the adapter and the head."""
    from kev.model import DecisionModel, load_tokenizer
    tok = load_tokenizer(tiny); m = DecisionModel(tiny, tok, "cpu", lora=4, dtype=torch.bfloat16)
    names = {n for n, p in m.lm.named_parameters() if p.requires_grad}
    hit = {n.split(".lora_")[0].split(".")[-1] for n in names}
    assert hit == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert not any(".experts." in n or ".router." in n for n in names)
    m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.train()
    enc = m.encode(tok, REC)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = sum(torch.nn.functional.cross_entropy(z.float()[None], torch.tensor([q["label"]])) for z, q in zip(m.forward(enc), REC["questions"]))
    loss.backward()
    grads = [p.grad for n, p in m.lm.named_parameters() if p.requires_grad and "lora_B" in n]
    assert torch.isfinite(loss) and m.head.q.weight.grad.abs().sum() > 0 and all(g is not None for g in grads)
