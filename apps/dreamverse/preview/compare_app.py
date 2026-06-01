#!/usr/bin/env python3
"""OmniNFT / dreamverse comparison website (Gradio).

Two kinds of comparison sets, shown in one searchable, paginated flow
(dreamverse sets sorted FIRST):

1. dreamverse sets: ``results/dreamverse_<preset_id>/<condition>/`` — one prompt
   per dir; each condition subdir holds 1+ ``.mp4`` plus ``meta.json`` /
   ``prompt.txt`` / ``run.log``. Conditions are discovered GENERICALLY at scan
   time (nothing hardcoded), grouped by model (LTX-2 vs LTX-2.3), and rendered
   as a per-model gallery that wraps to however many videos exist — the LTX-2
   and LTX-2.3 counts need NOT be equal, and a condition may itself contain
   several clips.

2. 4-way OmniNFT sets: ``results/compare*/<sub>/*.mp4`` — the original
   fixed quad (ltx2_base, ltx2_omninft, ltx23_base, ltx23_omninft), one .mp4
   each + a prompt.txt.

CONDITION NAMING CONVENTION (also documented in apps/dreamverse/docs/website.md):

    <model>_<variant>[_s06|_s08|_s10]

    model    : ltx2  -> "LTX-2",   ltx23 -> "LTX-2.3"  (the only two groups)
    variant  : base | omninft | omninft_iclora_union | <anything>
    strength : optional trailing _sNN suffix => LoRA strength NN/10
               (_s06 -> x0.6, _s08 -> x0.8, _s10 -> x1.0). A bare omninft
               variant with no suffix is treated as x1.0.

A ``meta.json`` (fields: id, label, description, condition, [model],
[lora_strength]) is parsed when present and takes precedence; otherwise the
subdir name is parsed into model + a human label by the convention above. New
condition names (e.g. ``ltx23_omninft_iclora_union_s08``) therefore render with
a sensible auto-derived label without any code change.

Scaling: videos are NOT all loaded at once. One prompt is shown per page; only
that page's clips are sent to the browser, via the same demo.load()/event path
that gr.set_static_paths + allowed_paths serve from disk (progressive faststart
streaming preserved). Search filters the prompt list before paging.
"""

import glob
import html
import json
import os
import re

import gradio as gr

ROOT = "/home/hal-kaiqin/dreamverse/OmniNFT"
RESULTS = os.path.join(ROOT, "results")

# How many videos a single page can display. A dreamverse prompt can carry the
# full OmniNFT x IC-LoRA intensity matrix across LTX-2 + LTX-2.3 (currently 35
# cells on the most-populated prompt), and conditions/clips keep growing, so
# keep generous headroom. The renderer pre-builds this many cards (shown/hidden
# per page); if a prompt ever exceeds it, page_view surfaces a visible warning
# instead of silently dropping cards (see overflow note below).
MAX_VIDEOS = 64

# Model grouping. Maps the leading model token of a condition subdir to a
# (group_key, human_label, audio_note). The ORDER here is the display order of
# the model groups within a prompt (LTX-2 above LTX-2.3).
MODEL_GROUPS = [
    ("ltx2", "LTX-2", "FastVideo/LTX2-Distilled-Diffusers · audio 24 kHz"),
    ("ltx23", "LTX-2.3", "FastVideo/LTX-2.3-Distilled-Diffusers · audio 48 kHz"),
]
MODEL_LABEL = {k: lbl for k, lbl, _ in MODEL_GROUPS}
MODEL_ORDER = {k: i for i, (k, _, _) in enumerate(MODEL_GROUPS)}

# 4-way OmniNFT comparison panels: subdir -> (label, audio caption).
OMNINFT_SUBDIRS = [
    ("ltx2_base", "LTX-2 base", "audio 24 kHz"),
    ("ltx2_omninft", "LTX-2 + OmniNFT", "audio 24 kHz"),
    ("ltx23_base", "LTX-2.3 base", "audio 48 kHz"),
    ("ltx23_omninft", "LTX-2.3 + OmniNFT", "audio 48 kHz"),
]

GUITAR_PROMPT = (
    "A man plays acoustic guitar on a wooden stage, warm applause from the audience"
)


# --------------------------------------------------------------------------- #
# Condition-name parsing (the generic, future-proof bit).
# --------------------------------------------------------------------------- #
# Per-model audio sample rate (used in card footers).
MODEL_AUDIO_RATE = {"ltx2": "24 kHz", "ltx23": "48 kHz"}


def parse_condition(name, meta=None):
    """Derive a structured config from a condition subdir name, preferring
    meta.json fields when present.

    Returns a dict with everything a card footer needs:
      model, model_label, variant, strength, label, audio_note,
      lora_name      -> "base" | "OmniNFT" | "OmniNFT + IC-LoRA …" etc.
      strength_text  -> "" (base) | "x0.8" | "x1.0"
      audio_rate     -> "24 kHz" / "48 kHz"
      ic_lora        -> meta ic_lora (or None)
      ic_strength    -> meta ic_strength (or None)
      extras         -> ordered list of (key, value) pulled from meta for the
                        card footer (omninft_strength, base_lora, note, …).
    `strength` is a float used for ordering (base -> 0.0; bare omninft -> 1.0).
    """
    meta = meta or {}
    # Model: meta first, else leading token of the subdir name.
    model = meta.get("model")
    rest = name
    if not model:
        # Match the longest known model prefix (ltx23 before ltx2).
        for key in sorted(MODEL_LABEL, key=len, reverse=True):
            if name == key or name.startswith(key + "_"):
                model = key
                rest = name[len(key):].lstrip("_")
                break
    else:
        # Strip the model token from the name to recover the variant tail.
        if name.startswith(model + "_"):
            rest = name[len(model) + 1:]
        elif name == model:
            rest = ""
    if not model:
        model = "other"
    model_label = MODEL_LABEL.get(model, model.upper())

    # Strength: meta.lora_strength / omninft_strength wins; else trailing _sNN;
    # else infer.
    strength = meta.get("lora_strength")
    if strength is None:
        strength = meta.get("omninft_strength")
    m = re.search(r"_s(\d{2})$", rest)
    if m:
        if strength is None:
            strength = int(m.group(1)) / 10.0
        rest = rest[: m.start()]
    rest = rest or "base"

    is_base = rest == "base" or rest == ""
    if is_base:
        strength = 0.0
    elif strength is None:
        # A LoRA variant with no explicit strength == native x1.0.
        strength = 1.0
    strength = float(strength)

    ic_lora = meta.get("ic_lora")
    ic_strength = meta.get("ic_strength")
    audio_rate = MODEL_AUDIO_RATE.get(model, "?")

    # Human variant label + structured LoRA list. `loras` is a list of
    # (name, strength) tuples — one entry per LoRA in the stack (OmniNFT, plus
    # any IC-LoRA) — which the card footer renders as a bullet list.
    if is_base:
        variant_label = "base"
        audio_note = "no LoRA"
        lora_name = "base"
        strength_text = ""
        loras = []
    else:
        pretty = rest.replace("_", " ").replace("omninft", "OmniNFT")
        strength_text = f"x{strength:.1f}"
        variant_label = f"{pretty} {strength_text}"
        audio_note = f"OmniNFT LoRA strength {strength:g}"
        # Build a readable LoRA-stack name (kept for search/labels).
        parts = ["OmniNFT"] if "OmniNFT" in pretty or "omninft" in rest else [pretty]
        if ic_lora:
            parts.append(f"IC-LoRA ({ic_lora})")
        elif "union" in rest or "iclora" in rest:
            parts.append("IC-LoRA")
        lora_name = " + ".join(parts)
        # Structured (name, strength) list for the footer. An explicit meta
        # `loras` list wins (supports ANY number of stacked LoRAs, e.g. OmniNFT
        # + Pixar + Transition → three bars); else derive from the other meta
        # fields (base_lora / omninft_strength / ic_lora / ic_strength); else
        # fall back to name-parsing.
        loras = []
        meta_loras = meta.get("loras")
        if isinstance(meta_loras, list) and meta_loras:
            try:
                loras = [(str(n), float(s)) for n, s in meta_loras]
            except (TypeError, ValueError):
                loras = []
        if not loras:
            # Derive: OmniNFT base (+ optional single IC/style LoRA).
            if meta.get("base_lora") == "omninft" or "OmniNFT" in pretty \
                    or "omninft" in rest:
                loras.append(("OmniNFT",
                              float(meta.get("omninft_strength", strength))))
            else:
                loras.append((pretty, strength))
            if ic_lora:
                loras.append((ic_lora,
                              float(ic_strength) if ic_strength is not None else 1.0))
            elif any(t in rest for t in ("union", "iclora", "detailer",
                                         "motiontrack", "hdr", "lipdub")):
                loras.append(("IC-LoRA",
                              float(ic_strength) if ic_strength is not None else 1.0))

    label = f"{model_label} · {variant_label}"

    # Extra meta fields worth surfacing in the card footer (ordered, deduped).
    extras = []
    seen = {"id", "label", "description", "condition", "model", "lora_strength",
            "omninft_strength", "ic_lora", "ic_strength"}
    for k in ("base_lora", "note"):
        if meta.get(k) and k not in (e[0] for e in extras):
            extras.append((k, meta[k]))
            seen.add(k)
    for k, v in meta.items():
        if k not in seen and v not in (None, ""):
            extras.append((k, v))

    return {
        "model": model,
        "model_label": model_label,
        "variant": rest,
        "strength": strength,
        "label": label,
        "audio_note": audio_note,
        "lora_name": lora_name,
        "loras": loras,
        "strength_text": strength_text,
        "audio_rate": audio_rate,
        "ic_lora": ic_lora,
        "ic_strength": ic_strength,
        "extras": extras,
    }


def _condition_sort_key(cond):
    return (MODEL_ORDER.get(cond["model"], 99), cond["strength"], cond["variant"])


# --------------------------------------------------------------------------- #
# Scanning.
# --------------------------------------------------------------------------- #
def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _scan_dreamverse():
    """One dict per dreamverse prompt dir, with conditions discovered on disk.

    Each set: {dir, kind, id, label, description, prompt, models:[(model_label,
    audio_note, [cell, ...])], n_videos, search_text}, where each `cell` is a
    dict {label, path, model_label, lora_name, strength_text, audio_rate,
    ic_lora, ic_strength, extras} ready for a card footer.
    """
    sets = []
    for set_dir in sorted(glob.glob(os.path.join(RESULTS, "dreamverse_*"))):
        if not os.path.isdir(set_dir):
            continue
        base = os.path.basename(set_dir)
        pid = base[len("dreamverse_"):]

        conditions = []  # list of dicts with parsed info + clip list
        meta_any = {}
        prompt = None
        for sub in sorted(os.listdir(set_dir)):
            sub_path = os.path.join(set_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            mp4s = sorted(glob.glob(os.path.join(sub_path, "*.mp4")))
            if not mp4s:
                continue
            meta = _read_json(os.path.join(sub_path, "meta.json")) or {}
            if meta and not meta_any:
                meta_any = meta
            if prompt is None:
                prompt = _read_text(os.path.join(sub_path, "prompt.txt"))
            info = parse_condition(sub, meta)
            info["clips"] = mp4s
            conditions.append(info)

        if not conditions:
            continue  # set still empty / generating

        label = meta_any.get("label") or pid.replace("_", " ").title()
        description = meta_any.get("description") or ""

        # Group conditions by model, in MODEL_GROUPS order, each sorted by
        # strength. Counts per model may differ; that's fine.
        by_model = {}
        for c in sorted(conditions, key=_condition_sort_key):
            by_model.setdefault(c["model"], []).append(c)

        def _cells_for(conds):
            out = []
            for c in conds:
                multi = len(c["clips"]) > 1
                for i, mp4 in enumerate(c["clips"]):
                    cl = c["label"] + (f" [{i + 1}]" if multi else "")
                    out.append({
                        "label": cl,
                        "path": mp4,
                        "model_label": c["model_label"],
                        "lora_name": c["lora_name"],
                        "loras": c["loras"],
                        "strength_text": c["strength_text"],
                        "audio_rate": c["audio_rate"],
                        "ic_lora": c["ic_lora"],
                        "ic_strength": c["ic_strength"],
                        "extras": c["extras"],
                    })
            return out

        models = []
        n_videos = 0
        for key, model_label, audio_note in MODEL_GROUPS:
            conds = by_model.pop(key, [])
            if not conds:
                continue
            cells = _cells_for(conds)
            n_videos += len(cells)
            models.append((model_label, audio_note, cells))
        # Any unknown model groups (model == "other" or future tokens) go last.
        for key, conds in by_model.items():
            model_label = conds[0]["model_label"]
            cells = _cells_for(conds)
            n_videos += len(cells)
            models.append((model_label, "", cells))

        search_text = " ".join(
            [pid, label, description, prompt or ""]
        ).lower()
        sets.append({
            "dir": set_dir, "kind": "dreamverse", "id": pid,
            "label": label, "description": description,
            "prompt": prompt or "(prompt.txt not found)",
            "models": models, "n_videos": n_videos,
            "search_text": search_text,
        })
    return sets


def _scan_omninft():
    """4-way OmniNFT compare* sets (strict: needs all 4 subdirs)."""
    sets = []
    for set_dir in glob.glob(os.path.join(RESULTS, "compare*")):
        if not os.path.isdir(set_dir):
            continue
        cells = []
        complete = True
        prompt = None
        for sub, label, aud in OMNINFT_SUBDIRS:
            mp4s = sorted(glob.glob(os.path.join(set_dir, sub, "*.mp4")))
            if not mp4s:
                complete = False
                break
            info = parse_condition(sub)
            cells.append({
                "label": label,
                "path": mp4s[0],
                "model_label": info["model_label"],
                "lora_name": info["lora_name"],
                "loras": info["loras"],
                "strength_text": info["strength_text"],
                "audio_rate": aud.replace("audio ", ""),
                "ic_lora": None,
                "ic_strength": None,
                "extras": [],
            })
            if prompt is None:
                prompt = _read_text(os.path.join(set_dir, sub, "prompt.txt"))
        if not complete:
            continue
        base = os.path.basename(set_dir)
        if base == "compare":
            prompt = prompt or GUITAR_PROMPT
        prompt = prompt or base
        # Single "model group" so the renderer is uniform.
        models = [("4-way OmniNFT (official LTX-2 / LTX-2.3)", "", cells)]
        sets.append({
            "dir": set_dir, "kind": "omninft", "id": base,
            "label": base, "description": "",
            "prompt": prompt, "models": models, "n_videos": len(cells),
            "search_text": (base + " " + (prompt or "")).lower(),
        })
    return sets


def _sort_key(s):
    """dreamverse FIRST (by label), then compare_ex<N> by N, bare compare last."""
    name = os.path.basename(s["dir"])
    if s["kind"] == "dreamverse":
        return (0, 0, s["label"].lower())
    m = re.search(r"(\d+)$", name)
    if m:
        return (1, int(m.group(1)), "")
    return (2, 0, "")


def scan_sets():
    sets = _scan_dreamverse() + _scan_omninft()
    sets.sort(key=_sort_key)
    return sets


# --------------------------------------------------------------------------- #
# Filtering + paging.
# --------------------------------------------------------------------------- #
def model_choices(sets):
    keys = set()
    for s in sets:
        for model_label, _aud, _cells in s["models"]:
            keys.add(model_label)
    return ["All models"] + sorted(keys)


def filter_sets(sets, query, model):
    q = (query or "").strip().lower()
    out = []
    for s in sets:
        if q and q not in s["search_text"]:
            continue
        if model and model != "All models":
            if not any(ml == model for ml, _a, _c in s["models"]):
                continue
        out.append(s)
    return out


def truncate(text, n=70):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def dropdown_choices(sets):
    out = []
    for i, s in enumerate(sets):
        tag = "dreamverse · " if s["kind"] == "dreamverse" else ""
        title = s["label"] if s["kind"] == "dreamverse" else s["prompt"]
        out.append((f"{i + 1}. {tag}{truncate(title)}", i))
    return out


# --------------------------------------------------------------------------- #
# Rendering. The page paints heading + a copyable prompt box + a responsive
# wrapping grid of CARDS. Each card = one video on top with its config shown in
# a footer at the bottom (model, LoRA name, intensity, audio rate, + any useful
# meta fields). To keep a fixed component graph (Gradio needs that), we
# pre-build MAX_VIDEOS cards (a model-group header markdown + a video + a footer
# markdown each), and show/hide per page.
# --------------------------------------------------------------------------- #
def _empty_card():
    # (group_header, video, footer)
    return (
        gr.update(value="", visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
    )


def _footer_html(cell):
    """Bottom-of-card config: just `model: …` and a `LoRA:` bullet list (each
    LoRA with its strength). No title, no description prose."""
    model_row = (f"<div class='cfg-row'><span class='cfg-k'>model</span>"
                 f"<span class='cfg-v'>{html.escape(cell['model_label'])}</span></div>")
    loras = cell.get("loras") or []
    if not loras:
        lora_block = ("<div class='cfg-row'><span class='cfg-k'>LoRA</span>"
                      "<span class='cfg-v'>base (none)</span></div>")
    else:
        items = []
        for name, s in loras:
            s = float(s)
            pct = max(0.0, min(s, 1.0)) * 100.0  # bar scales 0..1, clamps
            items.append(
                "<div class='lora-item'>"
                f"<div class='lora-top'><span class='lora-nm'>{html.escape(str(name))}</span>"
                f"<span class='lora-val'>{s:.1f}</span></div>"
                "<span class='lora-bar'>"
                f"<span class='lora-fill' style='width:{pct:.0f}%'></span></span>"
                "</div>")
        lora_block = (f"<div class='cfg-lora-h'>LoRA</div>"
                      f"<div class='cfg-lora'>{''.join(items)}</div>")
    return f"<div class='cfg-footer'>{model_row}{lora_block}</div>"


def page_view(sets, idx):
    """Returns: heading_md, prompt_textbox_update, indicator_md,
    then MAX_VIDEOS * (group_header_update, video_update, footer_update).

    Each card carries (optional) a model-group header above the first card of a
    group, the video, and a config footer below it. Cards wrap into a
    responsive grid via CSS (elem_classes), so the unequal LTX-2 vs LTX-2.3
    counts and multi-clip conditions lay out cleanly.
    """
    n = len(sets)
    if n == 0:
        flat = []
        for _ in range(MAX_VIDEOS):
            flat += list(_empty_card())
        return (
            "## No comparison sets match.\n\nAdjust the search/filter or click Refresh.",
            gr.update(value=""),
            "0 / 0",
            *flat,
        )

    idx = max(0, min(idx, n - 1))
    s = sets[idx]

    if s["kind"] == "dreamverse":
        heading = (
            f"## {s['label']}  \n"
            f"<small>dreamverse 30s pipeline · prompt id `{s['id']}` · "
            f"{s['n_videos']} clip(s) · prompt {idx + 1} / {n}</small>"
        )
        if s["description"]:
            heading += f"\n\n{s['description']}"
        if s["n_videos"] > MAX_VIDEOS:
            heading += (f"\n\n> ⚠️ This prompt has {s['n_videos']} clips but the "
                        f"page can show {MAX_VIDEOS}; {s['n_videos'] - MAX_VIDEOS} "
                        "are not displayed. Raise MAX_VIDEOS in compare_app.py.")
    else:
        heading = (
            f"## 4-way OmniNFT comparison  \n"
            f"<small>set `{s['id']}` · prompt {idx + 1} / {n}</small>"
        )

    indicator = f"{idx + 1} / {n}"

    # Flatten model groups into the fixed card list. A group header is attached
    # to the FIRST card of each model group so the LTX-2 / LTX-2.3 boundary
    # stays visible even though the cards wrap into one uniform grid.
    flat_cells = []  # (group_header_html_or_"", cell_dict)
    for model_label, audio_note, cells in s["models"]:
        for j, cell in enumerate(cells):
            header = ""
            if j == 0:
                note = f" — {html.escape(audio_note)}" if audio_note else ""
                header = (f"<div class='grp-header'>{html.escape(model_label)}"
                          f"<span class='grp-note'>{note}</span></div>")
            flat_cells.append((header, cell))

    flat = []
    for i in range(MAX_VIDEOS):
        if i < len(flat_cells):
            header, cell = flat_cells[i]
            flat += [
                gr.update(value=header, visible=bool(header)),
                gr.update(value=cell["path"], label=None, visible=True),
                gr.update(value=_footer_html(cell), visible=True),
            ]
        else:
            flat += list(_empty_card())

    return (heading, gr.update(value=s["prompt"]), indicator, *flat)


CARD_CSS = """
/* Responsive wrapping grid of video cards. */
#card-grid { display: flex; flex-wrap: wrap; gap: 16px; align-items: stretch; }
/* Each card is a fixed-min-width flex item with a visible border + bg. */
.video-card {
    flex: 1 1 300px; max-width: 380px; min-width: 280px;
    display: flex; flex-direction: column;
    border: 1px solid var(--border-color-primary, #d0d5dd);
    border-radius: 12px; background: var(--block-background-fill, #fff);
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); overflow: hidden; padding: 8px;
}
.video-card video { border-radius: 8px; width: 100%; }
/* Model-group header spans a full row (forces a line break in the flex grid). */
.grp-header-card { flex-basis: 100%; border: none; box-shadow: none;
    background: transparent; padding: 0; margin-top: 4px; }
.grp-header { font-weight: 700; font-size: 1.1em; padding: 6px 2px;
    border-bottom: 2px solid var(--border-color-accent, #6b7280); }
.grp-note { font-weight: 400; font-size: 0.85em; opacity: 0.7; }
/* Config footer at the BOTTOM of each card: `model: …` + a LoRA bullet list. */
.cfg-footer { margin-top: 8px; padding-top: 8px; font-size: 0.9em;
    line-height: 1.45;
    border-top: 1px dashed var(--border-color-primary, #d0d5dd); }
.cfg-row { display: flex; gap: 6px; }
.cfg-k { opacity: 0.6; min-width: 44px; }
.cfg-v { font-weight: 600; }
.cfg-lora-h { opacity: 0.6; margin-top: 4px; }
/* Each LoRA = name + value on top, a filled intensity bar below. */
.cfg-lora { margin-top: 4px; display: flex; flex-direction: column; gap: 7px; }
.lora-item { display: flex; flex-direction: column; gap: 3px; }
.lora-top { display: flex; justify-content: space-between; gap: 8px;
    align-items: baseline; }
.lora-nm { font-weight: 600; }
.lora-val { font-weight: 700; font-variant-numeric: tabular-nums; opacity: 0.85; }
.lora-bar { height: 7px; border-radius: 999px; overflow: hidden;
    background: var(--background-fill-secondary, #e9ecef); }
.lora-fill { display: block; height: 100%; border-radius: 999px;
    background: linear-gradient(90deg,
        var(--color-accent, #2563eb), var(--color-accent, #2563eb)); }
"""


def build():
    initial_all = scan_sets()
    initial_filtered = initial_all

    with gr.Blocks(title="OmniNFT / dreamverse comparison", css=CARD_CSS) as demo:
        all_state = gr.State(initial_all)        # every set on disk
        view_state = gr.State(initial_filtered)  # filtered subset (paged)
        page_state = gr.State(0)

        gr.Markdown(
            "# OmniNFT — dreamverse video comparison\n"
            "<small>dreamverse 30s pipeline sets first (OmniNFT LoRA-strength "
            "sweeps on FastVideo LTX-2 & LTX-2.3 Distilled), then 4-way OmniNFT "
            "sets (official Lightricks LTX-2 / LTX-2.3). Conditions are "
            "discovered dynamically from disk — see "
            "<code>apps/dreamverse/docs/website.md</code>.</small>"
        )

        with gr.Row():
            search = gr.Textbox(
                label="Search prompts", placeholder="filter by text / id / label…",
                scale=4, container=True,
            )
            model_filter = gr.Dropdown(
                label="Model", choices=model_choices(initial_all),
                value="All models", scale=1,
            )

        with gr.Row():
            prev_btn = gr.Button("⬅ Prev")
            indicator = gr.Markdown("0 / 0")
            next_btn = gr.Button("Next ➡")
            refresh_btn = gr.Button("🔄 Refresh")

        jump = gr.Dropdown(
            label="Jump to prompt", choices=dropdown_choices(initial_filtered),
            value=0 if initial_filtered else None, interactive=True,
        )

        heading = gr.Markdown()

        prompt_box = gr.Textbox(
            label="Prompt (copyable)", lines=6, max_lines=20,
            interactive=False, show_copy_button=True,
        )

        # Fixed component graph: MAX_VIDEOS cards in a responsive wrapping grid.
        # Each slot = a full-width group-header markdown (shown only on the first
        # card of a model group) + a bordered card column holding the video on
        # top and its config footer at the bottom. CSS (CARD_CSS) gives the
        # cards their border/background and makes the grid wrap.
        card_headers, card_vids, card_footers = [], [], []
        with gr.Row(elem_id="card-grid"):
            for _ in range(MAX_VIDEOS):
                # Full-width header element (breaks the flex row at group bounds).
                hdr = gr.Markdown(elem_classes=["grp-header-card"])
                with gr.Column(elem_classes=["video-card"], min_width=280):
                    v = gr.Video(interactive=False, show_label=False)
                    footer = gr.Markdown()
                card_headers.append(hdr)
                card_vids.append(v)
                card_footers.append(footer)

        card_outputs = []
        for hdr, vid, footer in zip(card_headers, card_vids, card_footers):
            card_outputs += [hdr, vid, footer]
        # Order MUST match page_view: heading, prompt_box, indicator, then per
        # card (header, video, footer).
        outputs = [heading, prompt_box, indicator] + card_outputs

        def _apply(all_sets, query, model, page):
            view = filter_sets(all_sets, query, model)
            page = max(0, min(page, len(view) - 1)) if view else 0
            return (
                view, page,
                gr.update(choices=dropdown_choices(view),
                          value=page if view else None),
                *page_view(view, page),
            )

        def go_prev(all_sets, query, model, page):
            return _apply(all_sets, query, model, max(0, page - 1))

        def go_next(view, page):
            page = min(len(view) - 1, page + 1) if view else 0
            return (page, gr.update(value=page if view else None),
                    *page_view(view, page))

        def go_jump(view, sel):
            page = int(sel) if sel is not None else 0
            return (page, *page_view(view, page))

        def go_filter(all_sets, query, model):
            # Reset to page 0 on any filter change.
            return _apply(all_sets, query, model, 0)

        def go_refresh(query, model):
            all_sets = scan_sets()
            view, page, jump_u, *view_out = _apply(all_sets, query, model, 0)
            return (
                all_sets, view, page,
                gr.update(choices=model_choices(all_sets),
                          value=model if model in model_choices(all_sets)
                          else "All models"),
                jump_u, *view_out,
            )

        prev_btn.click(
            go_prev, inputs=[all_state, search, model_filter, page_state],
            outputs=[view_state, page_state, jump] + outputs)
        next_btn.click(
            go_next, inputs=[view_state, page_state],
            outputs=[page_state, jump] + outputs)
        jump.change(
            go_jump, inputs=[view_state, jump],
            outputs=[page_state] + outputs)
        search.submit(
            go_filter, inputs=[all_state, search, model_filter],
            outputs=[view_state, page_state, jump] + outputs)
        model_filter.change(
            go_filter, inputs=[all_state, search, model_filter],
            outputs=[view_state, page_state, jump] + outputs)
        refresh_btn.click(
            go_refresh, inputs=[search, model_filter],
            outputs=[all_state, view_state, page_state, model_filter, jump] + outputs)

        # Re-scan on every page load so fresh sessions see current sets, and the
        # first page's videos serve via the normal event path (static-paths).
        demo.load(
            go_refresh, inputs=[search, model_filter],
            outputs=[all_state, view_state, page_state, model_filter, jump] + outputs)

    return demo


def _log(msg):
    print(msg, flush=True)
    try:
        with open(os.path.join(ROOT, "compare_app.log"), "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def _kill_stale_tunnels():
    """Kill ANY leftover gradio `frpc` tunnel subprocesses before launching.

    gradio's share=True spawns an `frpc` binary that opens the *.gradio.live
    tunnel. That subprocess can OUTLIVE its python parent (e.g. when the parent
    is `pkill`ed), and while it lives its OLD share URL stays reachable — which
    is exactly the "dead URLs still work" bug. We therefore reap every frpc on
    EVERY launch so only the freshly minted tunnel/URL survives. Does NOT touch
    `results/` data and only targets gradio's own frpc binary, so it is safe
    and reversible. The share URL legitimately rotates each launch.
    """
    import signal
    import subprocess
    try:
        out = subprocess.run(
            ["pgrep", "-f", "frpc_linux"], capture_output=True, text=True
        ).stdout
    except OSError:
        return
    me = os.getpid()
    for line in out.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        if pid == me:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            _log(f"[compare_app] killed stale frpc tunnel pid {pid}")
        except OSError:
            pass


if __name__ == "__main__":
    import threading
    import time

    _kill_stale_tunnels()  # reap orphaned tunnels so old URLs die.
    gr.set_static_paths(paths=[RESULTS])
    demo = build()

    shared = os.environ.get("OMNINFT_SHARE", "0") == "1"
    try:
        demo.launch(
            server_name="0.0.0.0", server_port=8080, share=shared,
            allowed_paths=[RESULTS], prevent_thread_lock=True,
        )
    except Exception as e:  # noqa: BLE001
        _log(f"[compare_app] share=True failed ({e!r}); retrying share=False")
        shared = False
        demo.launch(
            server_name="0.0.0.0", server_port=8080, share=False,
            allowed_paths=[RESULTS], prevent_thread_lock=True,
        )

    _log("[compare_app] local URL: http://0.0.0.0:8080/")
    if shared:
        for _ in range(30):
            url = getattr(demo, "share_url", None)
            if url:
                _log(f"[compare_app] SHARE URL: {url}")
                break
            time.sleep(1)
        else:
            _log("[compare_app] share URL not available (tunnel may have failed)")

    threading.Event().wait()
