# dnd_lora_webui.py  (DnD LoRA WebUI)
from __future__ import annotations
import os, io, base64, requests, pathlib
from typing import Dict, Tuple, Optional, List
from PIL import Image
import gradio as gr
import json

# =============== Backend ===============
SD_URL = os.environ.get("SD_URL", "http://127.0.0.1:7860")
NONE_LABEL = "(None)"   # 不使用 LoRA 的占位

# =============== Catalog ===============
# 结构：CATALOG[cat][label] = (lora_name, default_weight, trigger_words, thumb_path)
CATALOG: Dict[str, Dict[str, Tuple[str, float, str, str]]] = {
    "race": {
        "Human (Male)":   ("human",       0.80, "human, 1male, european appearance",   r"C:\Users\Nicolas\Desktop\dndui\race\human\human_male.png"),
        "Human (Female)": ("human",       0.80, "human, 1girl, european appearance", r"C:\Users\Nicolas\Desktop\dndui\race\human\human.png"),
        "Drow (Male)":    ("drow_offset", 0.80, "drow, 1boy, dark elf, colored skin, dark skin, blue skin, pointy ears",   r"C:\Users\Nicolas\Desktop\dndui\race\drow\drow_male.png"),
        "Drow (Female)":  ("drow_offset", 0.80, "drow, 1girl, dark elf, colored skin, dark skin, blue skin, pointy ears", r"C:\Users\Nicolas\Desktop\dndui\race\drow\drow.png"),
        "Orc (Male)":     ("Orc_WoW_V2_ILXL", 0.80, "orc, 1boy, 0rcw0w, tusks, pointy ears", r"C:\Users\Nicolas\Desktop\dndui\race\orc\orc.png"),
        "Orc (Female)":   ("dnd_female_orc",  0.80, "orc, 1girl, tusks, pointy ears, muscle", r"C:\Users\Nicolas\Desktop\dndui\race\orc\orc_female.png"),
        "Dwarf (Male)":   ("Dwarf_r2", 0.80, "dwarf, 1boy, short stature, big nose, Duergar", r"C:\Users\Nicolas\Desktop\dndui\race\dwarf\dwarf.png"),
        "Dwarf (Female)": ("Dwarf_r2", 0.80, "dwarf, 1girl, short stature, big nose, Duergar", r"C:\Users\Nicolas\Desktop\dndui\race\dwarf\dwarf_female.png"),
    },
    "class": {
        "Paladin":  ("Fantasy_Classes__XL", 0.80, "fantasy class, paladin, knight, holy symbol", r"C:\Users\Nicolas\Desktop\dndui\class\paladin\paladin.png"),
        "Rogue":    ("Fantasy_Classes__XL", 0.80, "fantasy class, rogue, daggers, hood", r"C:\Users\Nicolas\Desktop\dndui\class\rogue\rogue.png"),
        "Sorcerer": ("sorcerer",            0.80, "fantasy class, sorcerer, wand, magic",r"C:\Users\Nicolas\Desktop\dndui\class\sorcerer\sorcerer.png"),
        # TODO: Cleric / Ranger / Bard / Barbarian / Monk / Warlock / ...
    },
    "armor": {
        "Plate":   ("xuer plate armor",   0.80, "xuer plate armor, plate armor", r"C:\Users\Nicolas\Desktop\dndui\armor\plate\plate.png"),
        "Leather": ("leatherarmor IL",    0.80, "leather armor, leather bracers, leather shoulder pads", r"C:\Users\Nicolas\Desktop\dndui\armor\leather\leather.png"),
        # "Chain": ("your_chainmail_lora", 0.8, "chainmail", r"...\armor\chain\chain.png"),
        # "Robe":  ("your_robe_lora",      0.8, "mage robe", r"...\armor\robe\robe.png"),
    },
    "character": {
        "Aegis": ("aegis_android_girl", 0.80, "Aegis, android girl", r"C:\Users\Nicolas\Desktop\dndui\style\aegis\aegis.png"),
        # TODO: 继续添加你的风格/角色 LoRA，如 Naruto / Sasuke ...
    },
}

DEFAULT_PROMPT = "((masterpiece, best quality)),"

DEFAULT_NEG = ("worst quality, large head, low quality, extra digits, bad eye, EasyNegativeV2, FastNegativeV2, ng_deepnegative_v1_75t, "
                "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, "
                "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
                "bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, "
                "extra arms, extra legs, fused fingers, too many fingers, long neck ")
SAMPLERS = ["DPM++ 2M Karras", "Euler a", "Euler", "DPM++ SDE Karras", "DDIM"]

# ---------- 参数预设（你可按喜好微调） ----------
PRESETS = {
    "authentic": {  # 偏真实 / 写实
        "sampler": "DPM++ 2M Karras",
        "steps": 20,
        "cfg": 8.0,
        "denoise": 0.45,  # 仅用于 img2img
    },
    "anime": {      # 偏二次元
        "sampler": "Euler a",
        "steps": 32,
        "cfg": 4.5,
        "denoise": 0.50,
    },
    "cartoon": {    # 偏卡通 / 扁平
        "sampler": "DPM++ SDE Karras",
        "steps": 20,
        "cfg": 4.5,
        "denoise": 0.55,
    },
}

# Text->Image：返回 steps, cfg, sampler 的“新值”
def _apply_preset_txt(name: str):
    p = PRESETS[name]
    return (
        gr.update(value=p["steps"]),
        gr.update(value=p["cfg"]),
        gr.update(value=p["sampler"]),
    )

# Image->Image：返回 steps2, cfg2, sampler2, denoise 的“新值”
def _apply_preset_img(name: str):
    p = PRESETS[name]
    return (
        gr.update(value=p["steps"]),
        gr.update(value=p["cfg"]),
        gr.update(value=p["sampler"]),
        gr.update(value=p["denoise"]),
    )

# =============== Helpers ===============
# 读取 A1111 的全局设置（拿 ClipSkip / 当前模型名等）
def get_sd_options(url: str) -> dict:
    try:
        r = requests.get(f"{url}/sdapi/v1/options", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# 组装“生成信息”文本：sampler / steps / cfg / seed / size / clipskip / checkpoint
def format_meta_text(sd_url: str, model_title: str,
                     sampler: str, steps: int, cfg: float, seed: int,
                     width: int, height: int, denoise: float | None = None) -> str:
    opts = get_sd_options(sd_url)
    clip_skip = opts.get("CLIP_stop_at_last_layers", None)
    cur_model = opts.get("sd_model_checkpoint", model_title or "")
    scheduler = "Karras" if ("Karras" in (sampler or "")) else "Default"

    lines = [
        "### 生成信息",
        f"- **Sampler**: {sampler}  | **Scheduler**: {scheduler}",
        f"- **Steps**: {steps}      | **CFG**: {cfg}",
        f"- **Seed**: {seed}        | **Size**: {width}×{height}",
        f"- **ClipSkip**: {clip_skip if clip_skip is not None else '—'}",
        f"- **Checkpoint**: {cur_model}",
        f"- **Backend**: {sd_url}",
    ]
    if denoise is not None:
        # 仅在 img2img 时显示
        lines.insert(3, f"- **Denoise**: {denoise}")
    return "\n".join(lines)

def safe_thumb(path: str) -> Image.Image:
    try:
        if path and pathlib.Path(path).exists():
            return Image.open(path).convert("RGB")
    except Exception:
        pass
    return Image.new("RGB", (320, 200), color=(235, 235, 235))

def lora_token(name: str, weight: float) -> str:
    return f"<lora:{name}:{weight:.2f}>"

def add_part(parts: List[str], cat: str, key: Optional[str], w: float, add_trig: bool = True):
    if not key or key == NONE_LABEL:
        return
    name, def_w, trig, _ = CATALOG[cat][key]
    parts.append(lora_token(name, w or def_w))
    if add_trig and trig:
        parts.append(trig)

def build_prompt(base: str,
                 race_key: Optional[str], race_w: float,
                 class_key: Optional[str], class_w: float,
                 armor_key: Optional[str], armor_w: float,
                 char_key: Optional[str],  char_w: float) -> str:
    parts: List[str] = ["(masterpiece, best quality),(ultra-detailed),(best illustration),(best shadow),(absurdres),(detailed background)," \
                "(very aesthetic),(beautiful hands:1.4), (separated fingers), (clear finger gaps)(five fingers, correct finger count:1.2)"]
    add_part(parts, "race",      race_key,  race_w)
    add_part(parts, "class",     class_key, class_w)
    add_part(parts, "armor",     armor_key, armor_w)
    add_part(parts, "character", char_key,  char_w)
    if base: parts.append(base)
    return ", ".join(parts)

def b64_to_pil(b64_img: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_img.split(",")[-1]))).convert("RGB")

# =============== A1111 API ===============
def check_api(url: str) -> str:
    try:
        r = requests.get(f"{url}/sdapi/v1/sd-models", timeout=10)
        r.raise_for_status()
        return f"✓ SD API OK ({url})，共 {len(r.json())} 个底模"
    except Exception as e:
        return f"✗ 无法访问 A1111 API：{url}\n{e}"

def list_checkpoints(url: str) -> List[str]:
    r = requests.get(f"{url}/sdapi/v1/sd-models", timeout=30)
    r.raise_for_status()
    return [m["title"] for m in r.json()]

def set_checkpoint(url: str, title: str) -> str:
    if not title: return "No model selected."
    r = requests.post(f"{url}/sdapi/v1/options",
                      json={"sd_model_checkpoint": title}, timeout=60)
    r.raise_for_status()
    return f"Applied: {title}"

def _extract_seed_from_info(data, fallback_seed):
    """
    A1111 会把实际种子放进 data['info']（字符串 JSON）
    优先取 'seed'，其次取 'all_seeds'[0]，否则用 fallback_seed
    """
    try:
        info = data.get("info", {})
        if isinstance(info, str):
            info = json.loads(info)
        if isinstance(info, dict):
            if "seed" in info and info["seed"] is not None:
                return info["seed"]
            if isinstance(info.get("all_seeds"), list) and info["all_seeds"]:
                return info["all_seeds"][0]
    except Exception:
        pass
    return fallback_seed

def sd_txt2img(url: str, prompt: str, negative: str, steps: int, cfg: float, sampler: str,
               width: int, height: int, seed: int):
    payload = {
        "prompt": prompt, "negative_prompt": negative or DEFAULT_NEG,
        "steps": int(steps), "cfg_scale": float(cfg), "sampler_name": sampler,
        "width": int(width), "height": int(height), "seed": int(seed),
    }
    r = requests.post(f"{url}/sdapi/v1/txt2img", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    img = b64_to_pil(data["images"][0])
    used_seed = _extract_seed_from_info(data, seed)
    return img, used_seed


def sd_img2img(url: str, init_img: Image.Image, prompt: str, negative: str, steps: int, cfg: float,
               sampler: str, denoise: float, seed: int):
    buf = io.BytesIO(); init_img.save(buf, format="PNG")
    payload = {
        "init_images": [base64.b64encode(buf.getvalue()).decode()],
        "prompt": prompt, "negative_prompt": negative or DEFAULT_NEG,
        "steps": int(steps), "cfg_scale": float(cfg), "sampler_name": sampler,
        "denoising_strength": float(denoise), "seed": int(seed),
    }
    r = requests.post(f"{url}/sdapi/v1/img2img", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    img = b64_to_pil(data["images"][0])
    used_seed = _extract_seed_from_info(data, seed)
    return img, used_seed

# ===== XY Grid helpers =====
import random
from PIL import ImageFont, ImageDraw

AXIS_TYPES = [
    "Nothing", "Prompt S/R", "Steps", "CFG Scale", "Sampler", "Seed", "Width", "Height",
]

def _parse_list(s: str, numeric: bool):
    vals = [v.strip() for v in (s or "").split(",") if v.strip() != ""]
    if numeric:
        out = []
        for v in vals:
            try:
                out.append(int(v))
            except ValueError:
                out.append(float(v))
        return out
    return vals

def _apply_axis(kind: str, value, prompt: str, params: dict):
    p = prompt
    q = params.copy()
    if kind == "Prompt S/R":
        find = q.get("_sr_find", "")
        if find:
            p = p.replace(find, str(value))
    elif kind == "Steps":
        q["steps"] = int(value)
    elif kind == "CFG Scale":
        q["cfg"] = float(value)
    elif kind == "Sampler":
        q["sampler"] = str(value)
    elif kind == "Seed":
        q["seed"] = int(value)
    elif kind == "Width":
        q["width"] = int(value)
    elif kind == "Height":
        q["height"] = int(value)
    return p, q

def _text_size(draw: ImageDraw.ImageDraw, text: str, font):
    # 兼容 Pillow 9/10
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:
        return draw.textsize(text, font=font)

def _make_grid(imgs, cols, rows, cell_w, cell_h, draw_legend=False, legends=None):
    pad = 8
    legend_h = 36 if draw_legend else 0
    W = cols * cell_w + (cols - 1) * pad
    H = rows * (cell_h + legend_h + (pad if draw_legend else 0)) + (rows - 1) * pad
    grid = Image.new("RGB", (W, H), (32, 32, 32))
    font = ImageFont.load_default()
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(imgs):
                break
            x = c * (cell_w + pad)
            y = r * (cell_h + legend_h + (pad if draw_legend else 0))
            if draw_legend and legends:
                draw = ImageDraw.Draw(grid)
                txt = legends[idx]
                tw, th = _text_size(draw, txt, font)
                draw.rectangle([x, y, x + cell_w, y + legend_h], fill=(22, 22, 22))
                draw.text((x + 6, y + (legend_h - th) // 2), txt, fill=(230, 230, 230), font=font)
                y += legend_h
            grid.paste(imgs[idx], (x, y))
    return grid

def run_xy_grid_txt(
    sd_url: str,
    base_prompt: str, neg_prompt: str,
    race_key, race_w, class_key, class_w, armor_key, armor_w, char_key, char_w,
    steps: int, cfg: float, sampler: str, width: int, height: int, seed: int, model_title: str,
    x_type: str, x_values: str,
    y_type: str, y_values: str,
    draw_legend: bool,
):
    # 用你自己的 build_prompt 产出“完整提示词”
    base_full = build_prompt(base_prompt, race_key, race_w, class_key, class_w, armor_key, armor_w, char_key, char_w)

    # 解析 X/Y
    xs = []; ys = []
    sr_find_x = sr_find_y = ""
    if x_type == "Prompt S/R":
        tmp = _parse_list(x_values, numeric=False)
        if len(tmp) < 2:
            raise gr.Error("X 轴 Prompt S/R 至少需要两项：第 1 项=查找片段，其余=替换片段。")
        sr_find_x, xs = tmp[0], tmp[1:]
    elif x_type != "Nothing":
        xs = _parse_list(x_values, numeric=True)
    else:
        xs = ["–"]

    if y_type == "Prompt S/R":
        tmp = _parse_list(y_values, numeric=False)
        if len(tmp) < 2:
            raise gr.Error("Y 轴 Prompt S/R 至少需要两项：第 1 项=查找片段，其余=替换片段。")
        sr_find_y, ys = tmp[0], tmp[1:]
    elif y_type != "Nothing" and y_values.strip():
        ys = _parse_list(y_values, numeric=True)
    else:
        y_type = "Nothing"
        ys = ["–"]

    # 固定种子（-1 则统一随机）
    if int(seed) == -1:
        seed = int(random.randint(0, 2**31 - 1))

    base_params = dict(steps=int(steps), cfg=float(cfg), sampler=sampler, width=int(width), height=int(height), seed=int(seed))
    if x_type == "Prompt S/R": base_params["_sr_find"] = sr_find_x
    if y_type == "Prompt S/R": base_params["_sr_find"] = sr_find_y  # 不建议 X/Y 同时 S/R

    images = []; legends = []
    for yv in ys:
        for xv in xs:
            prompt_cell, params = base_full, base_params.copy()
            prompt_cell, params = _apply_axis(x_type, xv, prompt_cell, params)
            prompt_cell, params = _apply_axis(y_type, yv, prompt_cell, params)

            # 适配你的 sd_txt2img：返回 image 或 (image, seed) 都兼容
            res = sd_txt2img(sd_url, prompt_cell, neg_prompt, params["steps"], params["cfg"],
                             params["sampler"], params["width"], params["height"], params["seed"])
            img = res[0] if isinstance(res, (tuple, list)) else res
            images.append(img)

            xlab = f"{x_type}: {xv}" if x_type != "Nothing" else ""
            ylab = f"{y_type}: {yv}" if y_type != "Nothing" else ""
            legends.append(", ".join([t for t in [xlab, ylab] if t]))

    cols, rows = len(xs), len(ys)
    cell_w, cell_h = images[0].width, images[0].height
    grid = _make_grid(images, cols, rows, cell_w, cell_h, draw_legend=draw_legend, legends=legends)

    # 信息框（用你现有的 format_meta_text，如果没有就简单拼一下）
    meta = format_meta_text(sd_url, model_title, base_params["sampler"], base_params["steps"], base_params["cfg"],
                            seed, base_params["width"], base_params["height"], None)
    meta += f"\n\n**XY 组图**：X={x_type}（{len(xs)} 值），Y={y_type}（{len(ys)} 值）"

    used_prompt = base_full  # 显示基础提示词（避免太长）
    return grid, used_prompt, meta


# =============== UI helpers ===============
def catalog_to_gallery(cat: str) -> List[List[object]]:
    # 第一项插入“(None)”
    items = [[safe_thumb(""), NONE_LABEL]]
    for label, (_, _, _, thumb) in CATALOG[cat].items():
        items.append([safe_thumb(thumb), label])
    return items

def make_pick_label(cat: str):
    # v4: 事件 gr.SelectData
    def _fn(evt: gr.SelectData):
        if evt and evt.index == 0:
            return NONE_LABEL
        keys = list(CATALOG[cat].keys())
        try:
            idx = evt.index - 1
            return keys[idx] if 0 <= idx < len(keys) else NONE_LABEL
        except Exception:
            return NONE_LABEL
    return _fn

# =============== Callbacks ===============
def gen_txt(sd_url, base_prompt, neg_prompt,
            race_key, race_w, class_key, class_w, armor_key, armor_w, char_key, char_w,
            steps, cfg, sampler, width, height, seed, model_title):
    if model_title:
        try: set_checkpoint(sd_url, model_title)
        except Exception as e: print("[WARN] set_checkpoint:", e)
    prompt = build_prompt(base_prompt, race_key, race_w, class_key, class_w, armor_key, armor_w, char_key, char_w)
    img, used_seed = sd_txt2img(sd_url, prompt, neg_prompt, steps, cfg, sampler, width, height, seed)
    meta_text = format_meta_text(sd_url, model_title, sampler, steps, cfg, used_seed, width, height, None)
    return img, prompt, meta_text

def gen_img(sd_url, init_img, base_prompt, neg_prompt,
            race_key, race_w, class_key, class_w, armor_key, armor_w, char_key, char_w,
            steps, cfg, sampler, denoise, seed, model_title):
    if model_title:
        try: set_checkpoint(sd_url, model_title)
        except Exception as e: print("[WARN] set_checkpoint:", e)
    prompt = build_prompt(base_prompt, race_key, race_w, class_key, class_w, armor_key, armor_w, char_key, char_w)
    img, used_seed = sd_txt2img(sd_url, prompt, neg_prompt, steps, cfg, sampler, width, height, seed)
    meta_text = format_meta_text(sd_url, model_title, sampler, steps, cfg, used_seed, img.width, img.height, denoise)
    return img, prompt, meta_text

# =============== UI Layout ===============
CSS = """
:root { --accent: #7c3aed; }
.gradio-container { max-width: 1160px; margin: auto; }
footer { display:none; }
.g-race  { height: 380px; overflow: auto; }
.g-class { height: 300px; overflow: auto; }
.g-armor { height: 180px; overflow: auto; }
.g-char  { height: 220px; overflow: auto; }
"""
with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# 🛡️ D&D LoRA WebUI — Race × Class × Armor × Character  \n*点击卡片选择 LoRA*")

    # 顶部：SD_URL + 测试 + 底模
    with gr.Row():
        sd_url_tb  = gr.Textbox(value=SD_URL, label="后端 URL (A1111 SD API)", scale=3)
        btn_test   = gr.Button("测试 API", scale=1)
        test_msg   = gr.Markdown("")
    btn_test.click(lambda u: check_api(u), inputs=sd_url_tb, outputs=test_msg)

    with gr.Row():
        model_dd    = gr.Dropdown(choices=[], label="底模 (Checkpoint)", interactive=True, scale=4)
        btn_refresh = gr.Button("刷新列表", scale=1)
        btn_apply   = gr.Button("应用底模", scale=1)
        apply_msg   = gr.Markdown("")

    def _refresh_models(u):
        try:
            return gr.update(choices=list_checkpoints(u)), ""
        except Exception as e:
            return gr.update(choices=[]), f"✗ 列表获取失败：{e}"

    btn_refresh.click(_refresh_models, inputs=sd_url_tb, outputs=[model_dd, apply_msg])
    btn_apply.click(lambda u, t: set_checkpoint(u, t), inputs=[sd_url_tb, model_dd], outputs=apply_msg)

    with gr.Tabs():
        # ---------- TXT2IMG ----------
        with gr.Tab("Text → Image"):
            
            with gr.Accordion("风格选择", open=False):
                with gr.Row():
                    steps   = gr.Slider(10, 80, value=36, step=1, label="Steps")
                    cfg     = gr.Slider(1.0, 12.0, value=4.5, step=0.5, label="CFG Scale")
                    sampler = gr.Dropdown(SAMPLERS, value="DPM++ 2M Karras", label="Sampler")
                with gr.Row():
                    width  = gr.Slider(512, 2048, value=1024, step=64, label="宽")
                    height = gr.Slider(512, 2048, value=1024, step=64, label="高")
                    seed   = gr.Number(value=-1, precision=0, label="Seed (-1 随机)")

                # ★ 参数预设按钮（Text->Image）
                with gr.Row():
                    btn_preset_auth = gr.Button("Preset: Authentic")
                    btn_preset_anime = gr.Button("Preset: Anime")
                    btn_preset_cartoon = gr.Button("Preset: Cartoon")

            btn_preset_auth.click(lambda: _apply_preset_txt("authentic"), outputs=[steps, cfg, sampler])
            btn_preset_anime.click(lambda: _apply_preset_txt("anime"), outputs=[steps, cfg, sampler])
            btn_preset_cartoon.click(lambda: _apply_preset_txt("cartoon"), outputs=[steps, cfg, sampler])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 选择 LoRA（点击卡片）")

                    gr.Markdown("**Race（含男女）**")
                    race_gallery = gr.Gallery(label="Race", value=catalog_to_gallery("race"), elem_classes=["g-race"])
                    race_choice  = gr.Dropdown([NONE_LABEL] + list(CATALOG["race"].keys()),
                                               value=NONE_LABEL, label="已选 Race", interactive=True)
                    race_w = gr.Slider(0.0, 1.2, value=0.80, step=0.05, label="Race 权重")
                    race_gallery.select(fn=make_pick_label("race"), outputs=race_choice)

                    gr.Markdown("**Class**")
                    class_gallery = gr.Gallery(label="Class", value=catalog_to_gallery("class"), elem_classes=["g-class"])
                    class_choice  = gr.Dropdown([NONE_LABEL] + list(CATALOG["class"].keys()),
                                                value=NONE_LABEL, label="已选 Class", interactive=True)
                    class_w = gr.Slider(0.0, 1.2, value=0.80, step=0.05, label="Class 权重")
                    class_gallery.select(fn=make_pick_label("class"), outputs=class_choice)

                    gr.Markdown("**Armor**")
                    armor_gallery = gr.Gallery(label="Armor", value=catalog_to_gallery("armor"), elem_classes=["g-armor"])
                    armor_choice  = gr.Dropdown([NONE_LABEL] + list(CATALOG["armor"].keys()),
                                                value=NONE_LABEL, label="已选 Armor", interactive=True)
                    armor_w = gr.Slider(0.0, 1.2, value=0.8, step=0.05, label="Armor 权重")
                    armor_gallery.select(fn=make_pick_label("armor"), outputs=armor_choice)

                    gr.Markdown("**Character Style（可选）**")
                    char_gallery = gr.Gallery(label="Character", value=catalog_to_gallery("character"), elem_classes=["g-char"])
                    char_choice  = gr.Dropdown([NONE_LABEL] + list(CATALOG["character"].keys()),
                                               value=NONE_LABEL, label="已选 Style", interactive=True)
                    char_w = gr.Slider(0.0, 1.2, value=0.8, step=0.05, label="Style 权重")
                    char_gallery.select(fn=make_pick_label("character"), outputs=char_choice)

                with gr.Column(scale=2):
                    base_prompt = gr.Textbox(lines=5, label="提示词 (可追加外观/动作/场景)",
                        value=DEFAULT_PROMPT,
                        placeholder="e.g., elegant hand pose, separated fingers, dramatic lighting")
                    neg_prompt = gr.Textbox(lines=3, label="负面提示词", value=DEFAULT_NEG)
                    with gr.Accordion("调参 (可选)", open=False):
                        with gr.Row():
                            steps   = gr.Slider(10, 80, value=36, step=1, label="Steps")
                            cfg     = gr.Slider(1.0, 12.0, value=4.5, step=0.5, label="CFG Scale")
                            sampler = gr.Dropdown(SAMPLERS, value="DPM++ 2M Karras", label="Sampler")
                        with gr.Row():
                            width  = gr.Slider(512, 2048, value=1024, step=64, label="宽")
                            height = gr.Slider(512, 2048, value=1024, step=64, label="高")
                            seed   = gr.Number(value=-1, precision=0, label="Seed (-1 随机)")
                    btn_txt     = gr.Button("生成", variant="primary")
                    out_img     = gr.Image(type="pil", label="结果")
                    used_prompt = gr.Textbox(label="实际提示词 (含 LoRA 标记)")
                    meta_md = gr.Markdown("", label="生成信息")
                
                with gr.Accordion("X/Y 组图（可选）", open=False):
                    with gr.Row():
                        x_type = gr.Dropdown(AXIS_TYPES, value="Prompt S/R", label="X type")
                        x_vals = gr.Textbox(value="", lines=1,
                                            label="X values（逗号分隔；S/R 时第 1 项=查找片段，其余=替换）")
                    with gr.Row():
                        y_type = gr.Dropdown(AXIS_TYPES, value="Nothing", label="Y type")
                        y_vals = gr.Textbox(value="", lines=1, label="Y values")
                    draw_legend = gr.Checkbox(value=True, label="绘制图例")
                    btn_xy = gr.Button("生成组图 (XY Plot)", variant="secondary")

                btn_xy.click(
                    fn=run_xy_grid_txt,
                    inputs=[
                        sd_url_tb, base_prompt, neg_prompt,
                        race_choice, race_w, class_choice, class_w, armor_choice, armor_w, char_choice, char_w,
                        steps, cfg, sampler, width, height, seed, model_dd,
                        x_type, x_vals, y_type, y_vals, draw_legend
                    ],
                    outputs=[out_img, used_prompt, meta_md]
)
            btn_txt.click(
                gen_txt,
                inputs=[sd_url_tb, base_prompt, neg_prompt,
                        race_choice, race_w, class_choice, class_w, armor_choice, armor_w, char_choice, char_w,
                        steps, cfg, sampler, width, height, seed, model_dd],
                outputs=[out_img, used_prompt, meta_md]
            )

        # ---------- IMG2IMG ----------
        with gr.Tab("Image → Image"):
            
            with gr.Accordion("风格选择", open=False):
                with gr.Row():
                    steps2   = gr.Slider(10, 80, value=36, step=1, label="Steps")
                    cfg2     = gr.Slider(1.0, 12.0, value=4.5, step=0.5, label="CFG Scale")
                    sampler2 = gr.Dropdown(SAMPLERS, value="DPM++ 2M Karras", label="Sampler")
                with gr.Row():
                    denoise = gr.Slider(0.1, 0.95, value=0.45, step=0.01, label="Denoising Strength")
                    seed2   = gr.Number(value=-1, precision=0, label="Seed (-1 随机)")

                # ★ 参数预设按钮（Image->Image）
                with gr.Row():
                    btn_preset_auth2 = gr.Button("Preset: Authentic")
                    btn_preset_anime2 = gr.Button("Preset: Anime")
                    btn_preset_cartoon2 = gr.Button("Preset: Cartoon")

            # 绑定：点击按钮 -> 更新 steps2, cfg2, sampler2, denoise
            btn_preset_auth2.click(lambda: _apply_preset_img("authentic"), outputs=[steps2, cfg2, sampler2, denoise])
            btn_preset_anime2.click(lambda: _apply_preset_img("anime"), outputs=[steps2, cfg2, sampler2, denoise])
            btn_preset_cartoon2.click(lambda: _apply_preset_img("cartoon"), outputs=[steps2, cfg2, sampler2, denoise])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 选择 LoRA（点击卡片）")
                    race_gallery2 = gr.Gallery(value=catalog_to_gallery("race"), elem_classes=["g-race"])
                    race_choice2  = gr.Dropdown([NONE_LABEL] + list(CATALOG["race"].keys()),
                                                value=NONE_LABEL, label="已选 Race", interactive=True)
                    race_w2 = gr.Slider(0.0, 1.2, value=0.80, step=0.05, label="Race 权重")
                    race_gallery2.select(fn=make_pick_label("race"), outputs=race_choice2)

                    class_gallery2 = gr.Gallery(value=catalog_to_gallery("class"), elem_classes=["g-class"])
                    class_choice2  = gr.Dropdown([NONE_LABEL] + list(CATALOG["class"].keys()),
                                                 value=NONE_LABEL, label="已选 Class", interactive=True)
                    class_w2 = gr.Slider(0.0, 1.2, value=0.80, step=0.05, label="Class 权重")
                    class_gallery2.select(fn=make_pick_label("class"), outputs=class_choice2)

                    armor_gallery2 = gr.Gallery(value=catalog_to_gallery("armor"), elem_classes=["g-armor"])
                    armor_choice2  = gr.Dropdown([NONE_LABEL] + list(CATALOG["armor"].keys()),
                                                 value=NONE_LABEL, label="已选 Armor", interactive=True)
                    armor_w2 = gr.Slider(0.0, 1.2, value=0.8, step=0.05, label="Armor 权重")
                    armor_gallery2.select(fn=make_pick_label("armor"), outputs=armor_choice2)

                    char_gallery2 = gr.Gallery(value=catalog_to_gallery("character"), elem_classes=["g-char"])
                    char_choice2  = gr.Dropdown([NONE_LABEL] + list(CATALOG["character"].keys()),
                                                 value=NONE_LABEL, label="已选 Style", interactive=True)
                    char_w2 = gr.Slider(0.0, 1.2, value=0.8, step=0.05, label="Style 权重")
                    char_gallery2.select(fn=make_pick_label("character"), outputs=char_choice2)

                with gr.Column(scale=2):
                    init_img     = gr.Image(type="pil", label="初始图像 (img2img)")
                    base_prompt2 = gr.Textbox(lines=5, label="提示词 (追加改动点)", value=DEFAULT_PROMPT)
                    neg_prompt2  = gr.Textbox(lines=3, label="负面提示词", value=DEFAULT_NEG)
                    with gr.Accordion("调参 (可选)", open=False):
                        with gr.Row():
                            steps2   = gr.Slider(10, 80, value=36, step=1, label="Steps")
                            cfg2     = gr.Slider(1.0, 12.0, value=4.5, step=0.5, label="CFG Scale")
                            sampler2 = gr.Dropdown(SAMPLERS, value="DPM++ 2M Karras", label="Sampler")
                        with gr.Row():
                            denoise = gr.Slider(0.1, 0.95, value=0.45, step=0.01, label="Denoising Strength")
                            seed2   = gr.Number(value=-1, precision=0, label="Seed (-1 随机)")
                    btn_img      = gr.Button("重绘", variant="primary")
                    out_img2     = gr.Image(type="pil", label="结果")
                    used_prompt2 = gr.Textbox(label="实际提示词 (含 LoRA 标记)")
                    meta_md2 = gr.Markdown("", label="生成信息")

            btn_img.click(
                gen_img,
                inputs=[sd_url_tb, init_img, base_prompt2, neg_prompt2,
                        race_choice2, race_w2, class_choice2, class_w2, armor_choice2, armor_w2, char_choice2, char_w2,
                        steps2, cfg2, sampler2, denoise, seed2, model_dd],
                outputs=[out_img2, used_prompt2, meta_md2]
            )

    gr.Markdown("""
**提示**
- 先点“测试 API”，确认 `后端 URL` 指向你的 A1111（必须用 `--api` 启动）。若换端口，在此修改即可。
- 底模下拉可刷新并应用；也会在每次生成前自动应用当前选择。
- 四个选择区均可选 **(None)** 表示不使用对应 LoRA。
- 想要更小图，直接把宽高调成 640/768 等尺寸生成；优先按目标尺寸出图，胜过事后再压缩。
""")

if __name__ == "__main__":
    demo.queue(max_size=64).launch(server_name="127.0.0.1", server_port=7861, inbrowser=False)
