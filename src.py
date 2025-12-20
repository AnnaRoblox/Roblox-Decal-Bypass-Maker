import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
import numpy as np
import random
import os
from pathlib import Path
import cv2
import threading
import webbrowser
import urllib.request
import json
from tkinter import simpledialog
import io
import base64

def apply_color_clearer(src_img, target_color):
    """The core steganography function to make an image \'clear\' against a specific color. """
    src_arr = np.array(src_img).astype(np.float64)
    target_color = np.array(target_color, dtype=np.float64)
    Rf, Gf, Bf, Af = (src_arr[:, :, 0], src_arr[:, :, 1], src_arr[:, :, 2], src_arr[:, :, 3])
    Rb, Gb, Bb = (target_color[0], target_color[1], target_color[2])
    alpha_norm = Af / 255.0
    Rc = Rb * (1 - alpha_norm) + Rf * alpha_norm
    Gc = Gb * (1 - alpha_norm) + Gf * alpha_norm
    Bc = Bb * (1 - alpha_norm) + Bf * alpha_norm

    def get_min_alpha(Cc, Cb):
        Ac = np.zeros_like(Cc, dtype=np.float64)
        mask1 = Cc > Cb
        denom1 = 255.0 - Cb
        valid_mask1 = mask1 & (denom1 > 0)
        Ac[valid_mask1] = np.ceil(255.0 * (Cc[valid_mask1] - Cb) / denom1)
        mask2 = Cc < Cb
        valid_mask2 = mask2 & (Cb > 0)
        Ac[valid_mask2] = np.ceil(255.0 * (Cb - Cc[valid_mask2]) / Cb)
        return Ac
    Ac_r = get_min_alpha(Rc, Rb)
    Ac_g = get_min_alpha(Gc, Gb)
    Ac_b = get_min_alpha(Bc, Bb)
    Ac_final = np.maximum.reduce([Ac_r, Ac_g, Ac_b])

    def get_new_fg_color(Cc, Cb, Ac):
        new_Cf = np.copy(Cc)
        mask = Ac > 0
        numerator = Cc[mask] * 255.0 - Cb * (255.0 - Ac[mask])
        denominator = Ac[mask]
        new_Cf[mask] = numerator / denominator
        return new_Cf
    new_Rf = get_new_fg_color(Rc, Rb, Ac_final)
    new_Gf = get_new_fg_color(Gc, Gb, Ac_final)
    new_Bf = get_new_fg_color(Bc, Bb, Ac_final)
    final_arr = np.stack([np.clip(new_Rf, 0, 255), np.clip(new_Gf, 0, 255), np.clip(new_Bf, 0, 255), np.clip(Ac_final, 0, 255)], axis=(-1))
    return Image.fromarray(final_arr.astype(np.uint8), 'RGBA')

def tint_image(img, tint_color, opacity):
    """Tints an image with a given color and opacity, preserving its alpha channel."""
    img = img.convert('RGBA') if img.mode!= 'RGBA' else img
    alpha = img.getchannel('A')
    solid_color_img = Image.new('RGB', img.size, tint_color)
    tinted_img = solid_color_img.copy()
    tinted_img.putalpha(alpha)
    final_alpha = (np.array(alpha).astype(np.float32) * (opacity / 255.0)).astype(np.uint8)
    tinted_img.putalpha(Image.fromarray(final_alpha))
    return tinted_img

def apply_watermark(base_img, watermark_settings, bait_clear_color_hex=None, apply_contrasting_color_logic=False):
    """
    Applies a watermark (text or image) to the given base image.
    """
    if not watermark_settings['use_watermark'].get():
        return base_img

    img = base_img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    wm_type = watermark_settings['watermark_type'].get()
    wm_opacity = watermark_settings['watermark_opacity'].get()
    wm_size = watermark_settings['watermark_size'].get()
    wm_position = watermark_settings['watermark_position'].get()
    if wm_type == "Text":
        text = watermark_settings['watermark_text'].get()
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(overlay)
        text_size = draw.textsize(text, font=font)
        scale = wm_size * min(img.size)
        font = ImageFont.truetype("arial.ttf", int(scale)) if os.path.exists("arial.ttf") else ImageFont.load_default()
        text_size = draw.textsize(text, font=font)
        positions = {
            "Top Left": (10, 10),
            "Top Right": (img.size[0] - text_size[0] - 10, 10),
            "Bottom Left": (10, img.size[1] - text_size[1] - 10),
            "Bottom Right": (img.size[0] - text_size[0] - 10, img.size[1] - text_size[1] - 10),
            "Center": ((img.size[0] - text_size[0]) // 2, (img.size[1] - text_size[1]) // 2),
        }
        pos = positions.get(wm_position, (10, 10))

        draw.text(pos, text, fill=(255, 255, 255, wm_opacity), font=font)
    elif wm_type == "Image":
        wm_path = watermark_settings['watermark_image_path'].get()
        if wm_path and os.path.exists(wm_path):
            wm_img = Image.open(wm_path).convert("RGBA")
            wm_w = int(img.size[0] * wm_size)
            wm_h = int(img.size[1] * wm_size)
            wm_img = wm_img.resize((wm_w, wm_h), Image.ANTIALIAS)
            alpha = wm_img.getchannel("A")
            alpha = alpha.point(lambda p: p * wm_opacity / 255)
            wm_img.putalpha(alpha)
            positions = {
                "Top Left": (10, 10),
                "Top Right": (img.size[0] - wm_w - 10, 10),
                "Bottom Left": (10, img.size[1] - wm_h - 10),
                "Bottom Right": (img.size[0] - wm_w - 10, img.size[1] - wm_h - 10),
                "Center": ((img.size[0] - wm_w) // 2, (img.size[1] - wm_h) // 2),
            }
            pos = positions.get(wm_position, (10, 10))

            overlay.paste(wm_img, pos, wm_img)
    return Image.alpha_composite(img, overlay)

class EditableLabel(tk.Frame):
    """\nA label that turns into an entry field on click, allowing for precise value setting.\nIt\'s associated with a Tkinter variable (e.g., IntVar, StringVar) that it updates.\n"""

    def __init__(self, parent, textvariable, **kwargs):
        frame_kwargs = {'bg': kwargs.get('bg')}
        super().__init__(parent, **frame_kwargs)
        self.textvariable = textvariable
        self.label = tk.Label(self, textvariable=textvariable, **kwargs)
        self.label.pack(fill='both', expand=True)
        entry_kwargs = kwargs.copy()
        entry_kwargs['relief'] = 'sunken'
        entry_kwargs['borderwidth'] = 1
        self.entry = tk.Entry(self, textvariable=textvariable, **entry_kwargs)
        self.label.bind('<Button-1>', self.show_entry)
        self.entry.bind('<FocusOut>', self.show_label)
        self.entry.bind('<Return>', self.show_label)

    def show_entry(self, event=None):
        self.label.pack_forget()
        self.entry.pack(fill='both', expand=True)
        self.entry.focus_set()
        self.entry.select_range(0, 'end')

    def show_label(self, event=None):
        self.entry.pack_forget()
        self.label.pack(fill='both', expand=True)

def adjust_levels(image, lower_bound, upper_bound):
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)

    if lower_bound >= upper_bound:
        return image
    img_arr = np.array(image).astype(np.float32)
    img_arr = (img_arr - lower_bound) * (255.0 / (upper_bound - lower_bound))
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr)

def create_pencil_sketch(image_bgr_np, pencil_tip_size=20, range_param=-1.5):
    """
    Creates a pencil sketch from a BGR numpy image array.
    Uses grayscale conversion, Gaussian blur, and division blending.
    """
    if image_bgr_np is None:
        return None, "Input image data is invalid."

    try:
        gray = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blur = cv2.GaussianBlur(inverted, (pencil_tip_size, pencil_tip_size), sigmaX=0, sigmaY=0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)

        return sketch, None
    except Exception as e:
        return None, str(e)

class StegoDecalBuilder:
    CURRENT_VERSION = '2.1'
    VERSION_URL = 'https://raw.githubusercontent.com/AnnaRoblox/Roblox-Decal-Bypass-Maker/refs/heads/main/Version'
    REPO_URL = 'https://github.com/AnnaRoblox/Roblox-Decal-Bypass-Maker/releases'
    SETTINGS_FILE = 'settings.json'

    def __init__(self, root):
        self.root = root
        self.root.title(f'AnnaRoblox\'s Decal Bypass Maker v{self.CURRENT_VERSION}')
        self.root.geometry('800x1200')
        self.root.configure(bg='#1a1a1a')
        self.real_path = None
        self.bait_path = None
        self.output_path = None
        self.preview_img_tk = None
        self.output_width = tk.StringVar(value='300')
        self.output_height = tk.StringVar(value='300')
        self.bait_clear_color = '#FFFFFF'
        self.real_clear_color = '#000000'
        self.bait_opacity = tk.IntVar(value=255)
        self.real_opacity = tk.IntVar(value=255)
        self.slip_mode_option = tk.StringVar(value='Single Random Pixel')
        self.use_custom_layer_method = tk.BooleanVar(value=False)
        self.custom_layers = []
        self.batch_copies = tk.StringVar(value='10')
        self.batch_basename = tk.StringVar(value='output_decal')
        self.use_auto_bait = tk.BooleanVar(value=False)
        self.pencil_tip_size = tk.DoubleVar(value=21.0)
        self.pencil_range = tk.DoubleVar(value=(-1.5))
        self.designated_bait_folder = tk.StringVar(value='')
        self.designated_image_folder = tk.StringVar(value='')
        self.use_watermark = tk.BooleanVar(value=False)
        self.watermark_type = tk.StringVar(value='Text')
        self.watermark_text = tk.StringVar(value='Watermark')
        self.watermark_image_path = tk.StringVar(value='')
        self.watermark_size = tk.DoubleVar(value=0.1)
        self.watermark_opacity = tk.IntVar(value=128)
        self.watermark_position = tk.StringVar(value='Center')
        self.watermark_target = tk.StringVar(value='Image')
        self.use_custom_preview_color = tk.BooleanVar(value=False)
        self.custom_preview_color = tk.StringVar(value='#808080')
        self.load_settings()
        self.output_width.trace_add('write', self.save_settings)
        self.output_height.trace_add('write', self.save_settings)
        self.slip_mode_option.trace_add('write', self.save_settings)
        self.batch_copies.trace_add('write', self.save_settings)
        self.batch_basename.trace_add('write', self.save_settings)
        self.use_auto_bait.trace_add('write', self.save_settings)
        self.pencil_tip_size.trace_add('write', self.save_settings)
        self.pencil_range.trace_add('write', self.save_settings)
        self.use_custom_preview_color.trace_add('write', self.save_settings)
        self.custom_preview_color.trace_add('write', self.save_settings)
        self.setup_ui()
        self.update_hex_entries()
        self.check_for_updates_in_background()

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.main_frame.pack(fill='both', expand=True)
        self.canvas = tk.Canvas(self.main_frame, bg='#1a1a1a', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.root.bind('<Control-v>', self.handle_paste)
        content = tk.Frame(self.canvas, bg='#1a1a1a')
        self.canvas.create_window((0, 0), window=content, anchor='nw')
        tk.Label(content, text='AnnaRoblox\'s Decal Bypass Maker', font=('Arial', 24, 'bold'), fg='#00ff99', bg='#1a1a1a').pack(pady=20)
        frame_select = tk.LabelFrame(content, text='1. Select Images', font=('Arial', 12), fg='#00ccff', bg='#1a1a1a', padx=10, pady=10)
        frame_select.pack(fill='x', padx=20, pady=10)
        frame_select.columnconfigure(1, weight=1)
        tk.Label(frame_select, text='Bait Layer', fg='#55ff55', bg='#1a1a1a', font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=3, sticky='w', padx=10)
        bait_buttons_frame = tk.Frame(frame_select, bg='#1a1a1a')
        bait_buttons_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
        tk.Button(bait_buttons_frame, text='Select Bait Image...', command=self.load_bait, bg='#55ff55', fg='black').pack(side='left')
        tk.Button(bait_buttons_frame, text='Set Bait Folder...', command=self.set_designated_bait_folder, bg='#4c806d', fg='white').pack(side='left', padx=10)
        tk.Button(bait_buttons_frame, text='Save Bait Image...', command=self.save_bait_image, bg='#5555ff', fg='white').pack(side='left', padx=10)
        self.bait_label = tk.Label(frame_select, text='No file', fg='#aaa', bg='#1a1a1a', anchor='w')
        self.bait_label.grid(row=1, column=1, padx=10, pady=5, sticky='ew')
        tk.Label(frame_select, text='Image Layer', fg='#ff5555', bg='#1a1a1a', font=('Arial', 10, 'bold')).grid(row=2, column=0, columnspan=3, sticky='w', padx=10, pady=(10, 0))
        tk.Button(frame_select, text='Select Real Image...', command=self.load_real, bg='#ff5555', fg='white').grid(row=3, column=0, padx=10, pady=5, sticky='ew')
        tk.Button(frame_select, text='Set Image Folder...', command=self.set_designated_image_folder, bg='#804c4c', fg='white').grid(row=3, column=1, padx=10, pady=5, sticky='ew')
        self.real_label = tk.Label(frame_select, text='No file', fg='#aaa', bg='#1a1a1a', anchor='w')
        self.real_label.grid(row=3, column=1, padx=10, pady=5, sticky='ew')
        frame_output = tk.LabelFrame(content, text='2. Output Settings', font=('Arial', 12), fg='#00ccff', bg='#1a1a1a', padx=10, pady=10)
        frame_output.pack(fill='x', padx=20, pady=10)
        tk.Label(frame_output, text='Width (px):', fg='#ccc', bg='#1a1a1a').grid(row=0, column=0, padx=10, pady=5, sticky='w')
        width_entry = tk.Entry(frame_output, textvariable=self.output_width, width=10, bg='#333', fg='#fff', insertbackground='#fff', relief='flat')
        width_entry.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        tk.Label(frame_output, text='Height (px):', fg='#ccc', bg='#1a1a1a').grid(row=0, column=2, padx=10, pady=5, sticky='w')
        height_entry = tk.Entry(frame_output, textvariable=self.output_height, width=10, bg='#333', fg='#fff', insertbackground='#fff', relief='flat')
        height_entry.grid(row=0, column=3, padx=10, pady=5, sticky='w')
        width_entry.bind('<FocusOut>', self.update_preview)
        width_entry.bind('<Return>', self.update_preview)
        height_entry.bind('<FocusOut>', self.update_preview)
        height_entry.bind('<Return>', self.update_preview)
        frame_adjust = tk.LabelFrame(content, text='3. Layer Adjustments', font=('Arial', 12), fg='#00ccff', bg='#1a1a1a', padx=10, pady=10)
        frame_adjust.pack(fill='x', padx=20, pady=10)
        frame_adjust.columnconfigure(1, weight=1)
        style = ttk.Style()
        style.configure('TCheckbutton', background='#1a1a1a', foreground='#00ccff', font=('Arial', 10, 'bold'))
        auto_bait_check = ttk.Checkbutton(frame_adjust, text='Automatic Pencil Sketch Bait', variable=self.use_auto_bait, command=self.toggle_auto_bait_controls, style='TCheckbutton')
        auto_bait_check.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky='w')
        self.auto_bait_frame = tk.Frame(frame_adjust, bg='#2a2a2a', relief='groove', borderwidth=2)
        self.auto_bait_frame.grid(row=1, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        self.auto_bait_frame.columnconfigure(1, weight=1)
        tk.Label(self.auto_bait_frame, text='Pencil Tip Size:', fg='#ccc', bg='#2a2a2a').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        tk.Scale(self.auto_bait_frame, from_=1, to=101, orient='horizontal', variable=self.pencil_tip_size, resolution=2, bg='#2a2a2a', fg='#00ccff', troughcolor='#444', command=self.update_preview).grid(row=0, column=1, sticky='ew', padx=5)
        tk.Label(self.auto_bait_frame, text='Sketch Range:', fg='#ccc', bg='#2a2a2a').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        tk.Scale(self.auto_bait_frame, from_=(-5.0), to=5.0, orient='horizontal', variable=self.pencil_range, resolution=0.1, bg='#2a2a2a', fg='#00ccff', troughcolor='#444', command=self.update_preview).grid(row=1, column=1, sticky='ew', padx=5)
        self.auto_bait_frame.grid_remove()
        tk.Label(frame_adjust, text='Bait Clear Color:', fg='#55ff55', bg='#1a1a1a').grid(row=2, column=0, padx=10, pady=5, sticky='w')
        tk.Button(frame_adjust, text='< Invert Image', command=self.invert_real_to_bait_color, bg='#ddd', fg='black').grid(row=2, column=1, padx=(0, 5), pady=5, sticky='e')
        bait_color_frame = tk.Frame(frame_adjust, bg='#1a1a1a')
        bait_color_frame.grid(row=2, column=2, columnspan=2, sticky='e', padx=(0, 10))
        self.bait_color_hex_entry = tk.Entry(bait_color_frame, width=7, bg='#333', fg='#fff', insertbackground='#fff', relief='flat')
        self.bait_color_hex_entry.pack(side='left', padx=(0, 5))
        self.bait_color_hex_entry.bind('<Return>', lambda e: self.update_color_from_hex('bait'))
        self.bait_color_hex_entry.bind('<FocusOut>', lambda e: self.update_color_from_hex('bait'))
        self.bait_color_preview = tk.Label(bait_color_frame, text='', bg=self.bait_clear_color, width=4, relief='sunken')
        self.bait_color_preview.pack(side='left', padx=(0, 5))
        tk.Button(bait_color_frame, text='Choose...', command=self.choose_bait_color).pack(side='left')
        bait_opacity_frame = tk.Frame(frame_adjust, bg='#1a1a1a')
        bait_opacity_frame.grid(row=3, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        bait_opacity_frame.columnconfigure(1, weight=1)
        tk.Label(bait_opacity_frame, text='Bait Opacity:', fg='#55ff55', bg='#1a1a1a').grid(row=0, column=0, sticky='w')
        EditableLabel(bait_opacity_frame, self.bait_opacity, bg='#1a1a1a', fg='#fff', width=4).grid(row=0, column=1, sticky='e', padx=(5, 0))
        tk.Scale(bait_opacity_frame, from_=0, to=255, orient='horizontal', variable=self.bait_opacity, bg='#1a1a1a', fg='#00ccff', troughcolor='#333', command=self.update_preview).grid(row=1, column=0, columnspan=2, sticky='ew')
        swap_button_frame = tk.Frame(frame_adjust, bg='#1a1a1a')
        swap_button_frame.grid(row=4, column=0, columnspan=4, pady=5)
        tk.Button(swap_button_frame, text='↕ Swap Colors ↕', command=self.swap_colors, bg='#444', fg='#ccc', relief='flat', padx=10).pack()
        ttk.Separator(frame_adjust, orient='horizontal').grid(row=5, column=0, columnspan=4, sticky='ew', pady=10)
        custom_method_check = ttk.Checkbutton(frame_adjust, text='Custom Layer Method', variable=self.use_custom_layer_method, command=self.toggle_custom_layer_controls, style='TCheckbutton')
        custom_method_check.grid(row=6, column=0, columnspan=4, padx=10, pady=5, sticky='w')
        self.custom_layers_frame = tk.Frame(frame_adjust, bg='#1a1a1a')
        self.custom_layers_frame.grid(row=7, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        controls_frame = tk.Frame(self.custom_layers_frame, bg='#1a1a1a')
        controls_frame.pack(fill='x')
        tk.Button(controls_frame, text='Add Layer', command=self.add_layer, bg='#00aa00', fg='white').pack(side='left', padx=5)
        tk.Button(controls_frame, text='Save Method', command=self.save_method, bg='#00aaff', fg='white').pack(side='left', padx=5)
        tk.Button(controls_frame, text='Load Method', command=self.load_method, bg='#ffaa00', fg='black').pack(side='left', padx=5)
        tk.Button(controls_frame, text='Import Methods', command=self.import_methods, bg='#9966ff', fg='white').pack(side='left', padx=5)
        self.layers_container = tk.Frame(self.custom_layers_frame, bg='#1a1a1a')
        self.layers_container.pack(fill='x', expand=True, pady=5)
        self.custom_layers_frame.grid_remove()
        tk.Label(frame_adjust, text='Image Clear Color:', fg='#ff5555', bg='#1a1a1a').grid(row=8, column=0, padx=10, pady=5, sticky='w')
        real_color_frame = tk.Frame(frame_adjust, bg='#1a1a1a')
        real_color_frame.grid(row=8, column=1, columnspan=3, sticky='e', padx=(0, 10))
        self.real_color_hex_entry = tk.Entry(real_color_frame, width=7, bg='#333', fg='#fff', insertbackground='#fff', relief='flat')
        self.real_color_hex_entry.pack(side='left', padx=(0, 5))
        self.real_color_hex_entry.bind('<Return>', lambda e: self.update_color_from_hex('real'))
        self.real_color_hex_entry.bind('<FocusOut>', lambda e: self.update_color_from_hex('real'))
        self.real_color_preview = tk.Label(real_color_frame, text='', bg=self.real_clear_color, width=4, relief='sunken')
        self.real_color_preview.pack(side='left', padx=(0, 5))
        self.real_color_button = tk.Button(real_color_frame, text='Choose...', command=self.choose_real_color)
        self.real_color_button.pack(side='left')
        real_opacity_frame = tk.Frame(frame_adjust, bg='#1a1a1a')
        real_opacity_frame.grid(row=9, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        real_opacity_frame.columnconfigure(1, weight=1)
        tk.Label(real_opacity_frame, text='Image Opacity (if not Custom):', fg='#ff5555', bg='#1a1a1a').grid(row=0, column=0, sticky='w')
        EditableLabel(real_opacity_frame, self.real_opacity, bg='#1a1a1a', fg='#fff', width=4).grid(row=0, column=1, sticky='e', padx=(5, 0))
        self.real_opacity_slider = tk.Scale(real_opacity_frame, from_=0, to=255, orient='horizontal', variable=self.real_opacity, bg='#1a1a1a', fg='#00ccff', troughcolor='#333', command=self.update_preview)
        self.real_opacity_slider.grid(row=1, column=0, columnspan=2, sticky='ew')
        ttk.Separator(frame_adjust, orient='horizontal').grid(row=10, column=0, columnspan=4, sticky='ew', pady=10)
        tk.Label(frame_adjust, text='Slip Mode:', fg='#ccc', bg='#1a1a1a').grid(row=11, column=0, padx=10, pady=5, sticky='w')
        slip_mode_options = ['None', 'Single Random Pixel', 'Random Amount of Pixels', 'All Non-Transparent Pixels']
        slip_mode_combo = ttk.Combobox(frame_adjust, textvariable=self.slip_mode_option, values=slip_mode_options, state='readonly')
        slip_mode_combo.grid(row=11, column=1, columnspan=3, sticky='ew', padx=10, pady=5)
        frame_watermark = tk.LabelFrame(content, text='4. Watermark', font=('Arial', 12), fg='#00ccff', bg='#1a1a1a', padx=10, pady=10)
        frame_watermark.pack(fill='x', padx=20, pady=10)
        tk.Checkbutton(frame_watermark, text='Enable Watermark', variable=self.use_watermark, bg='#1a1a1a', fg='#00ccff', selectcolor='#1a1a1a', command=self.update_preview).grid(row=0, column=0, columnspan=2, sticky='w', padx=5)
        tk.Label(frame_watermark, text='Type:', fg='#ccc', bg='#1a1a1a').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        watermark_type_combo = ttk.Combobox(frame_watermark, textvariable=self.watermark_type, values=['Text', 'Image'], state='readonly')
        watermark_type_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        watermark_type_combo.bind('<<ComboboxSelected>>', self.update_preview)
        self.watermark_text_frame = tk.Frame(frame_watermark, bg='#1a1a1a')
        self.watermark_text_frame.grid(row=2, column=0, columnspan=2, sticky='ew')
        tk.Label(self.watermark_text_frame, text='Text:', fg='#ccc', bg='#1a1a1a').pack(side='left', padx=5)
        tk.Entry(self.watermark_text_frame, textvariable=self.watermark_text, bg='#333', fg='#fff', insertbackground='#fff', relief='flat').pack(side='left', fill='x', expand=True, padx=5)
        self.watermark_image_frame = tk.Frame(frame_watermark, bg='#1a1a1a')
        self.watermark_image_frame.grid(row=2, column=0, columnspan=2, sticky='ew')
        tk.Button(self.watermark_image_frame, text='Select Watermark Image...', command=self.load_watermark_image, bg='#9966ff', fg='white').pack(side='left', padx=5)
        self.watermark_image_label = tk.Label(self.watermark_image_frame, text='No file selected', fg='#aaa', bg='#1a1a1a')
        self.watermark_image_label.pack(side='left', padx=5)
        tk.Label(frame_watermark, text='Size:', fg='#ccc', bg='#1a1a1a').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        tk.Scale(frame_watermark, from_=0.01, to=1.0, orient='horizontal', variable=self.watermark_size, resolution=0.01, bg='#1a1a1a', fg='#00ccff', troughcolor='#333', command=self.update_preview).grid(row=3, column=1, sticky='ew', padx=5)
        tk.Label(frame_watermark, text='Opacity:', fg='#ccc', bg='#1a1a1a').grid(row=4, column=0, sticky='w', padx=5, pady=5)
        tk.Scale(frame_watermark, from_=0, to=255, orient='horizontal', variable=self.watermark_opacity, bg='#1a1a1a', fg='#00ccff', troughcolor='#333', command=self.update_preview).grid(row=4, column=1, sticky='ew', padx=5)
        tk.Label(frame_watermark, text='Position:', fg='#ccc', bg='#1a1a1a').grid(row=5, column=0, sticky='w', padx=5, pady=5)
        position_combo = ttk.Combobox(frame_watermark, textvariable=self.watermark_position, values=['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right', 'Center'], state='readonly')
        position_combo.grid(row=5, column=1, sticky='ew', padx=5, pady=5)
        position_combo.bind('<<ComboboxSelected>>', self.update_preview)
        tk.Label(frame_watermark, text='Apply To:', fg='#ccc', bg='#1a1a1a').grid(row=6, column=0, sticky='w', padx=5, pady=5)
        target_combo = ttk.Combobox(frame_watermark, textvariable=self.watermark_target, values=['Bait', 'Image'], state='readonly')
        target_combo.grid(row=6, column=1, sticky='ew', padx=5, pady=5)
        target_combo.bind('<<ComboboxSelected>>', self.update_preview)
        self.toggle_watermark_controls()
        self.watermark_type.trace_add('write', self.toggle_watermark_controls)
        frame_preview = tk.LabelFrame(content, text='5. Preview', font=('Arial', 12), fg='#00ccff', bg='#1a1a1a', padx=10, pady=10)
        frame_preview.pack(padx=20, pady=10)
        custom_preview_frame = tk.Frame(frame_preview, bg='#1a1a1a')
        custom_preview_frame.pack(fill='x', pady=(0, 10))
        ttk.Checkbutton(custom_preview_frame, text='Enable Custom Preview Color', variable=self.use_custom_preview_color, command=self.toggle_custom_preview_controls, style='TCheckbutton').pack(side='left', padx=5)
        self.custom_preview_color_controls_frame = tk.Frame(custom_preview_frame, bg='#1a1a1a')
        self.custom_preview_color_controls_frame.pack(side='left', padx=10)
        self.custom_preview_color_hex_entry = tk.Entry(self.custom_preview_color_controls_frame, textvariable=self.custom_preview_color, width=7, bg='#333', fg='#fff', insertbackground='#fff', relief='flat')
        self.custom_preview_color_hex_entry.pack(side='left', padx=(0, 5))
        self.custom_preview_color_hex_entry.bind('<Return>', lambda e: self.update_color_from_hex('custom_preview'))
        self.custom_preview_color_hex_entry.bind('<FocusOut>', lambda e: self.update_color_from_hex('custom_preview'))
        self.custom_preview_color_preview = tk.Label(self.custom_preview_color_controls_frame, text='', bg=self.custom_preview_color.get(), width=4, relief='sunken')
        self.custom_preview_color_preview.pack(side='left', padx=(0, 5))
        tk.Button(self.custom_preview_color_controls_frame, text='Choose...', command=self.choose_custom_preview_color).pack(side='left')
        preview_container = tk.Frame(frame_preview, bg='#1a1a1a')
        preview_container.pack()
        left_preview_frame = tk.Frame(preview_container, bg='#1a1a1a')
        left_preview_frame.grid(row=0, column=0, padx=10, pady=5)
        self.canvas_bait_bg = tk.Canvas(left_preview_frame, bg=self.bait_clear_color, width=256, height=256, highlightthickness=0)
        self.canvas_bait_bg.pack()
        self.label_bait_bg = tk.Label(left_preview_frame, text=f'On Bait Color ({self.bait_clear_color.upper()})', font=('Arial', 10), fg='#ccc', bg='#1a1a1a')
        self.label_bait_bg.pack(pady=(5, 0))
        middle_preview_frame = tk.Frame(preview_container, bg='#1a1a1a')
        middle_preview_frame.grid(row=0, column=1, padx=10, pady=5)
        self.canvas_real_bg = tk.Canvas(middle_preview_frame, bg=self.real_clear_color, width=256, height=256, highlightthickness=0)
        self.canvas_real_bg.pack()
        self.label_real_bg = tk.Label(middle_preview_frame, text=f'On Real Color ({self.real_clear_color.upper()})', font=('Arial', 10), fg='#ccc', bg='#1a1a1a')
        self.label_real_bg.pack(pady=(5, 0))
        self.custom_preview_frame = tk.Frame(preview_container, bg='#1a1a1a')
        self.custom_preview_frame.grid(row=0, column=2, padx=10, pady=5)
        self.canvas_custom_bg = tk.Canvas(self.custom_preview_frame, bg=self.custom_preview_color.get(), width=256, height=256, highlightthickness=0)
        self.canvas_custom_bg.pack()
        self.label_custom_bg = tk.Label(self.custom_preview_frame, text=f'On Custom Color ({self.custom_preview_color.get().upper()})', font=('Arial', 10), fg='#ccc', bg='#1a1a1a')
        self.label_custom_bg.pack(pady=(5, 0))
        self.custom_preview_frame.grid_remove()
        frame_batch = tk.LabelFrame(content, text='6. Batch Export', font=('Arial', 12), fg='#00ccff', bg='#1a1a1a', padx=10, pady=10)
        frame_batch.pack(fill='x', padx=20, pady=10)
        frame_batch.columnconfigure(1, weight=1)
        tk.Label(frame_batch, text='Number of Copies:', fg='#ccc', bg='#1a1a1a').grid(row=0, column=0, padx=10, pady=5, sticky='w')
        tk.Entry(frame_batch, textvariable=self.batch_copies, width=10, bg='#333', fg='#fff', insertbackground='#fff', relief='flat').grid(row=0, column=1, padx=10, pady=5, sticky='w')
        tk.Label(frame_batch, text='Base Filename:', fg='#ccc', bg='#1a1a1a').grid(row=1, column=0, padx=10, pady=5, sticky='w')
        tk.Entry(frame_batch, textvariable=self.batch_basename, width=20, bg='#333', fg='#fff', insertbackground='#fff', relief='flat').grid(row=1, column=1, padx=10, pady=5, sticky='w')
        tk.Button(frame_batch, text='EXPORT BATCH', command=self.export_batch, bg='#ffaa00', fg='black', font=('Arial', 10, 'bold')).grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky='ns')
        info_label = tk.Label(frame_batch, text='Generates unique copies using \'Slip Mode\'.\nSaves to a new folder inside your system\'s Downloads folder.', fg='#999', bg='#1a1a1a', justify='left')
        info_label.grid(row=2, column=0, columnspan=3, padx=10, pady=(5, 0), sticky='w')
        frame_save = tk.Frame(content, bg='#1a1a1a')
        frame_save.pack(pady=20)
        tk.Button(frame_save, text='BUILD & SAVE SINGLE', command=self.build_and_save, bg='#00ff00', fg='black', font=('Arial', 12, 'bold')).pack(side='left', padx=10)
        tk.Button(frame_save, text='Set Output File...', command=self.save_as, bg='#0088ff', fg='white').pack(side='left', padx=10)
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.status_frame = tk.Frame(self.root, bg='#333333')
        self.status_frame.pack(side='bottom', fill='x')
        self.status = tk.Label(self.status_frame, text='Ready.', fg='#00ff99', bg='#333333')
        self.status.pack(pady=3, padx=10, side='left')
        self.toggle_auto_bait_controls()
        self.toggle_custom_preview_controls()

    def load_settings(self):
        """Loads settings from the settings file."""
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
                    self.output_width.set(settings.get('output_width', '300'))
                    self.output_height.set(settings.get('output_height', '300'))
                    self.bait_clear_color = settings.get('bait_clear_color', '#FFFFFF')
                    self.real_clear_color = settings.get('real_clear_color', '#000000')
                    self.slip_mode_option.set(settings.get('slip_mode_option', 'Single Random Pixel'))
                    self.batch_copies.set(settings.get('batch_copies', '10'))
                    self.batch_basename.set(settings.get('batch_basename', 'output_decal'))
                    self.use_auto_bait.set(settings.get('use_auto_bait', False))
                    self.pencil_tip_size.set(settings.get('pencil_tip_size', 21.0))
                    self.pencil_range.set(settings.get('pencil_range', (-1.5)))
                    self.designated_bait_folder.set(settings.get('designated_bait_folder', ''))
                    self.designated_image_folder.set(settings.get('designated_image_folder', ''))
                    self.use_custom_preview_color.set(settings.get('use_custom_preview_color', False))
                    loaded_color = settings.get('custom_preview_color')
                    if loaded_color and isinstance(loaded_color, str):
                        self.custom_preview_color.set(loaded_color)
            return
        except (IOError, json.JSONDecodeError) as e:
            print(f'Could not load settings: {e}')

    def save_settings(self, *args):
        """Saves the current settings to the settings file."""
        settings = {'output_width': self.output_width.get(), 'output_height': self.output_height.get(), 'bait_clear_color': self.bait_clear_color, 'real_clear_color': self.real_clear_color, 'slip_mode_option': self.slip_mode_option.get(), 'batch_copies': self.batch_copies.get(), 'batch_basename': self.batch_basename.get(), 'use_auto_bait': self.use_auto_bait.get(), 'pencil_tip_size': self.pencil_tip_size.get(), 'pencil_range': self.pencil_range.get(), 'designated_bait_folder': self.designated_bait_folder.get(), 'designated_image_folder': self.designated_image_folder.get(), 'use_custom_preview_color': self.use_custom_preview_color.get(), 'custom_preview_color': self.custom_preview_color.get()}
        try:
            with open(self.SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
        except IOError as e:
            print(f'Could not save settings: {e}')

    def check_for_updates_in_background(self):
        """Starts the update check in a separate thread to avoid freezing the UI."""
        update_thread = threading.Thread(target=self.check_for_updates, daemon=True)
        update_thread.start()

    def check_for_updates(self):
        """Fetches the latest version from GitHub and compares it with the current version."""
        try:
            with urllib.request.urlopen(self.VERSION_URL, timeout=5) as response:
                remote_version_str = response.read().decode('utf-8').strip()
                    local_parts = tuple(map(int, self.CURRENT_VERSION.split('.')))
                    remote_parts = tuple(map(int, remote_version_str.split('.')))
                    if remote_parts > local_parts:
                        message = f'A new version ({remote_version_str}) is available!\nYou are using version {self.CURRENT_VERSION}.\n\nWould you like to go to the GitHub page to download it?'
                        if messagebox.askyesno('Update Available', message):
                            webbrowser.open_new_tab(self.REPO_URL)
                        return
        except Exception as e:
            print(f'Could not check for updates: {e}')
            return None

    def choose_bait_color(self):
        color_code = colorchooser.askcolor(title='Choose Bait Clear Color', initialcolor=self.bait_clear_color)
        if color_code and color_code[1]:
            self.bait_clear_color = color_code[1]
            self.bait_color_preview.config(bg=self.bait_clear_color)
            self.update_hex_entries()
            self.update_preview()
            self.save_settings()
            return None

    def choose_real_color(self):
        color_code = colorchooser.askcolor(title='Choose Image Clear Color', initialcolor=self.real_clear_color)
        if color_code and color_code[1]:
            self.real_clear_color = color_code[1]
            self.real_color_preview.config(bg=self.real_clear_color)
            self.update_hex_entries()
            self.update_preview()
            self.save_settings()
            return None

    def update_color_from_hex(self, color_type):
        """Updates a color based on the content of its hex entry field."""
        try:
            if color_type == 'bait':
                hex_val = self.bait_color_hex_entry.get()
                if not hex_val.startswith('#'):
                    hex_val = '#' + hex_val
                self.bait_color_preview.config(bg=hex_val)
                self.bait_clear_color = hex_val
            self.update_hex_entries()
            self.update_preview()
            self.save_settings()
            self.status.config(text=f'Color set to {hex_val.upper()}', fg='#00ccff')
        except tk.TclError:
            self.status.config(text='Invalid hex color code.', fg='#ff5555')

    def update_hex_entries(self):
        """Updates the hex entry fields to match the current color variables."""
        self.bait_color_hex_entry.delete(0, tk.END)
        self.bait_color_hex_entry.insert(0, self.bait_clear_color)
        self.real_color_hex_entry.delete(0, tk.END)
        self.real_color_hex_entry.insert(0, self.real_clear_color)

    def choose_custom_preview_color(self):
        initial_color = self.custom_preview_color.get()
        if not initial_color:
            initial_color = '#808080'
        color_code = colorchooser.askcolor(title='Choose Custom Preview Color', initialcolor=initial_color)
        if color_code and color_code[1]:
            self.custom_preview_color.set(color_code[1])
            self.custom_preview_color_preview.config(bg=self.custom_preview_color.get())
            self.update_hex_entries()
            self.update_preview()
            self.save_settings()
            return None

    def toggle_custom_preview_controls(self):
        if self.use_custom_preview_color.get():
            self.custom_preview_color_controls_frame.pack(side='left', padx=10)
            self.custom_preview_frame.grid()
        self.update_preview()

    def invert_real_to_bait_color(self):
        try:
            real_rgb = self.hex_to_rgb(self.real_clear_color)
            inverted_rgb = [255 - c for c in real_rgb]
                inverted_hex = f'#{inverted_rgb[0]:02x}{inverted_rgb[1]:02x}{inverted_rgb[2]:02x}'
                self.bait_clear_color = inverted_hex
                self.bait_color_preview.config(bg=self.bait_clear_color)
                self.update_hex_entries()
                self.update_preview()
                self.save_settings()
                self.status.config(text=f'Bait color set to {inverted_hex.upper()}', fg='#00ccff')
        except Exception as e:
            self.status.config(text=f'Error inverting color: {e}', fg='#ff5555')

    def swap_colors(self):
        """Swaps the bait and real clear colors."""
        bait_c = self.bait_clear_color
        real_c = self.real_clear_color
        self.bait_clear_color = real_c
        self.real_clear_color = bait_c
        self.bait_color_preview.config(bg=self.bait_clear_color)
        self.real_color_preview.config(bg=self.real_clear_color)
        self.update_hex_entries()
        self.update_preview()
        self.save_settings()
        self.status.config(text='Colors swapped.', fg='#00ccff')

    def hex_to_rgb(self, hex_color):
        h = hex_color.lstrip('#')
        return [int(h[i:i + 2], 16) for i in [0, 2, 4]]

    def set_designated_bait_folder(self):
        """Opens a dialog to choose a permanent folder for bait images."""
        folder_path = filedialog.askdirectory(title='Select a Default Bait Folder')
        if folder_path:
            self.designated_bait_folder.set(folder_path)
            self.save_settings()
            self.status.config(text=f'Bait folder set to: {folder_path}', fg='#00ccff')
        return None

    def set_designated_image_folder(self):
        """Opens a dialog to choose a permanent folder for real images."""
        folder_path = filedialog.askdirectory(title='Select a Default Image Folder')
        if folder_path:
            self.designated_image_folder.set(folder_path)
            self.save_settings()
            self.status.config(text=f'Image folder set to: {folder_path}', fg='#00ccff')
        return None

    def load_bait(self):
        initial_dir = self.designated_bait_folder.get()
        initial_dir = Path.home() if not os.path.isdir(initial_dir) else initial_dir
        path = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[('Images', '*.png *.jpg *.jpeg')])
        if path:
            self.bait_path = path
            self.bait_label.config(text=os.path.basename(path))
            self.update_preview()
        return None

    def load_real(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg')])
        if path:
            self.real_path = path
            self.real_label.config(text=os.path.basename(path))
            self.update_preview()
        return None

    def handle_drop(self, event):
        file_path = event.data.strip('{}')
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.show_image_selection_popup(file_path)
        return None

    def show_image_selection_popup(self, image_path):
        self = tk.Toplevel(self.root)
        self.title('Select Image Type')
        self.geometry('300x100')
        self.transient(self.root)
        self.grab_set()
        label = tk.Label(self, text=f'How do you want to use \'{os.path.basename(image_path)}\'?', pady=10)
        label.pack()
        button_frame = tk.Frame(self)
        button_frame.pack(pady=5)

        def load_as_bait():
            self.bait_path = image_path
            self.bait_label.config(text=os.path.basename(image_path))
            self.update_preview()
            popup.destroy()

        def load_as_real():
            self.real_path = image_path
            self.real_label.config(text=os.path.basename(image_path))
            self.update_preview()
            popup.destroy()
        tk.Button(button_frame, text='Bait Image', command=load_as_bait).pack(side='left', padx=5)
        tk.Button(button_frame, text='Real Image', command=load_as_real).pack(side='left', padx=5)
        tk.Button(button_frame, text='Cancel', command=self.destroy).pack(side='left', padx=5)
        self.root.wait_window(self)

    def handle_paste(self, event):
        try:
            clipboard_content = self.root.clipboard_get()
            if os.path.isfile(clipboard_content) and clipboard_content.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.show_image_selection_popup(clipboard_content)
            return None
        except Exception as e:
            print(f'Error handling paste: {e}')

    def update_preview(self, event=None):
        try:
            self.canvas_bait_bg.config(bg=self.bait_clear_color)
            self.label_bait_bg.config(text=f'On Bait Color ({self.bait_clear_color.upper()})')
            self.canvas_real_bg.config(bg=self.real_clear_color)
            self.label_real_bg.config(text=f'On Real Color ({self.real_clear_color.upper()})')
            if self.use_custom_preview_color.get():
                custom_color = self.custom_preview_color.get()
                if not custom_color:
                    custom_color = '#808080'
                    self.custom_preview_color.set(custom_color)
                self.canvas_custom_bg.config(bg=custom_color)
                self.label_custom_bg.config(text=f'On Custom Color ({custom_color.upper()})')
        if self.real_path and self.bait_path:
            try:
                img = self.build_stego(apply_noise=False)
                if img is None:
                    pass
                return None
            self.root.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        except Exception:
            pass
        except Exception as e:
            self.status.config(text=f'Error: {e}', fg='#ff5555')
            import traceback
            traceback.print_exc()

    def _render_process_layer(self, settings_dict, base_img):
        """Renders a single process layer from a dictionary of settings."""
        layer_img = base_img.copy()
        if settings_dict.get('is_inverted'):
            if layer_img.mode == 'RGBA':
                rgb, alpha = (layer_img.convert('RGB'), layer_img.getchannel('A'))
                rgb = ImageOps.invert(rgb)
                rgb.putalpha(alpha)
                layer_img = rgb
        layer_img = layer_img.convert('L').convert('RGBA') if settings_dict.get('is_bw') else layer_img
        if settings_dict.get('use_color_blend'):
            blend_color_rgb = self.hex_to_rgb(settings_dict.get('blend_color', '#FFFFFF'))
            blend_alpha = settings_dict.get('blend_opacity', 0.5)
            if layer_img.mode == 'RGBA':
                rgb, alpha = (layer_img.convert('RGB'), layer_img.getchannel('A'))
                color_img = Image.new('RGB', layer_img.size, tuple(blend_color_rgb))
                blended_rgb = Image.blend(rgb, color_img, blend_alpha)
                blended_rgb.putalpha(alpha)
                layer_img = blended_rgb
        clear_color_hex = self.real_clear_color
        if settings_dict.get('invert_clear'):
            real_rgb = self.hex_to_rgb(self.real_clear_color)
            inverted_rgb = [255 - c for c in real_rgb]
            clear_color_hex = f'#{inverted_rgb[0]:02x}{inverted_rgb[1]:02x}{inverted_rgb[2]:02x}'
        layer_img = apply_color_clearer(layer_img, target_color=self.hex_to_rgb(clear_color_hex))
        opacity = settings_dict.get('opacity', 255)
        if opacity < 255:
            layer_img = layer_img.convert('RGBA') if layer_img.mode!= 'RGBA' else layer_img
            alpha = layer_img.getchannel('A')
            new_alpha_data = (np.array(alpha).astype(np.float32) * (opacity / 255.0)).astype(np.uint8)
            layer_img.putalpha(Image.fromarray(new_alpha_data))
        return layer_img

    def build_stego(self, apply_noise=True):
        try:
            width = int(self.output_width.get())
            height = int(self.output_height.get())
            raise ValueError('Dimensions must be positive.') if width <= 0 or height <= 0 else height
        except ValueError as Invalid size. Enter positive numbers.:
            pass
    def _process_bait_layer(self, bait_img, size):
        bait_img = bait_img.resize(size, Image.LANCZOS)
        if self.use_watermark.get() and self.watermark_target.get() == 'Bait':
            watermark_settings = {'use_watermark': self.use_watermark, 'type': self.watermark_type, 'text': self.watermark_text, 'image_path': self.watermark_image_path, 'size': self.watermark_size, 'opacity': self.watermark_opacity, 'position': self.watermark_position, 'target': self.watermark_target}
            apply_contrasting_color = self.watermark_type.get() == 'Text'
        if self.use_auto_bait.get():
            self.status.config(text='Creating bait sketch...', fg='#ffff00')
            self.root.update_idletasks()
            bait_rgb_np = np.array(bait_img.convert('RGB'))
            bait_bgr_np = cv2.cvtColor(bait_rgb_np, cv2.COLOR_RGB2BGR)
            tip_size = self.pencil_tip_size.get()
            range_param = self.pencil_range.get()
            sketch_np, msg = create_pencil_sketch(bait_bgr_np, tip_size, range_param)
            raise ValueError(f'Sketch creation failed: {msg}') if sketch_np is None else None
        self.status.config(text='Processing Bait Layer...', fg='#ffff00')
        self.root.update_idletasks()
        bait_target_rgb = self.hex_to_rgb(self.bait_clear_color)
        bait_processed = apply_color_clearer(bait_img, target_color=bait_target_rgb)
        bait_op_val = self.bait_opacity.get()
        if bait_op_val < 255:
            alpha = bait_processed.getchannel('A')
            new_alpha_data = (np.array(alpha).astype(np.float32) * (bait_op_val / 255.0)).astype(np.uint8)
            bait_processed.putalpha(Image.fromarray(new_alpha_data))
        return bait_processed

    def apply_slip_mode(self, arr):
        """Applies a small, unnoticeable change to pixel data to ensure a unique file hash."""
        mode = self.slip_mode_option.get()
        if mode == 'None':
            return arr

    def save_as(self):
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            self.output_path = path
        self.status.config(text=f'Output: {os.path.basename(path)}', fg='#00ff00')

    def save_bait_image(self):
        """Processes and saves only the bait layer."""
        if not self.bait_path:
            messagebox.showerror('Error', 'Select a bait image first!')
        return None

    def build_and_save(self):
        if not self.real_path or not self.bait_path:
            messagebox.showerror('Error', 'Select both images!')
        return None

    def export_batch(self):
        if not self.real_path or not self.bait_path:
            messagebox.showerror('Error', 'Select both a Bait and Real image first.')
        return None

    def read_methods_from_file(self):
        """Reads saved methods from methods.json, returns empty dict if error."""
        try:
            if os.path.exists('methods.json'):
                with open('methods.json', 'r') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            messagebox.showerror('Error Reading Methods', f'Could not read methods.json.\n{e}')

    def write_methods_to_file(self, methods_data):
        """Writes the given dictionary to methods.json."""
        try:
            with open('methods.json', 'w') as f:
                json.dump(methods_data, f, indent=4)
                    return True
        except IOError as e:
            messagebox.showerror('Error Saving Method', f'Could not write to methods.json.\n{e}')
            return False

    def save_method(self):
        """Saves the current custom layer setup as a named method."""
        if not self.custom_layers:
            messagebox.showwarning('Cannot Save', 'There are no layers to save.')
        return None

    def import_methods(self):
        """Imports methods from a JSON file and merges them with the current methods."""
        path = filedialog.askopenfilename(title='Select a methods JSON file to import', filetypes=[('JSON files', '*.json')])
        if not path:
            pass
        return None

    def rebuild_layers_ui(self):
        """Clears and rebuilds the entire custom layer UI from the self.custom_layers list."""
        layers_to_rebuild = list(self.custom_layers)
        for child in self.layers_container.winfo_children():
            child.destroy()
        self.custom_layers.clear()
        for layer_data in layers_to_rebuild:
            self.add_layer(existing_data=layer_data)

    def load_method(self):
        """Shows a dialog to load a saved method."""
        listbox = self.read_methods_from_file()
        if not listbox:
            messagebox.showinfo('No Saved Methods', 'There are no methods saved in methods.json.')
        return None

    def clear_layers(self):
        """Removes all custom layers from the UI and data list."""
        for layer in self.custom_layers:
            layer['frame'].destroy()
        self.custom_layers.clear()

    def toggle_custom_layer_controls(self):
        if self.use_custom_layer_method.get():
            self.custom_layers_frame.grid()
            self.real_opacity_slider.config(state='disabled')
        self.update_preview()

    def choose_layer_blend_color(self, color_var):
        """Opens a color chooser for a layer\'s blend color."""
        initial_color = color_var.get()
        color_code = colorchooser.askcolor(title='Choose Blend Color', initialcolor=initial_color)
        if color_code and color_code[1]:
            color_var.set(color_code[1])
            self.update_preview()
            return None

    def add_layer(self, existing_data=None):
        """Adds a new layer to the UI, either new or from existing data."""
        layer_index = len(self.custom_layers)
        frame = tk.Frame(self.layers_container, bg='#2a2a2a', relief='groove', borderwidth=1)
        frame.pack(fill='x', padx=5, pady=5)
        if existing_data:
            layer_data = existing_data
            layer_data['frame'] = frame
        top_row = tk.Frame(frame, bg='#2a2a2a')
        top_row.pack(fill='x')
        label = tk.Label(top_row, text=f'Layer {layer_index + 1}', fg='#00ccff', bg='#2a2a2a')
        label.pack(side='left', padx=5, pady=2)
        layer_data['label'] = label
        name_entry = tk.Entry(top_row, textvariable=layer_data['name'], bg='#333', fg='#fff', insertbackground='#fff', relief='flat', width=15)
        name_entry.pack(side='left', padx=5, pady=2)
        remove_button = tk.Button(top_row, text='Remove', command=lambda i=layer_index: self.remove_layer(i), bg='#ff5555', fg='white', relief='flat')
        remove_button.pack(side='right', padx=5, pady=2)
        down_button = tk.Button(top_row, text='▼', command=lambda i=layer_index: self.move_layer(i, 1), bg='#444', fg='white', relief='flat')
        down_button.pack(side='right', padx=(0, 5), pady=2)
        up_button = tk.Button(top_row, text='▲', command=lambda i=layer_index: self.move_layer(i, (-1)), bg='#444', fg='white', relief='flat')
        up_button.pack(side='right', padx=(5, 0), pady=2)
        layer_data.update({'up_button': up_button, 'down_button': down_button})
        if layer_data.get('type', 'process') == 'process':
            self._add_process_layer_controls(frame, layer_data, layer_index)
        if not existing_data:
            self.custom_layers.append(layer_data)
        self.update_layer_controls()
        self.update_preview()

    def _add_process_layer_controls(self, parent_frame, layer_data, layer_index):
        opacity_frame = tk.Frame(parent_frame, bg='#2a2a2a')
        opacity_frame.pack(fill='x', expand=True, padx=5)
        tk.Label(opacity_frame, text='Opacity:', fg='#ccc', bg='#2a2a2a').pack(side='left')
        EditableLabel(opacity_frame, layer_data['opacity'], bg='#2a2a2a', fg='#fff', width=4).pack(side='right', padx=(5, 0))
        tk.Scale(opacity_frame, from_=0, to=255, orient='horizontal', variable=layer_data['opacity'], bg='#2a2a2a', fg='#00ccff', troughcolor='#444', command=self.update_preview).pack(side='right', fill='x', expand=True)
        controls_row = tk.Frame(parent_frame, bg='#2a2a2a')
        controls_row.pack(fill='x', padx=5)
        ttk.Checkbutton(controls_row, text='B&W', variable=layer_data['is_bw'], command=self.update_preview).pack(side='left', padx=5)
        ttk.Checkbutton(controls_row, text='Invert', variable=layer_data['is_inverted'], command=self.update_preview).pack(side='left', padx=5)
        ttk.Checkbutton(controls_row, text='Invert Clear', variable=layer_data['invert_clear'], command=self.update_preview).pack(side='left', padx=5)
        merge_button = tk.Button(controls_row, text='Merge Down', command=lambda i=layer_index: self.merge_down(i), bg='#f0ad4e', fg='black')
        merge_button.pack(side='right', padx=5)
        layer_data['merge_button'] = merge_button
        blend_frame = tk.Frame(parent_frame, bg='#3a3a3a')
        blend_frame.pack(fill='x', expand=True, padx=5, pady=(0, 5))
        ttk.Checkbutton(blend_frame, text='Color Blend', variable=layer_data['use_color_blend'], command=self.update_preview).pack(side='left', padx=5)
        blend_color_button = tk.Button(blend_frame, text='Choose...', command=lambda c=layer_data['blend_color']: self.choose_layer_blend_color(c))
        blend_color_button.pack(side='left', padx=5)
        tk.Label(blend_frame, text='Blend:', fg='#ccc', bg='#3a3a3a').pack(side='left', padx=(10, 0))
        tk.Scale(blend_frame, from_=0.0, to=1.0, orient='horizontal', variable=layer_data['blend_opacity'], resolution=0.01, bg='#3a3a3a', fg='#00ccff', troughcolor='#555', command=self.update_preview).pack(side='left', fill='x', expand=True, padx=5)

    def _add_merged_layer_controls(self, parent_frame, layer_data, layer_index):
        opacity_frame = tk.Frame(parent_frame, bg='#2a2a2a')
        opacity_frame.pack(fill='x', expand=True, padx=5)
        tk.Label(opacity_frame, text='Group Opacity:', fg='#ccc', bg='#2a2a2a').pack(side='left')
        EditableLabel(opacity_frame, layer_data['opacity'], bg='#2a2a2a', fg='#fff', width=4).pack(side='right', padx=(5, 0))
        tk.Scale(opacity_frame, from_=0, to=255, orient='horizontal', variable=layer_data['opacity'], bg='#2a2a2a', fg='#00ccff', troughcolor='#444', command=self.update_preview).pack(side='right', fill='x', expand=True)
        info_frame = tk.Frame(parent_frame, bg='#2a2a2a')
        info_frame.pack(fill='x', padx=5, pady=5)
        sub_layer_count = len(layer_data.get('sub_layers', []))
        tk.Label(info_frame, text=f'Group of {sub_layer_count} layers', fg='#aaa', bg='#2a2a2a').pack(side='left', padx=5)
        ungroup_button = tk.Button(info_frame, text='Ungroup', command=lambda i=layer_index: self.ungroup_layer(i), bg='#5bc0de', fg='black')
        ungroup_button.pack(side='right', padx=5)
        merge_button = tk.Button(info_frame, text='Merge Down', command=lambda i=layer_index: self.merge_down(i), bg='#f0ad4e', fg='black')
        merge_button.pack(side='right', padx=5)
        layer_data['merge_button'] = merge_button

    def merge_down(self, index):
        if index >= len(self.custom_layers) - 1:
            pass
        return None

    def ungroup_layer(self, index):
        if self.custom_layers[index].get('type')!= 'merged':
            pass
        return None

    def remove_layer(self, index_to_remove):
        if 0 <= index_to_remove < len(self.custom_layers):
            self.custom_layers.pop(index_to_remove)
            self.rebuild_layers_ui()

    def move_layer(self, index_to_move, direction):
        """Moves a layer up or down in the list."""
        new_index = index_to_move + direction
        if 0 <= new_index < len(self.custom_layers):
            self.custom_layers[index_to_move], self.custom_layers[new_index] = (self.custom_layers[new_index], self.custom_layers[index_to_move])
            self.rebuild_layers_ui()

    def update_layer_controls(self):
        """Updates layer labels and button states (like enabling/disabling move/merge buttons)."""
        num_layers = len(self.custom_layers)
        for i, layer_data in enumerate(self.custom_layers):
            layer_data['label'].config(text=f'Layer {i + 1}')
            layer_data['up_button'].config(state='disabled' if i == 0 else 'normal')
            layer_data['down_button'].config(state='disabled' if i == num_layers - 1 else 'normal')
            if 'merge_button' in layer_data:
                pass
            else:
                layer_data['merge_button'].config(state='disabled' if i >= num_layers - 1 else 'normal')

    def toggle_auto_bait_controls(self):
        if self.use_auto_bait.get():
            self.auto_bait_frame.grid()
        self.update_preview()

    def toggle_watermark_controls(self, *args):
        """Shows/hides watermark controls based on type."""
        if self.watermark_type.get() == 'Text':
            self.watermark_text_frame.grid()
            self.watermark_image_frame.grid_remove()
        self.update_preview()

    def load_watermark_image(self):
        """Loads a watermark image from file."""
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png *.jpg *.jpeg')])
        if path:
            self.watermark_image_path.set(path)
            self.watermark_image_label.config(text=os.path.basename(path))
            self.update_preview()
        return None

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int((-1) * (event.delta / abs(event.delta)) * 3), 'units')
if __name__ == '__main__':
    root = tk.Tk()
    app = StegoDecalBuilder(root)
    root.mainloop()
