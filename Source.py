# source code use the exe version if you dont have python or it fails to load
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

# --- HELPER FUNCTIONS ---

def apply_color_clearer(src_img, target_color):
    """ The core steganography function to make an image 'clear' against a specific color. """
    src_arr = np.array(src_img).astype(np.float64)
    target_color = np.array(target_color, dtype=np.float64)
    Rf, Gf, Bf, Af = src_arr[:, :, 0], src_arr[:, :, 1], src_arr[:, :, 2], src_arr[:, :, 3]
    Rb, Gb, Bb = target_color[0], target_color[1], target_color[2]
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
        numerator = (Cc[mask] * 255.0) - (Cb * (255.0 - Ac[mask]))
        denominator = Ac[mask]
        new_Cf[mask] = numerator / denominator
        return new_Cf
    new_Rf = get_new_fg_color(Rc, Rb, Ac_final)
    new_Gf = get_new_fg_color(Gc, Gb, Ac_final)
    new_Bf = get_new_fg_color(Bc, Bb, Ac_final)
    final_arr = np.stack([
        np.clip(new_Rf, 0, 255), np.clip(new_Gf, 0, 255),
        np.clip(new_Bf, 0, 255), np.clip(Ac_final, 0, 255)
    ], axis=-1)
    return Image.fromarray(final_arr.astype(np.uint8), 'RGBA')

def tint_image(img, tint_color, opacity):
    """Tints an image with a given color and opacity, preserving its alpha channel."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Extract the alpha channel
    alpha = img.getchannel('A')

    # Create a solid color image with the tint color
    solid_color_img = Image.new('RGB', img.size, tint_color)

    # Convert the solid color image to RGBA and apply the original alpha channel
    tinted_img = solid_color_img.copy()
    tinted_img.putalpha(alpha)

    # Apply the desired opacity to the entire tinted image
    final_alpha = (np.array(alpha).astype(np.float32) * (opacity / 255.0)).astype(np.uint8)
    tinted_img.putalpha(Image.fromarray(final_alpha))

    return tinted_img

def apply_watermark(base_img, watermark_settings, bait_clear_color_hex=None, apply_contrasting_color_logic=False):
    """Applies a watermark to a given image based on the settings provided."""
    # Settings are now expected to be raw values, not Tkinter variables
    
    # Create a transparent layer to draw the watermark on
    watermark_layer = Image.new('RGBA', base_img.size, (0,0,0,0))
    draw = ImageDraw.Draw(watermark_layer)

    # Get watermark content
    if watermark_settings['type'] == "Text":
        try:
            # Calculate font size based on image width and size setting
            font_size = int(base_img.width * watermark_settings['size'])
            
            # Load font from selection or default to arial
            font_name = watermark_settings.get('font_family', 'arial.ttf')
            
            if os.path.isabs(font_name):
                font_path = font_name
            else:
                # Check standard Windows fonts
                font_path = os.path.join(os.environ['WINDIR'], 'Fonts', font_name)
                
                if not os.path.exists(font_path):
                    # Check user-specific fonts
                    local_appdata = os.environ.get('LOCALAPPDATA')
                    if local_appdata:
                        user_font_path = os.path.join(local_appdata, 'Microsoft', 'Windows', 'Fonts', font_name)
                        if os.path.exists(user_font_path):
                            font_path = user_font_path
                        else:
                            # Fallback to name-only if neither path works (let PIL try to find it)
                            font_path = font_name
            
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default() # Fallback font
        
        text = watermark_settings['text']
        
        # Use textbbox to get the bounding box of the text
        bbox = draw.textbbox((0, 0), text, font=font)
        content_width = bbox[2] - bbox[0]
        content_height = bbox[3] - bbox[1]

    else: # Image watermark
        try:
            watermark_img = Image.open(watermark_settings['image_path']).convert("RGBA")
            # Resize watermark based on base image size and size setting
            wm_width = int(base_img.width * watermark_settings['size'])
            aspect_ratio = watermark_img.height / watermark_img.width
            wm_height = int(wm_width * aspect_ratio)
            watermark_img = watermark_img.resize((wm_width, wm_height), Image.LANCZOS)
            content_width, content_height = watermark_img.size
        except Exception as e:
            print(f"Could not load watermark image: {e}")
            return base_img # Return original if watermark image fails

    # Position calculation
    if watermark_settings.get('use_specific_pos', False):
        x = watermark_settings.get('pos_x', 0)
        y = watermark_settings.get('pos_y', 0)
    else:
        pos = watermark_settings['position']
        margin = 10
        if pos == "Top Left":
            x, y = margin, margin
        elif pos == "Top Right":
            x, y = base_img.width - content_width - margin, margin
        elif pos == "Bottom Left":
            x, y = margin, base_img.height - content_height - margin
        elif pos == "Bottom Right":
            x, y = base_img.width - content_width - margin, base_img.height - content_height - margin
        else: # Center
            x, y = (base_img.width - content_width) // 2, (base_img.height - content_height) // 2

    # Apply opacity
    opacity = watermark_settings['opacity']
    
    if watermark_settings['type'] == "Text":
        # Determine text color
        text_color_hex = watermark_settings.get('text_color', '#FFFFFF')
        
        if apply_contrasting_color_logic and bait_clear_color_hex:
            # Calculate luminance of bait_clear_color
            h = bait_clear_color_hex.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
            if luminance > 128: # Light background, use black text
                text_color_rgba = (0, 0, 0, opacity)
            else: # Dark background, use white text
                text_color_rgba = (255, 255, 255, opacity)
        else:
             # Convert hex to RGB and add opacity
            h = text_color_hex.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            text_color_rgba = rgb + (opacity,)

        # Determine outline
        stroke_width = 0
        stroke_fill = None
        if watermark_settings.get('use_outline', False):
            stroke_width = int(font_size * 0.05) # 5% of font size
            if stroke_width < 1: stroke_width = 1
            
            outline_hex = watermark_settings.get('outline_color', '#000000')
            h = outline_hex.lstrip('#')
            rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            stroke_fill = rgb + (opacity,)

        draw.text((x, y), text, font=font, fill=text_color_rgba, stroke_width=stroke_width, stroke_fill=stroke_fill)
        
    else: # Image
        if apply_contrasting_color_logic and bait_clear_color_hex:
            # Convert bait_clear_color_hex to RGB tuple
            h = bait_clear_color_hex.lstrip('#')
            bait_rgb_tuple = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            
            watermark_img = tint_image(watermark_img, bait_rgb_tuple, opacity)
        else:
            # Apply opacity to the watermark if not tinted
            alpha = watermark_img.getchannel('A')
            new_alpha = (np.array(alpha).astype(np.float32) * (opacity / 255.0)).astype(np.uint8)
            watermark_img.putalpha(Image.fromarray(new_alpha))
        
        watermark_layer.paste(watermark_img, (x, y), watermark_img)

    return Image.alpha_composite(base_img, watermark_layer)


# --- MODIFICATION: Reusable Editable Label ---
class EditableLabel(tk.Frame):
    """
    A label that turns into an entry field on click, allowing for precise value setting.
    It's associated with a Tkinter variable (e.g., IntVar, StringVar) that it updates.
    """
    def __init__(self, parent, textvariable, **kwargs):
        # Separate frame-specific kwargs from label/entry-specific ones
        frame_kwargs = {'bg': kwargs.get('bg')}
        super().__init__(parent, **frame_kwargs)
        
        self.textvariable = textvariable
        
        self.label = tk.Label(self, textvariable=textvariable, **kwargs)
        self.label.pack(fill='both', expand=True)
        
        # The entry should also get the kwargs for consistency
        entry_kwargs = kwargs.copy()
        entry_kwargs['relief'] = 'sunken' # Make entry more visible
        entry_kwargs['borderwidth'] = 1
        self.entry = tk.Entry(self, textvariable=textvariable, **entry_kwargs)

        self.label.bind("<Button-1>", self.show_entry)
        self.entry.bind("<FocusOut>", self.show_label)
        self.entry.bind("<Return>", self.show_label)

    def show_entry(self, event=None):
        self.label.pack_forget()
        self.entry.pack(fill='both', expand=True)
        self.entry.focus_set()
        self.entry.select_range(0, 'end')

    def show_label(self, event=None):
        # This will trigger the variable's trace, if any, to update the UI
        self.entry.pack_forget()
        self.label.pack(fill='both', expand=True)
        # The linked textvariable is automatically updated by the Entry widget.
        # We just need to make sure the view switches back.

# --- Pencil Sketch Effect Functions ---
def adjust_levels(image, lower_bound, upper_bound):
    """ Adjusts the contrast of a grayscale image like Photoshop's Levels tool. """
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)
    if lower_bound >= upper_bound:
        return image
    lut = np.array([int(((i - lower_bound) / (upper_bound - lower_bound)) * 255) if lower_bound < i < upper_bound else (0 if i <= lower_bound else 255) for i in np.arange(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)

def create_pencil_sketch(image_bgr_np, pencil_tip_size=20, range_param=-1.5):
    """
    Creates a pencil sketch from a BGR numpy image array.
    This is MODIFIED to accept a numpy array instead of a file path.
    """
    if image_bgr_np is None:
        return None, "Input image data is invalid."

    gray_img = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2GRAY)
    inverted_gray_img = 255 - gray_img

    kernel_size = int(pencil_tip_size)
    if kernel_size % 2 == 0: kernel_size += 1
    if kernel_size < 1: kernel_size = 1

    blurred_img = cv2.GaussianBlur(inverted_gray_img, (kernel_size, kernel_size), 0)
    inverted_blurred_img = 255 - blurred_img
    inverted_blurred_img_safe = inverted_blurred_img + 1
    pencil_sketch_float = cv2.divide(gray_img, inverted_blurred_img_safe, scale=256.0)
    pencil_sketch = np.clip(pencil_sketch_float, 0, 255).astype(np.uint8)

    contrast_factor = 20
    lower_bound = 0 - (range_param * contrast_factor)
    upper_bound = 255 + (range_param * contrast_factor)

    final_sketch = adjust_levels(pencil_sketch, lower_bound, upper_bound)
    return final_sketch, "Success" # Returns a grayscale numpy array


# --- MAIN APPLICATION CLASS ---

class StegoDecalBuilder:
    # --- MODIFICATION: Added version constants ---
    CURRENT_VERSION = "2.2"
    # Note: Corrected URL to point to the direct raw content
    VERSION_URL = "https://raw.githubusercontent.com/AnnaRoblox/Roblox-Decal-Bypass-Maker/refs/heads/main/Version"
    REPO_URL = "https://github.com/AnnaRoblox/Roblox-Decal-Bypass-Maker/releases"
    SETTINGS_FILE = "settings.json"

    def __init__(self, root):
        self.root = root
        self.root.title(f"AnnaRoblox's Decal Bypass Maker v{self.CURRENT_VERSION}") # Add version to title
        self.root.geometry("800x1200")
        self.root.configure(bg="#1a1a1a")
        self.real_path = None
        self.bait_path = None
        self.output_path = None
        self.preview_img_tk = None

        # Standard variables
        self.output_width = tk.StringVar(value="300")
        self.output_height = tk.StringVar(value="300")
        self.bait_clear_color = "#FFFFFF"
        self.real_clear_color = "#000000"
        self.bait_opacity = tk.IntVar(value=255)
        self.real_opacity = tk.IntVar(value=255)
        self.slip_mode_option = tk.StringVar(value="Single Random Pixel")
        self.use_custom_layer_method = tk.BooleanVar(value=False)
        self.custom_layers = []

        # Batch variables
        self.batch_copies = tk.StringVar(value="10")
        self.batch_basename = tk.StringVar(value="output_decal")

        # Variables for automatic bait sketching
        self.use_auto_bait = tk.BooleanVar(value=False)
        self.pencil_tip_size = tk.DoubleVar(value=21.0)
        self.pencil_range = tk.DoubleVar(value=-1.5)
        self.designated_bait_folder = tk.StringVar(value="")
        self.designated_image_folder = tk.StringVar(value="")

        # Watermark variables (List of dicts)
        self.watermarks = []
        self.saved_watermarks_data = []

        # Custom Preview Color variables
        self.use_custom_preview_color = tk.BooleanVar(value=False)
        self.custom_preview_color = tk.StringVar(value="#808080") # Default to gray

        # New "Use Real Image Size" variable
        self.use_real_image_size = tk.BooleanVar(value=False)

        self.load_settings() # Load settings before setting up UI

        # --- MODIFICATION: Add traces to save settings automatically ---
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
        self.use_real_image_size.trace_add('write', self.save_settings)
        self.use_real_image_size.trace_add('write', self.toggle_size_inputs) # Toggle input state
        self.use_custom_layer_method.trace_add('write', self.save_settings)


        # Opacity is not in the settings list, but if you wanted to save it:
        # self.bait_opacity.trace_add('write', self.save_settings)
        # self.real_opacity.trace_add('write', self.save_settings)

        self.setup_ui()
        self.update_hex_entries()
        
        # --- MODIFICATION: Start update check in the background ---
        self.check_for_updates_in_background()

    def setup_ui(self):
        # Create a main frame to hold all content and allow scrolling
        self.main_frame = tk.Frame(self.root, bg="#1a1a1a")
        self.main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_frame, bg="#1a1a1a", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # --- MODIFICATION: Bind mouse wheel for scrolling ---
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # self.root.dnd_bind('<<Drop>>', self.handle_drop) # Bind drop event
        self.root.bind('<Control-v>', self.handle_paste) # Bind paste event

        content = tk.Frame(self.canvas, bg="#1a1a1a")
        self.canvas.create_window((0, 0), window=content, anchor="nw")

        tk.Label(content, text="AnnaRoblox's Decal Bypass Maker", font=("Arial", 24, "bold"), fg="#00ff99", bg="#1a1a1a").pack(pady=20)

        frame_select = tk.LabelFrame(content, text="1. Select Images", font=("Arial", 12), fg="#00ccff", bg="#1a1a1a", padx=10, pady=10)
        frame_select.pack(fill="x", padx=20, pady=10)
        frame_select.columnconfigure(1, weight=1) # Allow label to expand

        tk.Label(frame_select, text="Bait Layer", fg="#55ff55", bg="#1a1a1a", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", padx=10)
        
        bait_buttons_frame = tk.Frame(frame_select, bg="#1a1a1a")
        bait_buttons_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        tk.Button(bait_buttons_frame, text="Select Bait Image...", command=self.load_bait, bg="#55ff55", fg="black").pack(side="left")
        tk.Button(bait_buttons_frame, text="Set Bait Folder...", command=self.set_designated_bait_folder, bg="#4c806d", fg="white").pack(side="left", padx=10)
        tk.Button(bait_buttons_frame, text="Save Bait Image...", command=self.save_bait_image, bg="#5555ff", fg="white").pack(side="left", padx=10)

        self.bait_label = tk.Label(frame_select, text="No file", fg="#aaa", bg="#1a1a1a", anchor='w')
        self.bait_label.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        tk.Label(frame_select, text="Image Layer", fg="#ff5555", bg="#1a1a1a", font=("Arial", 10, "bold")).grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=(10,0))
        tk.Button(frame_select, text="Select Real Image...", command=self.load_real, bg="#ff5555", fg="white").grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        tk.Button(frame_select, text="Set Image Folder...", command=self.set_designated_image_folder, bg="#804c4c", fg="white").grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.real_label = tk.Label(frame_select, text="No file", fg="#aaa", bg="#1a1a1a", anchor='w')
        self.real_label.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        
        frame_output = tk.LabelFrame(content, text="2. Output Settings", font=("Arial", 12), fg="#00ccff", bg="#1a1a1a", padx=10, pady=10)
        frame_output.pack(fill="x", padx=20, pady=10)
        tk.Label(frame_output, text="Width (px):", fg="#ccc", bg="#1a1a1a").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.width_entry = tk.Entry(frame_output, textvariable=self.output_width, width=10, bg="#333", fg="#fff", insertbackground="#fff", relief="flat")
        self.width_entry.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        tk.Label(frame_output, text="Height (px):", fg="#ccc", bg="#1a1a1a").grid(row=0, column=2, padx=10, pady=5, sticky='w')
        self.height_entry = tk.Entry(frame_output, textvariable=self.output_height, width=10, bg="#333", fg="#fff", insertbackground="#fff", relief="flat")
        self.height_entry.grid(row=0, column=3, padx=10, pady=5, sticky='w')
        self.width_entry.bind("<FocusOut>", self.update_preview)
        self.width_entry.bind("<Return>", self.update_preview)
        self.height_entry.bind("<FocusOut>", self.update_preview)
        self.height_entry.bind("<Return>", self.update_preview)

        # "Use Real Image Size" Checkbox
        tk.Checkbutton(frame_output, text="Use Real Image Size", variable=self.use_real_image_size, bg="#1a1a1a", fg="#00ccff", selectcolor="#1a1a1a", command=self.update_preview).grid(row=1, column=0, columnspan=4, sticky='w', padx=10)


        frame_adjust = tk.LabelFrame(content, text="3. Layer Adjustments", font=("Arial", 12), fg="#00ccff", bg="#1a1a1a", padx=10, pady=10)
        frame_adjust.pack(fill="x", padx=20, pady=10)
        frame_adjust.columnconfigure(1, weight=1)

        style = ttk.Style()
        style.configure("TCheckbutton", background="#1a1a1a", foreground="#00ccff", font=('Arial', 10, 'bold'))
        
        auto_bait_check = ttk.Checkbutton(frame_adjust, text="Automatic Pencil Sketch Bait",
                                          variable=self.use_auto_bait, command=self.toggle_auto_bait_controls, style="TCheckbutton")
        auto_bait_check.grid(row=0, column=0, columnspan=4, padx=10, pady=5, sticky='w')

        self.auto_bait_frame = tk.Frame(frame_adjust, bg="#2a2a2a", relief="groove", borderwidth=2)
        self.auto_bait_frame.grid(row=1, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        self.auto_bait_frame.columnconfigure(1, weight=1)
        tk.Label(self.auto_bait_frame, text="Pencil Tip Size:", fg="#ccc", bg="#2a2a2a").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        tk.Scale(self.auto_bait_frame, from_=1, to=101, orient="horizontal", variable=self.pencil_tip_size, resolution=2, bg="#2a2a2a", fg="#00ccff", troughcolor="#444", command=self.update_preview).grid(row=0, column=1, sticky='ew', padx=5)
        tk.Label(self.auto_bait_frame, text="Sketch Range:", fg="#ccc", bg="#2a2a2a").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        tk.Scale(self.auto_bait_frame, from_=-5.0, to=5.0, orient="horizontal", variable=self.pencil_range, resolution=0.1, bg="#2a2a2a", fg="#00ccff", troughcolor="#444", command=self.update_preview).grid(row=1, column=1, sticky='ew', padx=5)
        self.auto_bait_frame.grid_remove()

        tk.Label(frame_adjust, text="Bait Clear Color:", fg="#55ff55", bg="#1a1a1a").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        tk.Button(frame_adjust, text="< Invert Image", command=self.invert_real_to_bait_color, bg="#ddd", fg="black").grid(row=2, column=1, padx=(0,5), pady=5, sticky='e')
        
        bait_color_frame = tk.Frame(frame_adjust, bg="#1a1a1a")
        bait_color_frame.grid(row=2, column=2, columnspan=2, sticky='e', padx=(0,10))
        
        self.bait_color_hex_entry = tk.Entry(bait_color_frame, width=7, bg="#333", fg="#fff", insertbackground="#fff", relief="flat")
        self.bait_color_hex_entry.pack(side='left', padx=(0,5))
        self.bait_color_hex_entry.bind("<Return>", lambda e: self.update_color_from_hex('bait'))
        self.bait_color_hex_entry.bind("<FocusOut>", lambda e: self.update_color_from_hex('bait'))

        self.bait_color_preview = tk.Label(bait_color_frame, text="", bg=self.bait_clear_color, width=4, relief="sunken")
        self.bait_color_preview.pack(side='left', padx=(0,5))
        tk.Button(bait_color_frame, text="Choose...", command=self.choose_bait_color).pack(side='left')

        bait_opacity_frame = tk.Frame(frame_adjust, bg="#1a1a1a")
        bait_opacity_frame.grid(row=3, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        bait_opacity_frame.columnconfigure(1, weight=1) # Allow slider to expand

        tk.Label(bait_opacity_frame, text="Bait Opacity:", fg="#55ff55", bg="#1a1a1a").grid(row=0, column=0, sticky='w')
        EditableLabel(bait_opacity_frame, self.bait_opacity, bg="#1a1a1a", fg="#fff", width=4).grid(row=0, column=1, sticky='e', padx=(5,0))
        tk.Scale(bait_opacity_frame, from_=0, to=255, orient="horizontal", variable=self.bait_opacity, bg="#1a1a1a", fg="#00ccff", troughcolor="#333", command=self.update_preview).grid(row=1, column=0, columnspan=2, sticky='ew')

        swap_button_frame = tk.Frame(frame_adjust, bg="#1a1a1a")
        swap_button_frame.grid(row=4, column=0, columnspan=4, pady=5)
        tk.Button(swap_button_frame, text="↕ Swap Colors ↕", command=self.swap_colors, bg="#444", fg="#ccc", relief="flat", padx=10).pack()

        ttk.Separator(frame_adjust, orient='horizontal').grid(row=5, column=0, columnspan=4, sticky='ew', pady=10)

        # --- Custom Layer Method ---
        custom_method_check = ttk.Checkbutton(frame_adjust, text="Custom Layer Method",
                                              variable=self.use_custom_layer_method, command=self.toggle_custom_layer_controls, style="TCheckbutton")
        custom_method_check.grid(row=6, column=0, columnspan=4, padx=10, pady=5, sticky='w')

        self.custom_layers_frame = tk.Frame(frame_adjust, bg="#1a1a1a")
        self.custom_layers_frame.grid(row=7, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        
        controls_frame = tk.Frame(self.custom_layers_frame, bg="#1a1a1a")
        controls_frame.pack(fill='x')
        tk.Button(controls_frame, text="Add Layer", command=self.add_layer, bg="#00aa00", fg="white").pack(side='left', padx=5)
        tk.Button(controls_frame, text="Save Method", command=self.save_method, bg="#00aaff", fg="white").pack(side='left', padx=5)
        tk.Button(controls_frame, text="Load Method", command=self.load_method, bg="#ffaa00", fg="black").pack(side='left', padx=5)
        tk.Button(controls_frame, text="Import Methods", command=self.import_methods, bg="#9966ff", fg="white").pack(side='left', padx=5)

        self.layers_container = tk.Frame(self.custom_layers_frame, bg="#1a1a1a")
        self.layers_container.pack(fill='x', expand=True, pady=5)
        
        self.custom_layers_frame.grid_remove() # Initially hidden
        # --- End Custom Layer Method ---

        tk.Label(frame_adjust, text="Image Clear Color:", fg="#ff5555", bg="#1a1a1a").grid(row=8, column=0, padx=10, pady=5, sticky='w')
        
        real_color_frame = tk.Frame(frame_adjust, bg="#1a1a1a")
        real_color_frame.grid(row=8, column=1, columnspan=3, sticky='e', padx=(0,10))

        self.real_color_hex_entry = tk.Entry(real_color_frame, width=7, bg="#333", fg="#fff", insertbackground="#fff", relief="flat")
        self.real_color_hex_entry.pack(side='left', padx=(0,5))
        self.real_color_hex_entry.bind("<Return>", lambda e: self.update_color_from_hex('real'))
        self.real_color_hex_entry.bind("<FocusOut>", lambda e: self.update_color_from_hex('real'))

        self.real_color_preview = tk.Label(real_color_frame, text="", bg=self.real_clear_color, width=4, relief="sunken")
        self.real_color_preview.pack(side='left', padx=(0,5))
        self.real_color_button = tk.Button(real_color_frame, text="Choose...", command=self.choose_real_color)
        self.real_color_button.pack(side='left')

        real_opacity_frame = tk.Frame(frame_adjust, bg="#1a1a1a")
        real_opacity_frame.grid(row=9, column=0, columnspan=4, sticky='ew', padx=10, pady=5)
        real_opacity_frame.columnconfigure(1, weight=1) # Allow slider to expand

        tk.Label(real_opacity_frame, text="Image Opacity (if not Custom):", fg="#ff5555", bg="#1a1a1a").grid(row=0, column=0, sticky='w')
        EditableLabel(real_opacity_frame, self.real_opacity, bg="#1a1a1a", fg="#fff", width=4).grid(row=0, column=1, sticky='e', padx=(5,0))
        self.real_opacity_slider = tk.Scale(real_opacity_frame, from_=0, to=255, orient="horizontal", variable=self.real_opacity, bg="#1a1a1a", fg="#00ccff", troughcolor="#333", command=self.update_preview)
        self.real_opacity_slider.grid(row=1, column=0, columnspan=2, sticky='ew')

        # --- MODIFICATION: Watermark Frame ---
        frame_watermark = tk.LabelFrame(content, text="4. Watermark", font=("Arial", 12), fg="#00ccff", bg="#1a1a1a", padx=10, pady=10)
        frame_watermark.pack(fill="x", padx=20, pady=10)
        
        tk.Button(frame_watermark, text="+ Add Watermark", command=self.add_watermark, bg="#00aa00", fg="white").pack(anchor='w', padx=5, pady=5)

        self.watermarks_container = tk.Frame(frame_watermark, bg="#1a1a1a")
        self.watermarks_container.pack(fill='x', expand=True, pady=5)

        # Load saved watermarks into UI
        for wm_data in self.saved_watermarks_data:
            self.add_watermark(existing_data=wm_data)

        frame_preview = tk.LabelFrame(content, text="5. Preview", font=("Arial", 12), fg="#00ccff", bg="#1a1a1a", padx=10, pady=10)
        frame_preview.pack(padx=20, pady=10)
        
        # Custom Preview Color Toggle and Controls
        custom_preview_frame = tk.Frame(frame_preview, bg="#1a1a1a")
        custom_preview_frame.pack(fill='x', pady=(0, 10))
        ttk.Checkbutton(custom_preview_frame, text="Enable Custom Preview Color",
                        variable=self.use_custom_preview_color, command=self.toggle_custom_preview_controls, style="TCheckbutton").pack(side='left', padx=5)
        
        self.custom_preview_color_controls_frame = tk.Frame(custom_preview_frame, bg="#1a1a1a")
        self.custom_preview_color_controls_frame.pack(side='left', padx=10)
        
        self.custom_preview_color_hex_entry = tk.Entry(self.custom_preview_color_controls_frame, textvariable=self.custom_preview_color, width=7, bg="#333", fg="#fff", insertbackground="#fff", relief="flat")
        self.custom_preview_color_hex_entry.pack(side='left', padx=(0,5))
        self.custom_preview_color_hex_entry.bind("<Return>", lambda e: self.update_color_from_hex('custom_preview'))
        self.custom_preview_color_hex_entry.bind("<FocusOut>", lambda e: self.update_color_from_hex('custom_preview'))

        self.custom_preview_color_preview = tk.Label(self.custom_preview_color_controls_frame, text="", bg=self.custom_preview_color.get(), width=4, relief="sunken")
        self.custom_preview_color_preview.pack(side='left', padx=(0,5))
        tk.Button(self.custom_preview_color_controls_frame, text="Choose...", command=self.choose_custom_preview_color).pack(side='left')


        preview_container = tk.Frame(frame_preview, bg="#1a1a1a")
        preview_container.pack()

        # Original Bait Color Preview
        left_preview_frame = tk.Frame(preview_container, bg="#1a1a1a")
        left_preview_frame.grid(row=0, column=0, padx=10, pady=5)
        self.canvas_bait_bg = tk.Canvas(left_preview_frame, bg=self.bait_clear_color, width=256, height=256, highlightthickness=0)
        self.canvas_bait_bg.pack()
        self.label_bait_bg = tk.Label(left_preview_frame, text=f"On Bait Color ({self.bait_clear_color.upper()})", font=("Arial", 10), fg="#ccc", bg="#1a1a1a")
        self.label_bait_bg.pack(pady=(5,0))

        # Original Real Color Preview
        middle_preview_frame = tk.Frame(preview_container, bg="#1a1a1a")
        middle_preview_frame.grid(row=0, column=1, padx=10, pady=5)
        self.canvas_real_bg = tk.Canvas(middle_preview_frame, bg=self.real_clear_color, width=256, height=256, highlightthickness=0)
        self.canvas_real_bg.pack()
        self.label_real_bg = tk.Label(middle_preview_frame, text=f"On Real Color ({self.real_clear_color.upper()})", font=("Arial", 10), fg="#ccc", bg="#1a1a1a")
        self.label_real_bg.pack(pady=(5,0))

        # New Custom Color Preview
        self.custom_preview_frame = tk.Frame(preview_container, bg="#1a1a1a")
        self.custom_preview_frame.grid(row=0, column=2, padx=10, pady=5)
        self.canvas_custom_bg = tk.Canvas(self.custom_preview_frame, bg=self.custom_preview_color.get(), width=256, height=256, highlightthickness=0)
        self.canvas_custom_bg.pack()
        self.label_custom_bg = tk.Label(self.custom_preview_frame, text=f"On Custom Color ({self.custom_preview_color.get().upper()})", font=("Arial", 10), fg="#ccc", bg="#1a1a1a")
        self.label_custom_bg.pack(pady=(5,0))
        self.custom_preview_frame.grid_remove() # Initially hidden

        frame_batch = tk.LabelFrame(content, text="6. Batch Export", font=("Arial", 12), fg="#00ccff", bg="#1a1a1a", padx=10, pady=10)
        frame_batch.pack(fill="x", padx=20, pady=10)
        frame_batch.columnconfigure(1, weight=1)
        tk.Label(frame_batch, text="Number of Copies:", fg="#ccc", bg="#1a1a1a").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        tk.Entry(frame_batch, textvariable=self.batch_copies, width=10, bg="#333", fg="#fff", insertbackground="#fff", relief="flat").grid(row=0, column=1, padx=10, pady=5, sticky='w')
        tk.Label(frame_batch, text="Base Filename:", fg="#ccc", bg="#1a1a1a").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        tk.Entry(frame_batch, textvariable=self.batch_basename, width=20, bg="#333", fg="#fff", insertbackground="#fff", relief="flat").grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        tk.Label(frame_batch, text="Slip Mode:", fg="#ccc", bg="#1a1a1a").grid(row=2, column=0, padx=10, pady=5, sticky='w')
        slip_mode_options = ["None", "Single Random Pixel", "Random Amount of Pixels", "All Non-Transparent Pixels"]
        slip_mode_combo = ttk.Combobox(frame_batch, textvariable=self.slip_mode_option, values=slip_mode_options, state="readonly")
        slip_mode_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=10, pady=5)

        tk.Button(frame_batch, text="EXPORT BATCH", command=self.export_batch, bg="#ffaa00", fg="black", font=("Arial", 10, "bold")).grid(row=0, column=2, rowspan=3, padx=10, pady=5, sticky='ns')
        info_label = tk.Label(frame_batch, text="Generates unique copies using 'Slip Mode'.\nSaves to a new folder inside your system's Downloads folder.", fg="#999", bg="#1a1a1a", justify="left")
        info_label.grid(row=3, column=0, columnspan=3, padx=10, pady=(5,0), sticky='w')

        frame_save = tk.Frame(content, bg="#1a1a1a")
        frame_save.pack(pady=20)
        tk.Button(frame_save, text="BUILD & SAVE SINGLE", command=self.build_and_save, bg="#00ff00", fg="black", font=("Arial", 12, "bold")).pack(side="left", padx=10)
        tk.Button(frame_save, text="Set Output File...", command=self.save_as, bg="#0088ff", fg="white").pack(side="left", padx=10)

        self.root.update_idletasks() # Ensure all widgets are updated
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))) # Add this line

        self.status_frame = tk.Frame(self.root, bg="#333333") # Initialize status_frame here
        self.status_frame.pack(side="bottom", fill="x")
        self.status = tk.Label(self.status_frame, text="Ready.", fg="#00ff99", bg="#333333") # Initialize status here
        self.status.pack(pady=3, padx=10, side="left")
        
        self.toggle_auto_bait_controls() # Ensure initial state is correct
        self.toggle_custom_preview_controls() # Ensure initial state for custom preview
        self.toggle_size_inputs() # Ensure initial state for size inputs
        self.toggle_custom_layer_controls() # Ensure initial state for custom layers

    def load_settings(self):
        """Loads settings from the settings file."""
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, "r") as f:
                    settings = json.load(f)
                    
                    # Set variables, with fallbacks for missing keys
                    self.output_width.set(settings.get('output_width', "300"))
                    self.output_height.set(settings.get('output_height', "300"))
                    self.bait_clear_color = settings.get('bait_clear_color', "#FFFFFF")
                    self.real_clear_color = settings.get('real_clear_color', "#000000")
                    self.slip_mode_option.set(settings.get('slip_mode_option', "Single Random Pixel"))
                    self.batch_copies.set(settings.get('batch_copies', "10"))
                    self.batch_basename.set(settings.get('batch_basename', "output_decal"))
                    self.use_auto_bait.set(settings.get('use_auto_bait', False))
                    self.pencil_tip_size.set(settings.get('pencil_tip_size', 21.0))
                    self.pencil_range.set(settings.get('pencil_range', -1.5))
                    self.designated_bait_folder.set(settings.get('designated_bait_folder', ""))
                    self.designated_image_folder.set(settings.get('designated_image_folder', ""))
                    self.use_custom_preview_color.set(settings.get('use_custom_preview_color', False))
                    self.use_real_image_size.set(settings.get('use_real_image_size', False))
                    self.use_custom_layer_method.set(settings.get('use_custom_layer_method', False))
                    
                    # Ensure a valid color is loaded, otherwise use default
                    loaded_color = settings.get('custom_preview_color')
                    if loaded_color and isinstance(loaded_color, str):
                        self.custom_preview_color.set(loaded_color)
                    else:
                        self.custom_preview_color.set("#808080")
                        
                    # Load watermarks list
                    self.saved_watermarks_data = settings.get('watermarks', [])

        except (IOError, json.JSONDecodeError) as e:
            print(f"Could not load settings: {e}") # Log error but don't crash

    def save_settings(self, *args):
        """Saves the current settings to the settings file."""
        # Serialize watermarks
        watermarks_data = []
        for wm in self.watermarks:
            wm_dict = {
                'type': wm['type'].get(),
                'text': wm['text'].get(),
                'font_family': wm['font_family'].get(),
                'image_path': wm['image_path'].get(),
                'size': wm['size'].get(),
                'opacity': wm['opacity'].get(),
                'position': wm['position'].get(),
                'target': wm['target'].get(),
                'text_color': wm['text_color'].get(),
                'auto_contrast': wm['auto_contrast'].get(),
                'use_outline': wm['use_outline'].get(),
                'outline_color': wm['outline_color'].get(),
                'use_specific_pos': wm['use_specific_pos'].get(),
                'pos_x': wm['pos_x'].get(),
                'pos_y': wm['pos_y'].get()
            }
            watermarks_data.append(wm_dict)

        settings = {
            'output_width': self.output_width.get(),
            'output_height': self.output_height.get(),
            'bait_clear_color': self.bait_clear_color,
            'real_clear_color': self.real_clear_color,
            'slip_mode_option': self.slip_mode_option.get(),
            'batch_copies': self.batch_copies.get(),
            'batch_basename': self.batch_basename.get(),
            'use_auto_bait': self.use_auto_bait.get(),
            'pencil_tip_size': self.pencil_tip_size.get(),
            'pencil_range': self.pencil_range.get(),
            'designated_bait_folder': self.designated_bait_folder.get(),
            'designated_image_folder': self.designated_image_folder.get(),
            'use_custom_preview_color': self.use_custom_preview_color.get(),
            'custom_preview_color': self.custom_preview_color.get(),
            'use_real_image_size': self.use_real_image_size.get(),
            'use_custom_layer_method': self.use_custom_layer_method.get(),
            'watermarks': watermarks_data
        }
        try:
            with open(self.SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=4)
        except IOError as e:
            print(f"Could not save settings: {e}") # Log error but don't crash

    def toggle_size_inputs(self, *args):
        """Enables or disables the width/height inputs based on the checkbox state."""
        state = 'disabled' if self.use_real_image_size.get() else 'normal'
        try:
            self.width_entry.config(state=state)
            self.height_entry.config(state=state)
        except AttributeError:
            pass # UI might not be fully initialized yet
    
    # --- Watermark Management Methods ---

    def add_watermark(self, existing_data=None):
        wm_index = len(self.watermarks)
        
        frame = tk.Frame(self.watermarks_container, bg="#2a2a2a", relief="groove", borderwidth=1)
        frame.pack(fill='x', padx=5, pady=5)
        
        # Initialize variables
        if existing_data:
            wm_data = {
                'frame': frame,
                'type': tk.StringVar(value=existing_data.get('type', "Text")),
                'text': tk.StringVar(value=existing_data.get('text', "Watermark")),
                'font_family': tk.StringVar(value=existing_data.get('font_family', "arial.ttf")),
                'image_path': tk.StringVar(value=existing_data.get('image_path', "")),
                'size': tk.DoubleVar(value=existing_data.get('size', 0.1)),
                'opacity': tk.IntVar(value=existing_data.get('opacity', 128)),
                'position': tk.StringVar(value=existing_data.get('position', "Center")),
                'target': tk.StringVar(value=existing_data.get('target', "Image")),
                'text_color': tk.StringVar(value=existing_data.get('text_color', "#FFFFFF")),
                'auto_contrast': tk.BooleanVar(value=existing_data.get('auto_contrast', True)),
                'use_outline': tk.BooleanVar(value=existing_data.get('use_outline', False)),
                'outline_color': tk.StringVar(value=existing_data.get('outline_color', "#000000")),
                'use_specific_pos': tk.BooleanVar(value=existing_data.get('use_specific_pos', False)),
                'pos_x': tk.IntVar(value=existing_data.get('pos_x', 0)),
                'pos_y': tk.IntVar(value=existing_data.get('pos_y', 0))
            }
        else:
            wm_data = {
                'frame': frame,
                'type': tk.StringVar(value="Text"),
                'text': tk.StringVar(value="Watermark"),
                'font_family': tk.StringVar(value="arial.ttf"),
                'image_path': tk.StringVar(value=""),
                'size': tk.DoubleVar(value=0.1),
                'opacity': tk.IntVar(value=128),
                'position': tk.StringVar(value="Center"),
                'target': tk.StringVar(value="Image"),
                'text_color': tk.StringVar(value="#FFFFFF"),
                'auto_contrast': tk.BooleanVar(value=True),
                'use_outline': tk.BooleanVar(value=False),
                'outline_color': tk.StringVar(value="#000000"),
                'use_specific_pos': tk.BooleanVar(value=False),
                'pos_x': tk.IntVar(value=0),
                'pos_y': tk.IntVar(value=0)
            }
        
        # Add trace to save settings on change
        for var in wm_data.values():
            if isinstance(var, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar)):
                 var.trace_add('write', self.save_settings)

        self.watermarks.append(wm_data)

        # UI Construction
        # Row 1: Type, Target, Remove
        row1 = tk.Frame(frame, bg="#2a2a2a")
        row1.pack(fill='x', padx=5, pady=2)
        
        ttk.Combobox(row1, textvariable=wm_data['type'], values=["Text", "Image"], state="readonly", width=8).pack(side='left', padx=2)
        tk.Label(row1, text="on", fg="#aaa", bg="#2a2a2a").pack(side='left')
        ttk.Combobox(row1, textvariable=wm_data['target'], values=["Bait", "Image"], state="readonly", width=8).pack(side='left', padx=2)
        
        tk.Button(row1, text="X", command=lambda i=wm_index: self.remove_watermark(i), bg="#ff5555", fg="white", font=("Arial", 8, "bold"), padx=5).pack(side='right')

        # Row 2: Content (Text or Image Path)
        row2 = tk.Frame(frame, bg="#2a2a2a")
        row2.pack(fill='x', padx=5, pady=2)
        
        # Text Entry
        text_entry = tk.Entry(row2, textvariable=wm_data['text'], bg="#333", fg="#fff", insertbackground="#fff", relief="flat")
        
        # Font Selection (Hidden by default, shown for Text)
        font_frame = tk.Frame(row2, bg="#2a2a2a")
        
        # Get system fonts (scan both System and User font dirs)
        system_fonts = ["arial.ttf"] # Fallback
        try:
            font_dirs = [os.path.join(os.environ['WINDIR'], 'Fonts')]
            local_appdata = os.environ.get('LOCALAPPDATA')
            if local_appdata:
                font_dirs.append(os.path.join(local_appdata, 'Microsoft', 'Windows', 'Fonts'))
            
            found_fonts = set()
            for f_dir in font_dirs:
                if os.path.exists(f_dir):
                    for f in os.listdir(f_dir):
                        if f.lower().endswith('.ttf'):
                            found_fonts.add(f)
            
            system_fonts = sorted(list(found_fonts))
        except Exception:
            pass

        font_combo = ttk.Combobox(font_frame, textvariable=wm_data['font_family'], values=system_fonts, state="readonly", width=15)
        font_combo.pack(side='right', padx=2)
        font_combo.bind("<<ComboboxSelected>>", self.update_preview)
        
        tk.Button(font_frame, text="📁", command=lambda: self.browse_font(wm_data), bg="#444", fg="white", font=("Arial", 8), padx=2).pack(side='right')
        
        wm_data['font_combo_frame'] = font_frame

        # Image Controls
        img_frame = tk.Frame(row2, bg="#2a2a2a")
        tk.Button(img_frame, text="Select Img", command=lambda: self.load_watermark_image_for_item(wm_data), bg="#9966ff", fg="white", font=("Arial", 8)).pack(side='left')
        img_label = tk.Label(img_frame, textvariable=wm_data['image_path'], fg="#aaa", bg="#2a2a2a", width=15, anchor='w')
        img_label.pack(side='left', padx=5)

        # Dynamic Content Visibility
        def update_content_ui(*args):
            if wm_data['type'].get() == "Text":
                img_frame.pack_forget()
                text_entry.pack(side='left', fill='x', expand=True)
                wm_data['font_combo_frame'].pack(side='right', padx=2)
                wm_data['text_color_btn'].config(state='normal') # Enable text options
                wm_data['auto_check'].config(state='normal')
                wm_data['outline_check'].config(state='normal')
            else:
                text_entry.pack_forget()
                wm_data['font_combo_frame'].pack_forget()
                img_frame.pack(fill='x', expand=True)
                wm_data['text_color_btn'].config(state='disabled') # Disable text options
                wm_data['auto_check'].config(state='disabled')
                wm_data['outline_check'].config(state='disabled')
            self.update_preview()

        wm_data['type'].trace_add('write', update_content_ui)
        wm_data['target'].trace_add('write', lambda *a: self.update_preview())
        wm_data['text'].trace_add('write', lambda *a: self.update_preview())

        # Row 3: Appearance (Size, Opacity)
        row3 = tk.Frame(frame, bg="#2a2a2a")
        row3.pack(fill='x', padx=5, pady=2)
        tk.Label(row3, text="Size:", fg="#ccc", bg="#2a2a2a", font=("Arial", 9)).pack(side='left')
        tk.Label(row3, textvariable=wm_data['size'], fg="#00ccff", bg="#2a2a2a", font=("Arial", 8), width=4).pack(side='left')
        tk.Scale(row3, from_=0.01, to=1.0, orient="horizontal", variable=wm_data['size'], resolution=0.01, bg="#2a2a2a", fg="#00ccff", troughcolor="#444", showvalue=0, command=self.update_preview).pack(side='left', fill='x', expand=True, padx=5)
        
        tk.Label(row3, text="Opacity:", fg="#ccc", bg="#2a2a2a", font=("Arial", 9)).pack(side='left')
        tk.Label(row3, textvariable=wm_data['opacity'], fg="#00ccff", bg="#2a2a2a", font=("Arial", 8), width=3).pack(side='left')
        tk.Scale(row3, from_=0, to=255, orient="horizontal", variable=wm_data['opacity'], bg="#2a2a2a", fg="#00ccff", troughcolor="#444", showvalue=0, command=self.update_preview).pack(side='left', fill='x', expand=True, padx=5)

        # Row 4: Text Details (Color, Outline)
        row4 = tk.Frame(frame, bg="#2a2a2a")
        row4.pack(fill='x', padx=5, pady=2)
        
        color_btn = tk.Button(row4, text="Color", bg=wm_data['text_color'].get(), fg="black", font=("Arial", 8), command=lambda: self.choose_watermark_color(wm_data, 'text_color', color_btn))
        color_btn.pack(side='left', padx=2)
        wm_data['text_color_btn'] = color_btn

        auto_check = ttk.Checkbutton(row4, text="Auto", variable=wm_data['auto_contrast'], command=self.update_preview)
        auto_check.pack(side='left', padx=5)
        wm_data['auto_check'] = auto_check
        
        outline_check = ttk.Checkbutton(row4, text="Outline", variable=wm_data['use_outline'], command=self.update_preview)
        outline_check.pack(side='left', padx=5)
        wm_data['outline_check'] = outline_check
        
        outline_color_btn = tk.Button(row4, bg=wm_data['outline_color'].get(), width=2, command=lambda: self.choose_watermark_color(wm_data, 'outline_color', outline_color_btn))
        outline_color_btn.pack(side='left')

        # Row 5: Positioning
        row5 = tk.Frame(frame, bg="#2a2a2a")
        row5.pack(fill='x', padx=5, pady=2)
        
        pos_combo = ttk.Combobox(row5, textvariable=wm_data['position'], values=["Top Left", "Top Right", "Bottom Left", "Bottom Right", "Center"], state="readonly", width=10)
        pos_combo.pack(side='left', padx=2)
        pos_combo.bind("<<ComboboxSelected>>", self.update_preview)
        
        specific_check = ttk.Checkbutton(row5, text="Specific Pos", variable=wm_data['use_specific_pos'])
        specific_check.pack(side='left', padx=5)
        
        spinbox_frame = tk.Frame(row5, bg="#2a2a2a")
        tk.Label(spinbox_frame, text="X:", fg="#ccc", bg="#2a2a2a").pack(side='left')
        tk.Spinbox(spinbox_frame, from_=-2000, to=2000, textvariable=wm_data['pos_x'], width=4, bg="#333", fg="#fff", command=self.update_preview).pack(side='left')
        tk.Label(spinbox_frame, text="Y:", fg="#ccc", bg="#2a2a2a").pack(side='left', padx=(5,0))
        tk.Spinbox(spinbox_frame, from_=-2000, to=2000, textvariable=wm_data['pos_y'], width=4, bg="#333", fg="#fff", command=self.update_preview).pack(side='left')

        def toggle_pos_mode(*args):
            if wm_data['use_specific_pos'].get():
                pos_combo.config(state='disabled')
                spinbox_frame.pack(side='left', padx=5)
            else:
                pos_combo.config(state='readonly')
                spinbox_frame.pack_forget()
            self.update_preview()
            
        wm_data['use_specific_pos'].trace_add('write', toggle_pos_mode)

        # Initial Layout Updates
        update_content_ui()
        toggle_pos_mode()
    
    def remove_watermark(self, index):
        if 0 <= index < len(self.watermarks):
            self.watermarks[index]['frame'].destroy()
            self.watermarks.pop(index)
            # Re-bind remaining remove buttons to correct indices
            for i, wm in enumerate(self.watermarks):
                 # Find the remove button (it's the last child of row1, which is the first child of frame)
                 # A bit hacky, but robust enough for this structure
                 row1 = wm['frame'].winfo_children()[0]
                 remove_btn = row1.winfo_children()[-1]
                 remove_btn.config(command=lambda idx=i: self.remove_watermark(idx))
            self.update_preview()
            self.save_settings()

    def load_watermark_image_for_item(self, wm_data):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            wm_data['image_path'].set(path)
            self.update_preview()

    def choose_watermark_color(self, wm_data, color_var_key, btn_widget):
        initial = wm_data[color_var_key].get()
        code = colorchooser.askcolor(title="Choose Color", initialcolor=initial)
        if code and code[1]:
            wm_data[color_var_key].set(code[1])
            btn_widget.config(bg=code[1])
            self.update_preview()

    def browse_font(self, wm_data):
        """Opens a file dialog to manually select a font file."""
        font_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
        path = filedialog.askopenfilename(
            initialdir=font_dir,
            title="Select Font File",
            filetypes=[("All Files", "*.*")]
        )
        if path:
            # If it's in the fonts folder, just store the filename for portability
            if path.lower().startswith(font_dir.lower()):
                wm_data['font_family'].set(os.path.basename(path))
            else:
                # Otherwise store the full path
                wm_data['font_family'].set(path)
            self.update_preview()

    # --- MODIFICATION: New methods for update checking ---
    def check_for_updates_in_background(self):
        """Starts the update check in a separate thread to avoid freezing the UI."""
        update_thread = threading.Thread(target=self.check_for_updates, daemon=True)
        update_thread.start()

    def check_for_updates(self):
        """Fetches the latest version from GitHub and compares it with the current version."""
        try:
            # Use a timeout to prevent long waits on poor connections
            with urllib.request.urlopen(self.VERSION_URL, timeout=5) as response:
                remote_version_str = response.read().decode('utf-8').strip()

            # Compare version parts as tuples of integers to handle "1.10" > "1.9"
            local_parts = tuple(map(int, self.CURRENT_VERSION.split('.')))
            remote_parts = tuple(map(int, remote_version_str.split('.')))

            if remote_parts > local_parts:
                message = (f"A new version ({remote_version_str}) is available!\n"
                           f"You are using version {self.CURRENT_VERSION}.\n\n"
                           "Would you like to go to the GitHub page to download it?")
                if messagebox.askyesno("Update Available", message):
                    webbrowser.open_new_tab(self.REPO_URL)
        except Exception as e:
            # Fail silently on any error (e.g., no internet), but print for debugging
            print(f"Could not check for updates: {e}")

    def choose_bait_color(self):
        color_code = colorchooser.askcolor(title="Choose Bait Clear Color", initialcolor=self.bait_clear_color)
        if color_code and color_code[1]:
            self.bait_clear_color = color_code[1]
            self.bait_color_preview.config(bg=self.bait_clear_color)
            self.update_hex_entries()
            self.update_preview()
            self.save_settings()

    def choose_real_color(self):
        color_code = colorchooser.askcolor(title="Choose Image Clear Color", initialcolor=self.real_clear_color)
        if color_code and color_code[1]:
            self.real_clear_color = color_code[1]
            self.real_color_preview.config(bg=self.real_clear_color)
            self.update_hex_entries()
            self.update_preview()
            self.save_settings()

    def update_color_from_hex(self, color_type):
        """Updates a color based on the content of its hex entry field."""
        try:
            if color_type == 'bait':
                hex_val = self.bait_color_hex_entry.get()
                if not hex_val.startswith('#'): hex_val = '#' + hex_val
                # The following line will raise a TclError if the color is invalid
                self.bait_color_preview.config(bg=hex_val) 
                self.bait_clear_color = hex_val
            elif color_type == 'real':
                hex_val = self.real_color_hex_entry.get()
                if not hex_val.startswith('#'): hex_val = '#' + hex_val
                self.real_color_preview.config(bg=hex_val)
                self.real_clear_color = hex_val
            elif color_type == 'custom_preview':
                hex_val = self.custom_preview_color_hex_entry.get()
                if not hex_val.startswith('#'): hex_val = '#' + hex_val
                self.custom_preview_color_preview.config(bg=hex_val)
                self.custom_preview_color.set(hex_val)
            
            self.update_hex_entries() # Ensure formatting is consistent
            self.update_preview()
            self.save_settings()
            self.status.config(text=f"Color set to {hex_val.upper()}", fg="#00ccff")
        except tk.TclError:
            self.status.config(text="Invalid hex color code.", fg="#ff5555")

    def update_hex_entries(self):
        """Updates the hex entry fields to match the current color variables."""
        self.bait_color_hex_entry.delete(0, tk.END)
        self.bait_color_hex_entry.insert(0, self.bait_clear_color)
        self.real_color_hex_entry.delete(0, tk.END)
        self.real_color_hex_entry.insert(0, self.real_clear_color)
        # custom_preview_color_hex_entry is bound to a StringVar, so no manual update needed.

    def choose_custom_preview_color(self):
        initial_color = self.custom_preview_color.get()
        if not initial_color:
            initial_color = "#808080"
        color_code = colorchooser.askcolor(title="Choose Custom Preview Color", initialcolor=initial_color)
        if color_code and color_code[1]:
            self.custom_preview_color.set(color_code[1])
            self.custom_preview_color_preview.config(bg=self.custom_preview_color.get())
            self.update_hex_entries()
            self.update_preview()
            self.save_settings()

    def toggle_custom_preview_controls(self):
        if self.use_custom_preview_color.get():
            self.custom_preview_color_controls_frame.pack(side='left', padx=10)
            self.custom_preview_frame.grid()
        else:
            self.custom_preview_color_controls_frame.pack_forget()
            self.custom_preview_frame.grid_remove()
        self.update_preview()

    def invert_real_to_bait_color(self):
        try:
            real_rgb = self.hex_to_rgb(self.real_clear_color)
            inverted_rgb = [255 - c for c in real_rgb]
            inverted_hex = f"#{inverted_rgb[0]:02x}{inverted_rgb[1]:02x}{inverted_rgb[2]:02x}"
            self.bait_clear_color = inverted_hex
            self.bait_color_preview.config(bg=self.bait_clear_color)
            self.update_hex_entries()
            self.update_preview()
            self.save_settings()
            self.status.config(text=f"Bait color set to {inverted_hex.upper()}", fg="#00ccff")
        except Exception as e:
            self.status.config(text=f"Error inverting color: {e}", fg="#ff5555")

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
        self.status.config(text="Colors swapped.", fg="#00ccff")

    def hex_to_rgb(self, hex_color):
        h = hex_color.lstrip('#')
        return [int(h[i:i+2], 16) for i in (0, 2, 4)]

    def set_designated_bait_folder(self):
        """Opens a dialog to choose a permanent folder for bait images."""
        folder_path = filedialog.askdirectory(title="Select a Default Bait Folder")
        if folder_path:
            self.designated_bait_folder.set(folder_path)
            self.save_settings()
            self.status.config(text=f"Bait folder set to: {folder_path}", fg="#00ccff")

    def set_designated_image_folder(self):
        """Opens a dialog to choose a permanent folder for real images."""
        folder_path = filedialog.askdirectory(title="Select a Default Image Folder")
        if folder_path:
            self.designated_image_folder.set(folder_path)
            self.save_settings()
            self.status.config(text=f"Image folder set to: {folder_path}", fg="#00ccff")

    def load_bait(self):
        initial_dir = self.designated_bait_folder.get()
        if not os.path.isdir(initial_dir):
            initial_dir = Path.home() # Fallback to home directory

        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("Images", "*.png *.jpg *.jpeg")]
        )
        if path:
            self.bait_path = path
            self.bait_label.config(text=os.path.basename(path))
            self.update_preview()

    def load_real(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.real_path = path
            self.real_label.config(text=os.path.basename(path))
            self.update_preview()

    def handle_drop(self, event):
        # Tkinter's dnd events provide file paths as a string, often with curly braces
        file_path = event.data.strip('{}')
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.show_image_selection_popup(file_path)
        else:
            messagebox.showwarning("Invalid File", "Please drop a valid image file (PNG, JPG, JPEG).")

    def show_image_selection_popup(self, image_path):
        popup = tk.Toplevel(self.root)
        popup.title("Select Image Type")
        popup.geometry("300x100")
        popup.transient(self.root) # Make it appear on top of the main window
        popup.grab_set() # Disable interaction with the main window

        label = tk.Label(popup, text=f"How do you want to use '{os.path.basename(image_path)}'?", pady=10)
        label.pack()

        button_frame = tk.Frame(popup)
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

        tk.Button(button_frame, text="Bait Image", command=load_as_bait).pack(side="left", padx=5)
        tk.Button(button_frame, text="Real Image", command=load_as_real).pack(side="left", padx=5)
        tk.Button(button_frame, text="Cancel", command=popup.destroy).pack(side="left", padx=5)

        self.root.wait_window(popup) # Wait for the popup to close

    def handle_paste(self, event):
        try:
            # Get clipboard content as a string (might be a file path)
            clipboard_content = self.root.clipboard_get()
            
            if os.path.isfile(clipboard_content) and clipboard_content.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.show_image_selection_popup(clipboard_content)
                return

            # Try to open as an image directly from clipboard data
            img = Image.open(io.BytesIO(self.root.clipboard_get(type='PNG'))) # Try PNG first
            
            # Save the pasted image to a temporary file
            temp_dir = Path.home() / ".decal_bypass_maker_temp"
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / "pasted_image.png"
            img.save(temp_file_path)
            
            self.show_image_selection_popup(str(temp_file_path))

        except Exception as e:
            print(f"Error handling paste: {e}")

    def update_preview(self, event=None):
        # 1. Update Background Colors & Labels (Always)
        try:
            # Bait Preview Updates
            self.canvas_bait_bg.config(bg=self.bait_clear_color)
            self.label_bait_bg.config(text=f"On Bait Color ({self.bait_clear_color.upper()})")

            # Real Preview Updates
            self.canvas_real_bg.config(bg=self.real_clear_color)
            self.label_real_bg.config(text=f"On Real Color ({self.real_clear_color.upper()})")

            # Custom Preview Updates
            if self.use_custom_preview_color.get():
                custom_color = self.custom_preview_color.get()
                if not custom_color:
                    custom_color = "#808080"
                    self.custom_preview_color.set(custom_color)
                self.canvas_custom_bg.config(bg=custom_color)
                self.label_custom_bg.config(text=f"On Custom Color ({custom_color.upper()})")
        except Exception:
            pass # Ignore errors if UI isn't fully ready

        # 2. Update Image Content (Only if images are loaded)
        if self.real_path and self.bait_path:
            try:
                img = self.build_stego(apply_noise=False)
                if img is None: return

                preview_size = (256, 256)
                img_preview = img.resize(preview_size, Image.NEAREST)
                self.preview_img_tk = ImageTk.PhotoImage(img_preview)

                self.canvas_bait_bg.delete("all")
                self.canvas_bait_bg.create_image(128, 128, anchor="center", image=self.preview_img_tk)

                self.canvas_real_bg.delete("all")
                self.canvas_real_bg.create_image(128, 128, anchor="center", image=self.preview_img_tk)

                if self.use_custom_preview_color.get():
                    self.canvas_custom_bg.delete("all")
                    self.canvas_custom_bg.create_image(128, 128, anchor="center", image=self.preview_img_tk)

                self.status.config(text="Preview updated!", fg="#00ff99")
            except Exception as e:
                self.status.config(text=f"Error: {e}", fg="#ff5555")
                import traceback
                traceback.print_exc()
        
        # Explicitly update the scrollregion after content changes
        self.root.update_idletasks() # Ensure all widgets are updated
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _render_process_layer(self, settings_dict, base_img):
        """Renders a single process layer from a dictionary of settings."""
        layer_img = base_img.copy()
        
        if settings_dict.get('is_inverted'):
            if layer_img.mode == 'RGBA':
                rgb, alpha = layer_img.convert('RGB'), layer_img.getchannel('A')
                rgb = ImageOps.invert(rgb)
                rgb.putalpha(alpha)
                layer_img = rgb
            else:
                layer_img = ImageOps.invert(layer_img)

        if settings_dict.get('is_bw'):
            layer_img = layer_img.convert('L').convert('RGBA')
        
        if settings_dict.get('use_color_blend'):
            blend_color_rgb = self.hex_to_rgb(settings_dict.get('blend_color', '#FFFFFF'))
            blend_alpha = settings_dict.get('blend_opacity', 0.5)
            if layer_img.mode == 'RGBA':
                rgb, alpha = layer_img.convert('RGB'), layer_img.getchannel('A')
                color_img = Image.new('RGB', layer_img.size, tuple(blend_color_rgb))
                blended_rgb = Image.blend(rgb, color_img, blend_alpha)
                blended_rgb.putalpha(alpha)
                layer_img = blended_rgb

        clear_color_hex = self.real_clear_color
        if settings_dict.get('invert_clear'):
            real_rgb = self.hex_to_rgb(self.real_clear_color)
            inverted_rgb = [255 - c for c in real_rgb]
            clear_color_hex = f"#{inverted_rgb[0]:02x}{inverted_rgb[1]:02x}{inverted_rgb[2]:02x}"
        
        layer_img = apply_color_clearer(layer_img, target_color=self.hex_to_rgb(clear_color_hex))

        opacity = settings_dict.get('opacity', 255)
        if opacity < 255:
            if layer_img.mode != 'RGBA': layer_img = layer_img.convert('RGBA')
            alpha = layer_img.getchannel('A')
            new_alpha_data = (np.array(alpha).astype(np.float32) * (opacity / 255.0)).astype(np.uint8)
            layer_img.putalpha(Image.fromarray(new_alpha_data))
            
        return layer_img

    def build_stego(self, apply_noise=True):
        try:
            if not self.real_path or not self.bait_path: return None

            bait_img = Image.open(self.bait_path).convert("RGBA")
            real_img = Image.open(self.real_path).convert("RGBA")

            if self.use_real_image_size.get():
                width, height = real_img.size
            else:
                width = int(self.output_width.get())
                height = int(self.output_height.get())
            
            if width <= 0 or height <= 0: raise ValueError("Dimensions must be positive.")
            size = (width, height)
        except ValueError:
            raise ValueError("Invalid size. Enter positive numbers.")

        bait_processed = self._process_bait_layer(bait_img, size)
        real_img = real_img.resize(size, Image.LANCZOS)

        self.status.config(text="Processing Image Layer...", fg="#ffff00")
        self.root.update_idletasks()

        if self.use_custom_layer_method.get() and self.custom_layers:
            composite_image = Image.new('RGBA', size, (0, 0, 0, 0))
            
            for layer_data in self.custom_layers:
                layer_type = layer_data.get('type', 'process')
                
                if layer_type == 'process':
                    # For a process layer, convert its tk.Vars to a settings dict
                    settings = {
                        'name': layer_data['name'].get(), 'opacity': layer_data['opacity'].get(),
                        'is_bw': layer_data['is_bw'].get(), 'is_inverted': layer_data['is_inverted'].get(),
                        'invert_clear': layer_data['invert_clear'].get(), 'use_color_blend': layer_data['use_color_blend'].get(),
                        'blend_color': layer_data['blend_color'].get(), 'blend_opacity': layer_data['blend_opacity'].get()
                    }
                    layer_img = self._render_process_layer(settings, real_img)
                
                elif layer_type == 'merged':
                    # For a merged group, render its sub-layers onto a temporary composite
                    group_composite = Image.new('RGBA', size, (0, 0, 0, 0))
                    for sub_layer_settings in layer_data.get('sub_layers', []):
                        sub_layer_img = self._render_process_layer(sub_layer_settings, real_img)
                        group_composite = Image.alpha_composite(group_composite, sub_layer_img)
                    
                    # Apply the group's master opacity
                    master_opacity = layer_data['opacity'].get()
                    if master_opacity < 255:
                        alpha = group_composite.getchannel('A')
                        new_alpha = (np.array(alpha).astype(np.float32) * (master_opacity / 255.0)).astype(np.uint8)
                        group_composite.putalpha(Image.fromarray(new_alpha))
                    
                    layer_img = group_composite

                composite_image = Image.alpha_composite(composite_image, layer_img)
            
            intermediate_real = composite_image
            real_target_rgb = self.hex_to_rgb(self.real_clear_color)
            real_processed = apply_color_clearer(intermediate_real, target_color=real_target_rgb)

        else: # Fallback to original method
            real_target_rgb = self.hex_to_rgb(self.real_clear_color)
            real_processed = apply_color_clearer(real_img, target_color=real_target_rgb)
            real_op_val = self.real_opacity.get()
            if real_op_val < 255:
                alpha = real_processed.getchannel('A')
                new_alpha_data = (np.array(alpha).astype(np.float32) * (real_op_val / 255.0)).astype(np.uint8)
                real_processed.putalpha(Image.fromarray(new_alpha_data))

        # Apply watermarks targeted for "Image"
        for wm in self.watermarks:
            if wm['target'].get() == "Image":
                # Convert TkVars to raw values for helper function
                settings = {
                    'type': wm['type'].get(),
                    'text': wm['text'].get(),
                    'font_family': wm['font_family'].get(),
                    'image_path': wm['image_path'].get(),
                    'size': wm['size'].get(),
                    'opacity': wm['opacity'].get(),
                    'position': wm['position'].get(),
                    'text_color': wm['text_color'].get(),
                    'auto_contrast': wm['auto_contrast'].get(),
                    'use_outline': wm['use_outline'].get(),
                    'outline_color': wm['outline_color'].get(),
                    'use_specific_pos': wm['use_specific_pos'].get(),
                    'pos_x': wm['pos_x'].get(),
                    'pos_y': wm['pos_y'].get()
                }
                real_processed = apply_watermark(real_processed, settings, apply_contrasting_color_logic=settings['auto_contrast'])

        self.status.config(text="Compositing...", fg="#ffff00")
        self.root.update_idletasks()
        result_img = Image.alpha_composite(bait_processed, real_processed)

        if apply_noise and self.slip_mode_option.get() != "None":
            result_arr = self.apply_slip_mode(np.array(result_img))
            return Image.fromarray(result_arr)
        else:
            return result_img

    def _process_bait_layer(self, bait_img, size):
        # Resize the bait image to the final output size first to ensure consistent watermark scaling.
        bait_img = bait_img.resize(size, Image.LANCZOS)

        # Apply watermarks targeted for "Bait"
        for wm in self.watermarks:
            if wm['target'].get() == "Bait":
                settings = {
                    'type': wm['type'].get(),
                    'text': wm['text'].get(),
                    'font_family': wm['font_family'].get(),
                    'image_path': wm['image_path'].get(),
                    'size': wm['size'].get(),
                    'opacity': wm['opacity'].get(),
                    'position': wm['position'].get(),
                    'text_color': wm['text_color'].get(),
                    'auto_contrast': wm['auto_contrast'].get(),
                    'use_outline': wm['use_outline'].get(),
                    'outline_color': wm['outline_color'].get(),
                    'use_specific_pos': wm['use_specific_pos'].get(),
                    'pos_x': wm['pos_x'].get(),
                    'pos_y': wm['pos_y'].get()
                }
                # Use the auto_contrast setting from the UI
                bait_img = apply_watermark(bait_img, settings, self.bait_clear_color, apply_contrasting_color_logic=settings['auto_contrast'])

        # If auto_bait is enabled, create a pencil sketch from the (now resized and watermarked) bait image.
        if self.use_auto_bait.get():
            self.status.config(text="Creating bait sketch...", fg="#ffff00")
            self.root.update_idletasks()
            
            bait_rgb_np = np.array(bait_img.convert("RGB"))
            bait_bgr_np = cv2.cvtColor(bait_rgb_np, cv2.COLOR_RGB2BGR)

            tip_size = self.pencil_tip_size.get()
            range_param = self.pencil_range.get()
            
            sketch_np, msg = create_pencil_sketch(bait_bgr_np, tip_size, range_param)
            if sketch_np is None: raise ValueError(f"Sketch creation failed: {msg}")

            bait_target_rgb = self.hex_to_rgb(self.bait_clear_color)
            luminance = (0.299 * bait_target_rgb[0] + 0.587 * bait_target_rgb[1] + 0.114 * bait_target_rgb[2])
            is_dark_bg = luminance < 128
            
            alpha_data = 255 - sketch_np
            
            # Use black lines for light backgrounds and white lines for dark backgrounds.
            line_color_rgb = (0, 0, 0) if not is_dark_bg else (255, 255, 255)
            
            h, w = sketch_np.shape
            r_channel = np.full((h, w), line_color_rgb[0], dtype=np.uint8)
            g_channel = np.full((h, w), line_color_rgb[1], dtype=np.uint8)
            b_channel = np.full((h, w), line_color_rgb[2], dtype=np.uint8)
            
            rgba_arr = np.stack([r_channel, g_channel, b_channel, alpha_data], axis=-1)
            bait_img = Image.fromarray(rgba_arr, 'RGBA')

        # Process the final bait layer (which could be a sketch or the resized/watermarked image).
        self.status.config(text="Processing Bait Layer...", fg="#ffff00")
        self.root.update_idletasks()
        bait_target_rgb = self.hex_to_rgb(self.bait_clear_color)
        bait_processed = apply_color_clearer(bait_img, target_color=bait_target_rgb)

        # Apply final opacity to the processed bait layer.
        bait_op_val = self.bait_opacity.get()
        if bait_op_val < 255:
            alpha = bait_processed.getchannel('A')
            new_alpha_data = (np.array(alpha).astype(np.float32) * (bait_op_val / 255.0)).astype(np.uint8)
            bait_processed.putalpha(Image.fromarray(new_alpha_data))
        
        return bait_processed

    def apply_slip_mode(self, arr):
        """Applies a small, unnoticeable change to pixel data to ensure a unique file hash."""
        mode = self.slip_mode_option.get()

        if mode == "None":
            return arr

        temp = arr.astype(np.int16)
        mask = temp[:, :, 3] > 0
        ys, xs = np.where(mask)

        if len(ys) == 0:
            return arr

        num_visible_pixels = len(ys)
        delta_choices = [-3, -2, -1, 1, 2, 3]

        if mode == "Single Random Pixel":
            idx = np.random.randint(0, num_visible_pixels)
            y, x = ys[idx], xs[idx]
            c = random.randint(0, 2)
            delta = random.choice(delta_choices)
            temp[y, x, c] += delta

        elif mode == "Random Amount of Pixels":
            num_to_change = random.randint(1, num_visible_pixels)
            indices_to_change = np.random.choice(num_visible_pixels, num_to_change, replace=False)
            for i in indices_to_change:
                y, x = ys[i], xs[i]
                c = random.randint(0, 2)
                delta = random.choice(delta_choices)
                temp[y, x, c] += delta

        elif mode == "All Non-Transparent Pixels":
            for i in range(num_visible_pixels):
                y, x = ys[i], xs[i]
                c = random.randint(0, 2)
                delta = random.choice(delta_choices)
                temp[y, x, c] += delta

        return np.clip(temp, 0, 255).astype(np.uint8)

    def save_as(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path: self.output_path = path
        self.status.config(text=f"Output: {os.path.basename(path)}", fg="#00ff00")

    def save_bait_image(self):
        """Processes and saves only the bait layer."""
        if not self.bait_path:
            messagebox.showerror("Error", "Select a bait image first!")
            return

        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return

        try:
            self.status.config(text="Processing Bait Layer for saving...", fg="#ffff00")
            self.root.update_idletasks()

            if self.use_real_image_size.get() and self.real_path:
                with Image.open(self.real_path) as r_img:
                     width, height = r_img.size
            else:
                width = int(self.output_width.get())
                height = int(self.output_height.get())
            size = (width, height)

            bait_img = Image.open(self.bait_path).convert("RGBA")
            
            bait_processed = self._process_bait_layer(bait_img, size)

            bait_processed.save(path, "PNG")
            self.status.config(text=f"SAVED BAIT: {os.path.basename(path)}", fg="#00ff00")
            messagebox.showinfo("Success", f"Bait image saved!\n{path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save bait image: {e}")
            self.status.config(text=f"Error saving bait: {e}", fg="#ff5555")

    def build_and_save(self):
        if not self.real_path or not self.bait_path:
            messagebox.showerror("Error", "Select both images!")
            return
        if not self.output_path:
            self.save_as()
            if not self.output_path: return
        try:
            result = self.build_stego(apply_noise=True)
            if result is None: return
            result.save(self.output_path, "PNG")
            self.status.config(text=f"SAVED: {os.path.basename(self.output_path)}", fg="#00ff00")
            messagebox.showinfo("Success", f"Saved!\n{self.output_path}")
        except Exception as e: messagebox.showerror("Error", str(e))

    def export_batch(self):
        if not self.real_path or not self.bait_path:
            messagebox.showerror("Error", "Select both a Bait and Real image first.")
            return
        try:
            num_copies = int(self.batch_copies.get())
            if num_copies <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Number of copies must be a positive integer.")
            return
        base_name = self.batch_basename.get().strip()
        if not base_name or any(c in base_name for c in r'/\:*?"<>|'):
            messagebox.showerror("Error", "Please enter a valid base filename (no special characters).")
            return
        
        if self.slip_mode_option.get() == "None":
            if not messagebox.askyesno("Warning", "Slip Mode is set to 'None'.\nAll exported files will be identical. Do you want to continue?"):
                return
        try:
            downloads_path = Path.home() / "Downloads"
            output_dir_path = downloads_path / "decal_bypass_output"
            output_dir_path.mkdir(parents=True, exist_ok=True)
            self.status.config(text="Building base image for batch...", fg="#ffff00")
            self.root.update_idletasks()
            clean_base_image = self.build_stego(apply_noise=False)
            base_arr = np.array(clean_base_image)
            for i in range(1, num_copies + 1):
                self.status.config(text=f"Processing copy {i}/{num_copies}...", fg="#ffff00")
                self.root.update_idletasks()
                noisy_arr = self.apply_slip_mode(base_arr.copy())
                noisy_image = Image.fromarray(noisy_arr)
                filename = f"{base_name}_{i:03d}.png"
                save_path = output_dir_path / filename
                noisy_image.save(save_path, "PNG")
            self.status.config(text=f"Batch export complete! Saved to Downloads folder.", fg="#00ff00")
            messagebox.showinfo("Success", f"{num_copies} images were saved to:\n{output_dir_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred during batch export: {e}")
            self.status.config(text=f"Batch export failed: {e}", fg="#ff5555")

    # --- Method Save/Load ---
    def read_methods_from_file(self):
        """Reads saved methods from methods.json, returns empty dict if error."""
        try:
            if os.path.exists("methods.json"):
                with open("methods.json", "r") as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            messagebox.showerror("Error Reading Methods", f"Could not read methods.json.\n{e}")
        return {}

    def write_methods_to_file(self, methods_data):
        """Writes the given dictionary to methods.json."""
        try:
            with open("methods.json", "w") as f:
                json.dump(methods_data, f, indent=4)
            return True
        except IOError as e:
            messagebox.showerror("Error Saving Method", f"Could not write to methods.json.\n{e}")
            return False

    def save_method(self):
        """Saves the current custom layer setup as a named method."""
        if not self.custom_layers:
            messagebox.showwarning("Cannot Save", "There are no layers to save.")
            return

        method_name = simpledialog.askstring("Save Method", "Enter a name for this method:")
        if not method_name:
            return

        def serialize_layer_data(layer_data):
            """Helper to convert a layer's data (with tk.Vars) to a serializable dict."""
            layer_type = layer_data.get('type', 'process')
            config = {'type': layer_type}

            if layer_type == 'process':
                config.update({
                    'name': layer_data['name'].get(),
                    'opacity': layer_data['opacity'].get(),
                    'is_bw': layer_data['is_bw'].get(),
                    'is_inverted': layer_data['is_inverted'].get(),
                    'invert_clear': layer_data['invert_clear'].get(),
                    'use_color_blend': layer_data['use_color_blend'].get(),
                    'blend_color': layer_data['blend_color'].get(),
                    'blend_opacity': layer_data['blend_opacity'].get()
                })
            elif layer_type == 'merged':
                config.update({
                    'name': layer_data['name'].get(),
                    'opacity': layer_data['opacity'].get(),
                    # sub_layers are already serializable dicts, not full layer objects
                    'sub_layers': layer_data['sub_layers'] 
                })
            return config

        layer_config = [serialize_layer_data(layer) for layer in self.custom_layers]

        methods = self.read_methods_from_file()
        if method_name in methods:
            if not messagebox.askyesno("Overwrite?", f"A method named '{method_name}' already exists. Overwrite it?"):
                return
        
        methods[method_name] = layer_config
        if self.write_methods_to_file(methods):
            messagebox.showinfo("Success", f"Method '{method_name}' saved successfully.")

    def import_methods(self):
        """Imports methods from a JSON file and merges them with the current methods."""
        path = filedialog.askopenfilename(
            title="Select a methods JSON file to import",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return

        try:
            with open(path, "r") as f:
                new_methods = json.load(f)
            if not isinstance(new_methods, dict):
                raise ValueError("Imported file is not a valid method dictionary.")
        except (IOError, json.JSONDecodeError, ValueError) as e:
            messagebox.showerror("Import Error", f"Could not read or parse the selected file.\n{e}")
            return

        current_methods = self.read_methods_from_file()
        methods_to_add = {}
        skipped_count = 0
        overwritten_count = 0

        for name, config in new_methods.items():
            if name in current_methods:
                if messagebox.askyesno("Conflict", f"A method named '{name}' already exists. Overwrite it?"):
                    methods_to_add[name] = config
                    overwritten_count += 1
                else:
                    skipped_count += 1
            else:
                methods_to_add[name] = config
        
        if not methods_to_add:
            messagebox.showinfo("Import Complete", "No new methods were imported.")
            return

        current_methods.update(methods_to_add)
        if self.write_methods_to_file(current_methods):
            summary = (f"Import successful!\n\n"
                       f"New methods added: {len(methods_to_add) - overwritten_count}\n"
                       f"Methods overwritten: {overwritten_count}\n"
                       f"Methods skipped: {skipped_count}")
            messagebox.showinfo("Import Complete", summary)
        else:
            self.status.config(text="Failed to write imported methods.", fg="#ff5555")

    def rebuild_layers_ui(self):
        """Clears and rebuilds the entire custom layer UI from the self.custom_layers list."""
        # Store a copy of the layers data
        layers_to_rebuild = list(self.custom_layers)
        
        # Clear the UI and the main list
        for child in self.layers_container.winfo_children():
            child.destroy()
        self.custom_layers.clear()

        # Re-add each layer, which rebuilds its UI frame and appends it back to self.custom_layers
        for layer_data in layers_to_rebuild:
            self.add_layer(existing_data=layer_data)

    def load_method(self):
        """Shows a dialog to load a saved method."""
        methods = self.read_methods_from_file()
        if not methods:
            messagebox.showinfo("No Saved Methods", "There are no methods saved in methods.json.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Load Method")
        dialog.geometry("300x250")
        dialog.configure(bg="#1a1a1a")
        
        tk.Label(dialog, text="Select a method to load:", bg="#1a1a1a", fg="#00ccff").pack(pady=10)
        
        listbox = tk.Listbox(dialog, bg="#333", fg="#fff", selectbackground="#00aaff")
        listbox.pack(fill="both", expand=True, padx=10, pady=5)
        for name in sorted(methods.keys()):
            listbox.insert(tk.END, name)

        def on_load():
            selected_indices = listbox.curselection()
            if not selected_indices: return
            
            selected_name = listbox.get(selected_indices[0])
            
            self.clear_layers()
            
            def deserialize_layer_config(configs):
                """Helper to convert a list of saved configs into full layer objects with tk.Vars."""
                layer_objects = []
                for i, config in enumerate(configs):
                    layer_type = config.get('type', 'process')
                    
                    if layer_type == 'process':
                        layer_data = {
                            'type': 'process',
                            'name': tk.StringVar(value=config.get('name', f"Layer {i + 1}")),
                            'opacity': tk.IntVar(value=config.get('opacity', 255)),
                            'is_bw': tk.BooleanVar(value=config.get('is_bw', False)),
                            'is_inverted': tk.BooleanVar(value=config.get('is_inverted', False)),
                            'invert_clear': tk.BooleanVar(value=config.get('invert_clear', False)),
                            'use_color_blend': tk.BooleanVar(value=config.get('use_color_blend', False)),
                            'blend_color': tk.StringVar(value=config.get('blend_color', '#FFFFFF')),
                            'blend_opacity': tk.DoubleVar(value=config.get('blend_opacity', 0.5))
                        }
                    elif layer_type == 'merged':
                        layer_data = {
                            'type': 'merged',
                            'name': tk.StringVar(value=config.get('name', 'Merged Layer')),
                            'opacity': tk.IntVar(value=config.get('opacity', 255)),
                            'sub_layers': config.get('sub_layers', [])
                        }
                    layer_objects.append(layer_data)
                return layer_objects

            # Create full layer objects from the loaded config
            loaded_layers = deserialize_layer_config(methods[selected_name])
            
            # Now build the UI from the data
            for data in loaded_layers:
                self.add_layer(existing_data=data)

            self.update_preview()
            dialog.destroy()

        tk.Button(dialog, text="Load Selected", command=on_load, bg="#ffaa00", fg="black").pack(pady=10)
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)

    def clear_layers(self):
        """Removes all custom layers from the UI and data list."""
        for layer in self.custom_layers:
            layer['frame'].destroy()
        self.custom_layers.clear()

    def toggle_custom_layer_controls(self):
        if self.use_custom_layer_method.get():
            self.custom_layers_frame.grid()
            self.real_opacity_slider.config(state='disabled')
        else:
            self.custom_layers_frame.grid_remove()
            self.real_opacity_slider.config(state='normal')
        self.update_preview()

    def choose_layer_blend_color(self, color_var):
        """Opens a color chooser for a layer's blend color."""
        initial_color = color_var.get()
        color_code = colorchooser.askcolor(title="Choose Blend Color", initialcolor=initial_color)
        if color_code and color_code[1]:
            color_var.set(color_code[1])
            self.update_preview()

    def add_layer(self, existing_data=None):
        """Adds a new layer to the UI, either new or from existing data."""
        layer_index = len(self.custom_layers)
        
        frame = tk.Frame(self.layers_container, bg="#2a2a2a", relief="groove", borderwidth=1)
        frame.pack(fill='x', padx=5, pady=5)

        if existing_data:
            layer_data = existing_data
            layer_data['frame'] = frame
        else: # Create a new default 'process' layer
            layer_data = {
                'type': 'process',
                'frame': frame,
                'name': tk.StringVar(value=f"Layer {layer_index + 1}"),
                'opacity': tk.IntVar(value=255),
                'is_bw': tk.BooleanVar(value=False),
                'is_inverted': tk.BooleanVar(value=False),
                'invert_clear': tk.BooleanVar(value=False),
                'use_color_blend': tk.BooleanVar(value=False),
                'blend_color': tk.StringVar(value='#FFFFFF'),
                'blend_opacity': tk.DoubleVar(value=0.5)
            }
        
        # --- Top Row: Name, Type, and Action Buttons ---
        top_row = tk.Frame(frame, bg="#2a2a2a")
        top_row.pack(fill='x')
        
        label = tk.Label(top_row, text=f"Layer {layer_index + 1}", fg="#00ccff", bg="#2a2a2a")
        label.pack(side='left', padx=5, pady=2)
        layer_data['label'] = label
        
        name_entry = tk.Entry(top_row, textvariable=layer_data['name'], bg="#333", fg="#fff", insertbackground="#fff", relief="flat", width=15)
        name_entry.pack(side='left', padx=5, pady=2)

        remove_button = tk.Button(top_row, text="Remove", command=lambda i=layer_index: self.remove_layer(i), bg="#ff5555", fg="white", relief="flat")
        remove_button.pack(side='right', padx=5, pady=2)
        
        dup_button = tk.Button(top_row, text="Dupe", command=lambda i=layer_index: self.duplicate_layer(i), bg="#8888ff", fg="white", relief="flat")
        dup_button.pack(side='right', padx=5, pady=2)

        down_button = tk.Button(top_row, text="▼", command=lambda i=layer_index: self.move_layer(i, 1), bg="#444", fg="white", relief="flat")
        down_button.pack(side='right', padx=(0,5), pady=2)
        up_button = tk.Button(top_row, text="▲", command=lambda i=layer_index: self.move_layer(i, -1), bg="#444", fg="white", relief="flat")
        up_button.pack(side='right', padx=(5,0), pady=2)
        
        layer_data.update({'up_button': up_button, 'down_button': down_button})

        # --- Build UI based on layer type ---
        if layer_data.get('type', 'process') == 'process':
            self._add_process_layer_controls(frame, layer_data, layer_index)
        elif layer_data.get('type') == 'merged':
            self._add_merged_layer_controls(frame, layer_data, layer_index)

        if not existing_data:
            self.custom_layers.append(layer_data)
        else:
            self.custom_layers.insert(layer_index, layer_data)

        self.update_layer_controls()
        self.update_preview()

    def _add_process_layer_controls(self, parent_frame, layer_data, layer_index):
        # --- Opacity Row ---
        opacity_frame = tk.Frame(parent_frame, bg="#2a2a2a")
        opacity_frame.pack(fill='x', expand=True, padx=5)
        tk.Label(opacity_frame, text="Opacity:", fg="#ccc", bg="#2a2a2a").pack(side='left')
        EditableLabel(opacity_frame, layer_data['opacity'], bg="#2a2a2a", fg="#fff", width=4).pack(side='right', padx=(5,0))
        tk.Scale(opacity_frame, from_=0, to=255, orient="horizontal", variable=layer_data['opacity'], bg="#2a2a2a", fg="#00ccff", troughcolor="#444", command=self.update_preview).pack(side='right', fill='x', expand=True)

        # --- Process Controls Row ---
        controls_row = tk.Frame(parent_frame, bg="#2a2a2a")
        controls_row.pack(fill='x', padx=5)
        ttk.Checkbutton(controls_row, text="B&W", variable=layer_data['is_bw'], command=self.update_preview).pack(side='left', padx=5)
        ttk.Checkbutton(controls_row, text="Invert", variable=layer_data['is_inverted'], command=self.update_preview).pack(side='left', padx=5)
        ttk.Checkbutton(controls_row, text="Invert Clear", variable=layer_data['invert_clear'], command=self.update_preview).pack(side='left', padx=5)
        
        merge_button = tk.Button(controls_row, text="Merge Down", command=lambda i=layer_index: self.merge_down(i), bg="#f0ad4e", fg="black")
        merge_button.pack(side='right', padx=5)
        layer_data['merge_button'] = merge_button

        # --- Color Blend Row ---
        blend_frame = tk.Frame(parent_frame, bg="#3a3a3a")
        blend_frame.pack(fill='x', expand=True, padx=5, pady=(0, 5))
        ttk.Checkbutton(blend_frame, text="Color Blend", variable=layer_data['use_color_blend'], command=self.update_preview).pack(side='left', padx=5)
        blend_color_button = tk.Button(blend_frame, text="Choose...", command=lambda c=layer_data['blend_color']: self.choose_layer_blend_color(c))
        blend_color_button.pack(side='left', padx=5)
        tk.Label(blend_frame, text="Blend:", fg="#ccc", bg="#3a3a3a").pack(side='left', padx=(10, 0))
        tk.Scale(blend_frame, from_=0.0, to=1.0, orient="horizontal", variable=layer_data['blend_opacity'], resolution=0.01, bg="#3a3a3a", fg="#00ccff", troughcolor="#555", command=self.update_preview).pack(side='left', fill='x', expand=True, padx=5)

    def _add_merged_layer_controls(self, parent_frame, layer_data, layer_index):
        # --- Master Opacity Row ---
        opacity_frame = tk.Frame(parent_frame, bg="#2a2a2a")
        opacity_frame.pack(fill='x', expand=True, padx=5)
        tk.Label(opacity_frame, text="Group Opacity:", fg="#ccc", bg="#2a2a2a").pack(side='left')
        EditableLabel(opacity_frame, layer_data['opacity'], bg="#2a2a2a", fg="#fff", width=4).pack(side='right', padx=(5,0))
        tk.Scale(opacity_frame, from_=0, to=255, orient="horizontal", variable=layer_data['opacity'], bg="#2a2a2a", fg="#00ccff", troughcolor="#444", command=self.update_preview).pack(side='right', fill='x', expand=True)

        # --- Info and Actions Row ---
        info_frame = tk.Frame(parent_frame, bg="#2a2a2a")
        info_frame.pack(fill='x', padx=5, pady=5)
        sub_layer_count = len(layer_data.get('sub_layers', []))
        tk.Label(info_frame, text=f"Group of {sub_layer_count} layers", fg="#aaa", bg="#2a2a2a").pack(side='left', padx=5)
        
        ungroup_button = tk.Button(info_frame, text="Ungroup", command=lambda i=layer_index: self.ungroup_layer(i), bg="#5bc0de", fg="black")
        ungroup_button.pack(side='right', padx=5)
        
        merge_button = tk.Button(info_frame, text="Merge Down", command=lambda i=layer_index: self.merge_down(i), bg="#f0ad4e", fg="black")
        merge_button.pack(side='right', padx=5)
        layer_data['merge_button'] = merge_button

    def merge_down(self, index):
        if index >= len(self.custom_layers) - 1: return

        top_layer_obj = self.custom_layers[index]
        bottom_layer_obj = self.custom_layers[index + 1]

        def get_settings_list(layer_obj):
            """Extracts a list of serializable setting dicts from a layer object."""
            if layer_obj.get('type', 'process') == 'process':
                return [{
                    'name': layer_obj['name'].get(), 'opacity': layer_obj['opacity'].get(),
                    'is_bw': layer_obj['is_bw'].get(), 'is_inverted': layer_obj['is_inverted'].get(),
                    'invert_clear': layer_obj['invert_clear'].get(), 'use_color_blend': layer_obj['use_color_blend'].get(),
                    'blend_color': layer_obj['blend_color'].get(), 'blend_opacity': layer_obj['blend_opacity'].get()
                }]
            elif layer_obj.get('type') == 'merged':
                return layer_obj.get('sub_layers', [])
            return []

        new_sub_layers = get_settings_list(top_layer_obj) + get_settings_list(bottom_layer_obj)
        
        new_merged_layer = {
            'type': 'merged',
            'name': tk.StringVar(value=f"Group"),
            'opacity': tk.IntVar(value=255),
            'sub_layers': new_sub_layers
        }

        # Replace the two old layers with the new merged one
        self.custom_layers.pop(index)
        self.custom_layers.pop(index) # Pop again, since list shifted
        self.custom_layers.insert(index, new_merged_layer)

        self.rebuild_layers_ui()
        self.status.config(text="Layers merged into a group.", fg="#00ff99")

    def ungroup_layer(self, index):
        if self.custom_layers[index].get('type') != 'merged': return

        merged_layer = self.custom_layers.pop(index)
        sub_layers_config = merged_layer.get('sub_layers', [])

        # Convert the setting dicts back into full layer objects
        for i, config in enumerate(reversed(sub_layers_config)):
            layer_obj = {
                'type': 'process',
                'name': tk.StringVar(value=config.get('name', f"Layer {i + 1}")),
                'opacity': tk.IntVar(value=config.get('opacity', 128)),
                'is_bw': tk.BooleanVar(value=config.get('is_bw', False)),
                'is_inverted': tk.BooleanVar(value=config.get('is_inverted', False)),
                'invert_clear': tk.BooleanVar(value=config.get('invert_clear', False)),
                'use_color_blend': tk.BooleanVar(value=config.get('use_color_blend', False)),
                'blend_color': tk.StringVar(value=config.get('blend_color', '#FFFFFF')),
                'blend_opacity': tk.DoubleVar(value=config.get('blend_opacity', 0.5))
            }
            self.custom_layers.insert(index, layer_obj)
        
        self.rebuild_layers_ui()
        self.status.config(text="Layer group has been ungrouped.", fg="#00ff99")

    def duplicate_layer(self, index):
        if 0 <= index < len(self.custom_layers):
            source = self.custom_layers[index]
            layer_type = source.get('type', 'process')
            
            if layer_type == 'process':
                new_layer_data = {
                    'type': 'process',
                    'name': tk.StringVar(value=source['name'].get() + " Copy"),
                    'opacity': tk.IntVar(value=source['opacity'].get()),
                    'is_bw': tk.BooleanVar(value=source['is_bw'].get()),
                    'is_inverted': tk.BooleanVar(value=source['is_inverted'].get()),
                    'invert_clear': tk.BooleanVar(value=source['invert_clear'].get()),
                    'use_color_blend': tk.BooleanVar(value=source['use_color_blend'].get()),
                    'blend_color': tk.StringVar(value=source['blend_color'].get()),
                    'blend_opacity': tk.DoubleVar(value=source['blend_opacity'].get())
                }
            elif layer_type == 'merged':
                # Deep copy sub_layers (list of dicts)
                new_sub_layers = [d.copy() for d in source.get('sub_layers', [])]
                
                new_layer_data = {
                    'type': 'merged',
                    'name': tk.StringVar(value=source['name'].get() + " Copy"),
                    'opacity': tk.IntVar(value=source['opacity'].get()),
                    'sub_layers': new_sub_layers
                }
            
            self.custom_layers.insert(index + 1, new_layer_data)
            self.rebuild_layers_ui()
            self.status.config(text="Layer duplicated.", fg="#00ff99")

    def remove_layer(self, index_to_remove):
        if 0 <= index_to_remove < len(self.custom_layers):
            self.custom_layers.pop(index_to_remove)
            self.rebuild_layers_ui()

    def move_layer(self, index_to_move, direction):
        """Moves a layer up or down in the list."""
        new_index = index_to_move + direction
        if 0 <= new_index < len(self.custom_layers):
            # Swap layers in the data list
            self.custom_layers[index_to_move], self.custom_layers[new_index] = \
                self.custom_layers[new_index], self.custom_layers[index_to_move]
            self.rebuild_layers_ui()

    def update_layer_controls(self):
        """Updates layer labels and button states (like enabling/disabling move/merge buttons)."""
        num_layers = len(self.custom_layers)
        for i, layer_data in enumerate(self.custom_layers):
            # Update the main label for all layers
            layer_data['label'].config(text=f"Layer {i + 1}")
            
            # Update move button states
            layer_data['up_button'].config(state='disabled' if i == 0 else 'normal')
            layer_data['down_button'].config(state='disabled' if i == num_layers - 1 else 'normal')

            # Update merge button state (if it exists)
            if 'merge_button' in layer_data:
                layer_data['merge_button'].config(state='disabled' if i >= num_layers - 1 else 'normal')
    def toggle_auto_bait_controls(self):
        if self.use_auto_bait.get():
            self.auto_bait_frame.grid()
        else:
            self.auto_bait_frame.grid_remove()
        self.update_preview()

    # --- MODIFICATION: New methods for watermark controls ---
    def toggle_watermark_controls(self, *args):
        """Shows/hides watermark controls based on type."""
        if self.watermark_type.get() == "Text":
            self.watermark_text_frame.grid()
            self.watermark_image_frame.grid_remove()
        else: # Image
            self.watermark_text_frame.grid_remove()
            self.watermark_image_frame.grid()
        self.update_preview()

    def load_watermark_image(self):
        """Loads a watermark image from file."""
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if path:
            self.watermark_image_path.set(path)
            self.watermark_image_label.config(text=os.path.basename(path))
            self.update_preview()

    def _on_mousewheel(self, event):
        # On Windows, the delta is usually a multiple of 120
        # Scroll by a fixed number of units (e.g., 3 lines)
        self.canvas.yview_scroll(int(-1*(event.delta/abs(event.delta))*3), "units")

if __name__ == "__main__":
    root = tk.Tk()
    app = StegoDecalBuilder(root)
    root.mainloop()
