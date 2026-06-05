"""Build the 22-slide SceneSense Agent intro deck.

Run: python3 build_scenesense_intro_deck.py
Output: SceneSense_Agent_Intro_Deck.pptx in the same folder.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree


# -------------------- Palette --------------------
NAVY = RGBColor(0x0B, 0x3D, 0x91)
ID_BLUE = RGBColor(0x00, 0xA9, 0xE0)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
MED_GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xEF, 0xEF, 0xEF)
SOFT_BG = RGBColor(0xF7, 0xF9, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE = RGBColor(0xEA, 0x8A, 0x1F)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
RED = RGBColor(0xC0, 0x39, 0x2B)
PURPLE = RGBColor(0x6A, 0x4C, 0x93)

# -------------------- Presentation setup --------------------
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height


# -------------------- Helpers --------------------
def add_blank_slide():
    blank_layout = prs.slide_layouts[6]
    return prs.slides.add_slide(blank_layout)


def add_text_box(slide, left, top, width, height,
                 text, font_size=14, bold=False, color=DARK_GRAY,
                 align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False,
                 font_name="Calibri"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(36000)
    tf.margin_right = Emu(36000)
    tf.margin_top = Emu(18000)
    tf.margin_bottom = Emu(18000)
    tf.vertical_anchor = anchor
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.name = font_name
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
    return tb


def add_rich_text_box(slide, left, top, width, height, lines,
                       anchor=MSO_ANCHOR.TOP):
    """lines: list of dicts with keys: text, size, bold, color, italic, align."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(36000)
    tf.margin_right = Emu(36000)
    tf.margin_top = Emu(18000)
    tf.margin_bottom = Emu(18000)
    tf.vertical_anchor = anchor
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ln.get("align", PP_ALIGN.LEFT)
        run = p.add_run()
        run.text = ln.get("text", "")
        run.font.name = ln.get("font", "Calibri")
        run.font.size = Pt(ln.get("size", 14))
        run.font.bold = ln.get("bold", False)
        run.font.italic = ln.get("italic", False)
        run.font.color.rgb = ln.get("color", DARK_GRAY)
        if "space_after" in ln:
            p.space_after = Pt(ln["space_after"])
    return tb


def add_rect(slide, left, top, width, height,
             fill=WHITE, line=NAVY, line_width=1.25,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, shadow=False):
    s = slide.shapes.add_shape(shape, left, top, width, height)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_width)
    if not shadow:
        # Try to remove default shadow
        sppr = s.shadow._element  # not strictly needed; default shadow is light
    return s


def set_shape_text(shape, text, size=12, bold=True, color=DARK_GRAY,
                    align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(36000)
    tf.margin_right = Emu(36000)
    tf.margin_top = Emu(18000)
    tf.margin_bottom = Emu(18000)
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color


def add_label(slide, left, top, width, height, text,
              size=12, bold=True, color=DARK_GRAY, italic=False,
              align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(18000)
    tf.margin_right = Emu(18000)
    tf.margin_top = Emu(9000)
    tf.margin_bottom = Emu(9000)
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
    return tb


def add_arrow(slide, x1, y1, x2, y2, color=NAVY, line_width=2.0,
              head_style="triangle"):
    """Draw a connector from (x1,y1) to (x2,y2)."""
    connector = slide.shapes.add_connector(1, x1, y1, x2, y2)  # 1 = straight
    connector.line.color.rgb = color
    connector.line.width = Pt(line_width)
    # Add arrow end
    ln = connector.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return connector


def add_title_bar(slide, title, subtitle=None):
    # Top accent strip
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.08))
    strip.fill.solid()
    strip.fill.fore_color.rgb = ID_BLUE
    strip.line.fill.background()
    # Title text
    title_h = Inches(0.55) if subtitle else Inches(0.7)
    add_text_box(slide, Inches(0.4), Inches(0.18), Inches(12.5), title_h,
                 title, font_size=26, bold=True, color=NAVY)
    if subtitle:
        add_text_box(slide, Inches(0.4), Inches(0.75), Inches(12.5),
                     Inches(0.45), subtitle, font_size=14, italic=True,
                     color=MED_GRAY)
    # Bottom footer line
    footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.32),
                                    SLIDE_W, Inches(0.03))
    footer.fill.solid()
    footer.fill.fore_color.rgb = ID_BLUE
    footer.line.fill.background()
    add_text_box(slide, Inches(0.4), Inches(7.18), Inches(12.5), Inches(0.25),
                 "©2026 InterDigital, Inc. • SceneSense Agent intern presentation",
                 font_size=9, italic=True, color=MED_GRAY)


def add_bullets(slide, left, top, width, height, items,
                size=14, color=DARK_GRAY, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Emu(36000)
    tf.margin_right = Emu(36000)
    tf.margin_top = Emu(18000)
    tf.margin_bottom = Emu(18000)
    tf.vertical_anchor = anchor
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(6)
        run = p.add_run()
        # use a hyphen bullet for crisp rendering
        if isinstance(item, tuple):
            head, body = item
            r1 = p.add_run() if i > 0 or False else run
            # Reset existing run to bullet head
            run.text = "•  "
            run.font.bold = False
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.name = "Calibri"
            rh = p.add_run()
            rh.text = head + " "
            rh.font.bold = True
            rh.font.size = Pt(size)
            rh.font.color.rgb = NAVY
            rh.font.name = "Calibri"
            rb = p.add_run()
            rb.text = body
            rb.font.size = Pt(size)
            rb.font.color.rgb = color
            rb.font.name = "Calibri"
        else:
            run.text = "•  " + item
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.name = "Calibri"
    return tb


# ============================================================
# SLIDE 1 -- Title
# ============================================================
def slide_title():
    s = add_blank_slide()
    # Background
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    # Accent diagonal
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(6.1),
                                SLIDE_W, Inches(0.18))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ID_BLUE
    accent.line.fill.background()

    add_text_box(s, Inches(0.7), Inches(1.6), Inches(12), Inches(1.1),
                 "SceneSense Agent",
                 font_size=54, bold=True, color=WHITE)
    add_text_box(s, Inches(0.7), Inches(2.55), Inches(12), Inches(0.7),
                 "Agent-Controlled Split Inference for Network-aware",
                 font_size=24, color=WHITE)
    add_text_box(s, Inches(0.7), Inches(2.95), Inches(12), Inches(0.7),
                 "Cooperative Perception over Shared Spatial Maps",
                 font_size=24, color=WHITE)

    add_text_box(s, Inches(0.7), Inches(4.4), Inches(12), Inches(0.5),
                 "Abiodun Ganiyu  ·  IDCC × NEU 6-Month Internship",
                 font_size=20, color=ID_BLUE, bold=True)
    add_text_box(s, Inches(0.7), Inches(4.95), Inches(12), Inches(0.5),
                 "Under Subhramoy Mohanti  ·  June 2026",
                 font_size=16, color=WHITE)

    add_text_box(s, Inches(0.7), Inches(6.5), Inches(12), Inches(0.4),
                 "©2026 InterDigital, Inc. All Rights Reserved.",
                 font_size=11, italic=True, color=WHITE)
    return s


# ============================================================
# SLIDE 2 -- Where this fits
# ============================================================
def slide_position():
    s = add_blank_slide()
    add_title_bar(s, "Where this work fits",
                   "Intern-owned research thread inside the SceneSense / SCAN-AI program")

    # Three pillars
    pillars = [
        ("Strategic context",
         [
             "InterDigital research on AI traffic over 5G/6G",
             "SceneSense: network-aware split inference + cooperative perception",
             "Standards anchors: 3GPP 5QI, OAI, V2X, NWDAF/RIC",
         ], NAVY),
        ("My focus (Months 1–6)",
         [
             "Extend single-UE split inference to multi-UE cooperative perception",
             "Replace static compression knobs with a constrained RL agent",
             "Use task-precision guardrails so safety classes survive compression",
             "Close the loop with a shared spatial map for occluded-object alerts",
         ], ID_BLUE),
        ("Out of scope (for now)",
         [
             "Designing new perception or fusion models",
             "Full PC5 / Uu standards integration",
             "Large-scale RSU sensor fusion",
             "Production-grade autonomy stack",
         ], MED_GRAY),
    ]

    x = Inches(0.4)
    col_w = Inches(4.18)
    gap = Inches(0.13)
    top = Inches(1.45)
    h = Inches(5.4)
    for i, (head, items, color) in enumerate(pillars):
        left = x + (col_w + gap) * i
        card = add_rect(s, left, top, col_w, h, fill=WHITE,
                         line=color, line_width=1.5)
        # Header band
        band = add_rect(s, left, top, col_w, Inches(0.55),
                        fill=color, line=color, shape=MSO_SHAPE.RECTANGLE)
        set_shape_text(band, head, size=16, bold=True, color=WHITE)
        # Items
        add_bullets(s, left, top + Inches(0.7), col_w, h - Inches(0.8),
                    items, size=14)
    return s


# ============================================================
# SLIDE 3 -- The problem: occlusion
# ============================================================
def slide_problem_occlusion():
    s = add_blank_slide()
    add_title_bar(s, "The problem: a single car cannot see everything",
                   "Perception is bounded by occlusion and viewpoint — not by compute")

    # LEFT: sketch
    sx = Inches(0.5)
    sy = Inches(1.6)
    sw = Inches(6.3)
    sh = Inches(5.2)
    canvas = add_rect(s, sx, sy, sw, sh, fill=SOFT_BG, line=LIGHT_GRAY)

    # road
    road = add_rect(s, sx + Inches(0.4), sy + Inches(2.9), sw - Inches(0.8),
                    Inches(1.0), fill=RGBColor(0x55, 0x55, 0x55),
                    line=None, shape=MSO_SHAPE.RECTANGLE)
    # lane line
    for k in range(7):
        seg = add_rect(s, sx + Inches(0.6 + k * 0.8), sy + Inches(3.35),
                       Inches(0.4), Inches(0.06), fill=WHITE, line=None,
                       shape=MSO_SHAPE.RECTANGLE)

    # Ego car (left)
    ego = add_rect(s, sx + Inches(0.6), sy + Inches(3.0), Inches(1.0),
                    Inches(0.55), fill=ID_BLUE, line=NAVY,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    set_shape_text(ego, "EGO", size=11, color=WHITE)

    # Parked truck (occluder)
    truck = add_rect(s, sx + Inches(2.6), sy + Inches(2.95), Inches(1.4),
                      Inches(0.65), fill=DARK_GRAY, line=DARK_GRAY,
                      shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    set_shape_text(truck, "PARKED TRUCK", size=10, color=WHITE)

    # Pedestrian (hidden behind truck)
    ped = s.shapes.add_shape(MSO_SHAPE.OVAL, sx + Inches(3.1),
                              sy + Inches(2.45), Inches(0.3), Inches(0.3))
    ped.fill.solid()
    ped.fill.fore_color.rgb = RED
    ped.line.fill.background()
    add_label(s, sx + Inches(2.9), sy + Inches(2.15), Inches(0.7),
              Inches(0.3), "pedestrian", size=10, color=RED)

    # View cone from ego
    cone = s.shapes.add_shape(MSO_SHAPE.RIGHT_TRIANGLE,
                              sx + Inches(1.6), sy + Inches(2.7),
                              Inches(1.1), Inches(1.2))
    cone.fill.solid()
    cone.fill.fore_color.rgb = RGBColor(0x00, 0xA9, 0xE0)
    cone.fill.transparency = 0.55
    cone.line.fill.background()
    cone.rotation = 0
    add_label(s, sx + Inches(1.6), sy + Inches(3.95), Inches(1.4),
              Inches(0.3), "ego field of view", size=9, italic=True,
              color=MED_GRAY)

    # Approaching car (right)
    car2 = add_rect(s, sx + Inches(4.8), sy + Inches(3.05), Inches(1.0),
                     Inches(0.5), fill=ORANGE, line=DARK_GRAY,
                     shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    set_shape_text(car2, "OTHER UE", size=10, color=WHITE)

    # Big X over the pedestrian-ego line of sight
    add_text_box(s, sx + Inches(2.0), sy + Inches(0.5), Inches(4.5),
                 Inches(0.6), "Pedestrian darts from behind the parked truck.",
                 font_size=13, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_text_box(s, sx + Inches(2.0), sy + Inches(0.95), Inches(4.5),
                 Inches(0.6), "Ego cannot see them in time. The car on the right CAN.",
                 font_size=12, italic=True, color=DARK_GRAY,
                 align=PP_ALIGN.CENTER)

    # RIGHT: takeaways
    add_text_box(s, Inches(7.1), Inches(1.55), Inches(5.8), Inches(0.5),
                 "Why this matters", font_size=20, bold=True, color=NAVY)
    add_bullets(s, Inches(7.1), Inches(2.1), Inches(5.8), Inches(5.0),
                [
                    ("Occlusion is unavoidable.",
                     "Buildings, trucks, terrain hide objects from any single sensor."),
                    ("More local compute does not help.",
                     "If pixels never reach the ego, no model can detect them."),
                    ("Cooperative perception fills the gap.",
                     "Other cars / RSUs / drones already see what we cannot."),
                    ("The bottleneck is the network.",
                     "Sharing must respect bandwidth, latency, and reliability budgets."),
                ], size=13)


# ============================================================
# SLIDE 4 -- What cooperative perception adds
# ============================================================
def slide_coop_perception_concept():
    s = add_blank_slide()
    add_title_bar(s, "Cooperative perception in one picture",
                   "Many partial views → one shared world model")

    # 3 cars on left -> network -> shared world model on right
    car_color = [ID_BLUE, ORANGE, GREEN]
    for i in range(3):
        cy = Inches(1.7 + i * 1.5)
        car = add_rect(s, Inches(0.6), cy, Inches(1.4), Inches(0.9),
                        fill=car_color[i], line=DARK_GRAY,
                        shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        set_shape_text(car, f"Vehicle {i+1}\nlocal view {i+1}", size=11,
                        color=WHITE)
        # cone
        add_text_box(s, Inches(0.6), cy + Inches(0.95), Inches(1.4),
                     Inches(0.3),
                     ["partial ↑ occluded sides",
                      "different angle",
                      "different range"][i],
                     font_size=9, italic=True, color=MED_GRAY,
                     align=PP_ALIGN.CENTER)
        # arrow toward network
        add_arrow(s, Inches(2.05), cy + Inches(0.45),
                  Inches(4.6), Inches(3.7),
                  color=car_color[i], line_width=2.25)

    # Network/gNB box in middle
    net = add_rect(s, Inches(4.55), Inches(3.05), Inches(2.3), Inches(1.4),
                    fill=WHITE, line=NAVY, line_width=2)
    set_shape_text(net,
                   "5G uplink\n(OAI gNB)",
                   size=14, bold=True, color=NAVY)

    # Fusion / shared world model on right
    fuse = add_rect(s, Inches(7.4), Inches(2.6), Inches(2.7), Inches(2.4),
                     fill=SOFT_BG, line=NAVY, line_width=2)
    set_shape_text(fuse,
                   "Edge Fusion\n+\nShared Spatial Map",
                   size=14, bold=True, color=NAVY)

    add_arrow(s, Inches(6.85), Inches(3.75), Inches(7.4), Inches(3.75),
              color=NAVY, line_width=2.5)

    # Receivers (vehicles) on right
    add_arrow(s, Inches(10.1), Inches(3.4), Inches(11.5), Inches(2.4),
              color=NAVY, line_width=2)
    add_arrow(s, Inches(10.1), Inches(3.75), Inches(11.5), Inches(3.75),
              color=NAVY, line_width=2)
    add_arrow(s, Inches(10.1), Inches(4.1), Inches(11.5), Inches(5.1),
              color=NAVY, line_width=2)

    for i in range(3):
        cy = Inches(2.0 + i * 1.4)
        v = add_rect(s, Inches(11.5), cy, Inches(1.4), Inches(0.7),
                      fill=car_color[i], line=DARK_GRAY,
                      shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        set_shape_text(v, f"Vehicle {i+1}\nnow sees more", size=10,
                        color=WHITE)

    # Caption
    add_text_box(s, Inches(0.4), Inches(6.4), Inches(12.6), Inches(0.6),
                 "Key idea: no single machine has the full truth. Each one shares a partial view; the network and an aggregator build a better shared world model than any of them could alone.",
                 font_size=14, italic=True, color=DARK_GRAY,
                 align=PP_ALIGN.CENTER)


# ============================================================
# SLIDE 5 -- Use cases
# ============================================================
def slide_use_cases():
    s = add_blank_slide()
    add_title_bar(s, "Where this matters — use cases",
                   "Same pattern: spatially separated sensors, partial views, shared map")

    use_cases = [
        ("Connected vehicles",
         "Cars share blind-spot perception around occluding trucks, intersections.",
         ID_BLUE, MSO_SHAPE.OVAL),
        ("Smart intersections",
         "Roadside cameras share pedestrian/cyclist locations to crossing vehicles.",
         ORANGE, MSO_SHAPE.RECTANGLE),
        ("Drones · aerial assist",
         "Overhead view shared with ground vehicles for hazards and terrain.",
         GREEN, MSO_SHAPE.OCTAGON),
        ("Warehouse robots",
         "Fleet shares object locations under shelving and partial occlusions.",
         PURPLE, MSO_SHAPE.PENTAGON),
        ("Industrial safety",
         "Machines share safety-zone awareness; humans entering hazardous areas.",
         RED, MSO_SHAPE.HEXAGON),
        ("Multi-camera security",
         "Distributed cameras fuse detections of persons/vehicles across a site.",
         NAVY, MSO_SHAPE.DIAMOND),
    ]
    # 2 rows x 3 cols grid
    grid_left = Inches(0.45)
    grid_top = Inches(1.45)
    cw = Inches(4.13)
    ch = Inches(2.7)
    gx = Inches(0.13)
    gy = Inches(0.18)
    for idx, (head, body, col, icon_shape) in enumerate(use_cases):
        r = idx // 3
        c = idx % 3
        left = grid_left + (cw + gx) * c
        top = grid_top + (ch + gy) * r
        card = add_rect(s, left, top, cw, ch, fill=WHITE,
                         line=LIGHT_GRAY, line_width=1.5)
        # Icon circle
        icon = s.shapes.add_shape(icon_shape, left + Inches(0.25),
                                   top + Inches(0.25),
                                   Inches(0.65), Inches(0.65))
        icon.fill.solid()
        icon.fill.fore_color.rgb = col
        icon.line.fill.background()
        # Title
        add_text_box(s, left + Inches(1.05), top + Inches(0.25),
                     cw - Inches(1.1), Inches(0.55), head,
                     font_size=16, bold=True, color=NAVY)
        # Body
        add_text_box(s, left + Inches(0.25), top + Inches(1.1),
                     cw - Inches(0.5), ch - Inches(1.3), body,
                     font_size=13, color=DARK_GRAY)
    # Footer note
    add_text_box(s, Inches(0.4), Inches(7.0), Inches(12.5), Inches(0.4),
                 "Our research target: connected vehicles + smart intersection; results extend to the others.",
                 font_size=12, italic=True, color=MED_GRAY,
                 align=PP_ALIGN.CENTER)


# ============================================================
# SLIDE 6 -- Sharing options: why none are free
# ============================================================
def slide_sharing_options():
    s = add_blank_slide()
    add_title_bar(s, "What should the vehicles share?",
                   "Three obvious options — only one is workable for safety")

    options = [
        ("Raw video / camera frames",
         ["Highest fidelity for fusion",
          "Huge bandwidth (Mbps per stream, scales with resolution)",
          "Bursty, latency-sensitive, expensive to encode"],
         "Too big for shared 5G uplink at fleet scale.",
         RED, "✗"),
        ("Final detections / boxes only",
         ["Tiny payload",
          "Easy to log and standardize",
          "Cannot be re-fused; loses confidence, geometry, context"],
         "Too thin: receiver cannot recover or argue with them.",
         ORANGE, "−"),
        ("Intermediate feature tensors (split inference)",
         ["Already a compressed, task-aligned summary",
          "Rich enough to re-run the head and to fuse across vehicles",
          "Small enough to ship under cellular budgets if managed"],
         "Sweet spot for cooperative perception over 5G.",
         GREEN, "✓"),
    ]
    x = Inches(0.4)
    y = Inches(1.4)
    w = Inches(4.18)
    h = Inches(5.55)
    gap = Inches(0.13)
    for i, (head, bullets, verdict, col, mark) in enumerate(options):
        left = x + (w + gap) * i
        card = add_rect(s, left, y, w, h, fill=WHITE,
                         line=col, line_width=1.5)
        # Header
        band = add_rect(s, left, y, w, Inches(0.7), fill=col,
                        line=col, shape=MSO_SHAPE.RECTANGLE)
        set_shape_text(band, f"{mark}  {head}", size=15, bold=True,
                        color=WHITE)
        # Bullets
        add_bullets(s, left, y + Inches(0.85), w, Inches(3.4),
                    bullets, size=13)
        # Verdict band at bottom
        verdict_box = add_rect(s, left + Inches(0.2),
                                y + Inches(4.4), w - Inches(0.4),
                                Inches(1.0), fill=SOFT_BG,
                                line=col, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        set_shape_text(verdict_box, verdict, size=13, bold=True,
                        color=col)


# ============================================================
# SLIDE 7 -- What is split inferencing
# ============================================================
def slide_split_what():
    s = add_blank_slide()
    add_title_bar(s, "What is split inferencing?",
                   "Run the first half of the model on the UE, ship features, run the rest on the edge")

    # Top diagram band
    band_y = Inches(1.5)
    band_h = Inches(3.0)
    # UE box
    ue = add_rect(s, Inches(0.6), band_y, Inches(3.6), band_h,
                   fill=SOFT_BG, line=NAVY, line_width=2)
    add_text_box(s, Inches(0.6), band_y + Inches(0.05),
                  Inches(3.6), Inches(0.4),
                  "UE (vehicle / pole)", font_size=14, bold=True,
                  color=NAVY, align=PP_ALIGN.CENTER)
    # Inside UE: camera -> backbone
    cam = add_rect(s, Inches(0.85), band_y + Inches(0.7),
                    Inches(1.3), Inches(1.0),
                    fill=WHITE, line=DARK_GRAY)
    set_shape_text(cam, "Camera /\nSensor", size=12, color=DARK_GRAY)
    bb = add_rect(s, Inches(2.6), band_y + Inches(0.7),
                   Inches(1.4), Inches(1.0),
                   fill=ID_BLUE, line=NAVY)
    set_shape_text(bb, "Backbone\n(MobileNetV3,\nResNet, ...)", size=11,
                    color=WHITE)
    add_arrow(s, Inches(2.15), band_y + Inches(1.2),
              Inches(2.6), band_y + Inches(1.2),
              color=DARK_GRAY)

    # Split point marker
    add_text_box(s, Inches(0.85), band_y + Inches(1.95),
                  Inches(3.15), Inches(0.3),
                  "feature tensor [C, H, W]", font_size=11, italic=True,
                  color=NAVY, align=PP_ALIGN.CENTER)
    add_text_box(s, Inches(0.85), band_y + Inches(2.28),
                  Inches(3.15), Inches(0.3),
                  "(the SPLIT POINT)", font_size=11, bold=True,
                  color=NAVY, align=PP_ALIGN.CENTER)

    # Compression block between UE and network
    comp = add_rect(s, Inches(4.55), band_y + Inches(0.7),
                     Inches(1.4), Inches(1.6),
                     fill=ORANGE, line=DARK_GRAY)
    set_shape_text(comp, "Compress\n+\nQuantize", size=11, color=WHITE)
    add_arrow(s, Inches(4.0), band_y + Inches(1.2),
              Inches(4.55), band_y + Inches(1.2), color=DARK_GRAY)

    # Network cloud
    net = add_rect(s, Inches(6.3), band_y + Inches(0.4),
                    Inches(1.5), Inches(2.1),
                    fill=SOFT_BG, line=NAVY, line_width=2,
                    shape=MSO_SHAPE.CLOUD)
    set_shape_text(net, "5G uplink", size=12, bold=True, color=NAVY)
    add_arrow(s, Inches(5.95), band_y + Inches(1.5),
              Inches(6.3), band_y + Inches(1.5),
              color=NAVY)

    # Edge server box
    srv = add_rect(s, Inches(8.2), band_y, Inches(4.5), band_h,
                    fill=SOFT_BG, line=NAVY, line_width=2)
    add_text_box(s, Inches(8.2), band_y + Inches(0.05),
                  Inches(4.5), Inches(0.4),
                  "Edge / cloud server", font_size=14, bold=True,
                  color=NAVY, align=PP_ALIGN.CENTER)
    dq = add_rect(s, Inches(8.4), band_y + Inches(0.7),
                   Inches(1.3), Inches(1.0),
                   fill=ORANGE, line=DARK_GRAY)
    set_shape_text(dq, "De-quant\nDecompress", size=11, color=WHITE)
    head = add_rect(s, Inches(10.0), band_y + Inches(0.7),
                     Inches(1.4), Inches(1.0),
                     fill=NAVY, line=NAVY)
    set_shape_text(head, "Head\n(OD / SEG)", size=11, color=WHITE)
    out = add_rect(s, Inches(11.6), band_y + Inches(0.7),
                    Inches(1.0), Inches(1.0),
                    fill=WHITE, line=DARK_GRAY)
    set_shape_text(out, "Boxes /\nMasks", size=11, color=DARK_GRAY)
    add_arrow(s, Inches(7.8), band_y + Inches(1.5),
              Inches(8.4), band_y + Inches(1.2),
              color=NAVY)
    add_arrow(s, Inches(9.7), band_y + Inches(1.2),
              Inches(10.0), band_y + Inches(1.2),
              color=DARK_GRAY)
    add_arrow(s, Inches(11.4), band_y + Inches(1.2),
              Inches(11.6), band_y + Inches(1.2),
              color=DARK_GRAY)
    # Return arrow under
    add_text_box(s, Inches(8.2), band_y + Inches(2.0),
                  Inches(4.5), Inches(0.5),
                  "result returned to UE (UDP) for local control",
                  font_size=10, italic=True, color=MED_GRAY,
                  align=PP_ALIGN.CENTER)

    # Bottom takeaways
    add_bullets(s, Inches(0.5), Inches(4.85), Inches(12.4), Inches(2.0),
                [
                    ("Definition.",
                     "Cut a deep model at a chosen intermediate layer; the UE runs the backbone, the server runs the head."),
                    ("What is shipped.",
                     "Not pixels and not final detections — a compact feature tensor representing what the backbone saw."),
                    ("What is configurable.",
                     "Where to cut, how to compress (AE / quantize / ROI), and how often to send."),
                ], size=14)


# ============================================================
# SLIDE 8 -- Why split inferencing
# ============================================================
def slide_split_why():
    s = add_blank_slide()
    add_title_bar(s, "Why split inferencing?",
                   "Four reasons it beats both “ship raw video” and “ship only detections”")

    reasons = [
        ("Bandwidth savings", "Tensors after quantization + entropy coding are 1–2 orders of magnitude smaller than equivalent raw RGB at the same task fidelity.", ID_BLUE),
        ("Edge compute reuse", "UE runs only the backbone (cheap); heavy heads (OD / SEG) run once on the edge for many UEs — fleet economics work.", ORANGE),
        ("Privacy posture", "Raw pixels never leave the device; only task-relevant features cross the radio.", GREEN),
        ("Fusion-ready", "Feature tensors are richer than detections — server can re-run the head, fuse multiple UEs, and reason about confidence.", PURPLE),
    ]
    x = Inches(0.4)
    y = Inches(1.55)
    w = Inches(6.25)
    h = Inches(2.55)
    gx = Inches(0.18)
    gy = Inches(0.2)
    for i, (head, body, col) in enumerate(reasons):
        r = i // 2
        c = i % 2
        left = x + (w + gx) * c
        top = y + (h + gy) * r
        card = add_rect(s, left, top, w, h, fill=WHITE, line=col, line_width=1.5)
        # Left color strip
        strip = add_rect(s, left, top, Inches(0.22), h, fill=col,
                         line=col, shape=MSO_SHAPE.RECTANGLE)
        # Number
        num_circle = s.shapes.add_shape(MSO_SHAPE.OVAL,
                                        left + Inches(0.45),
                                        top + Inches(0.3),
                                        Inches(0.55), Inches(0.55))
        num_circle.fill.solid()
        num_circle.fill.fore_color.rgb = col
        num_circle.line.fill.background()
        set_shape_text(num_circle, str(i + 1), size=18, bold=True,
                        color=WHITE)

        add_text_box(s, left + Inches(1.15), top + Inches(0.25),
                     w - Inches(1.3), Inches(0.6), head,
                     font_size=18, bold=True, color=NAVY)
        add_text_box(s, left + Inches(1.15), top + Inches(0.85),
                     w - Inches(1.3), h - Inches(0.95), body,
                     font_size=13, color=DARK_GRAY)
    # Tagline
    add_text_box(s, Inches(0.4), Inches(6.85), Inches(12.5), Inches(0.4),
                 "Net effect: a transport-friendly intermediate format that keeps task utility high.",
                 font_size=13, italic=True, color=NAVY,
                 align=PP_ALIGN.CENTER)


# ============================================================
# SLIDE 9 -- Methodology with architecture
# ============================================================
def slide_split_methodology():
    s = add_blank_slide()
    add_title_bar(s, "Methodology — split-inference architecture",
                   "Two heads (OD, SEG) share the UE pipeline; the network sees the same kind of payload")

    # Big architecture
    diagram_y = Inches(1.5)
    diagram_h = Inches(4.4)

    # Common UE column
    ue = add_rect(s, Inches(0.5), diagram_y, Inches(3.4), diagram_h,
                   fill=SOFT_BG, line=NAVY, line_width=2)
    add_text_box(s, Inches(0.5), diagram_y + Inches(0.05),
                  Inches(3.4), Inches(0.4),
                  "UE side", font_size=13, bold=True, color=NAVY,
                  align=PP_ALIGN.CENTER)

    # Camera
    cam = add_rect(s, Inches(0.75), diagram_y + Inches(0.6),
                    Inches(1.0), Inches(0.7),
                    fill=WHITE, line=DARK_GRAY)
    set_shape_text(cam, "Camera", size=11)
    # Radar (optional, fusion path)
    rad = add_rect(s, Inches(0.75), diagram_y + Inches(1.45),
                    Inches(1.0), Inches(0.6),
                    fill=WHITE, line=DARK_GRAY)
    set_shape_text(rad, "Radar\n(optional)", size=10)
    # Backbone
    bb = add_rect(s, Inches(2.1), diagram_y + Inches(0.85),
                   Inches(1.5), Inches(1.05),
                   fill=ID_BLUE, line=NAVY)
    set_shape_text(bb,
                   "Backbone\nMobileNetV3 /\nResNet+FPN",
                   size=10, color=WHITE)
    add_arrow(s, Inches(1.75), diagram_y + Inches(0.95),
              Inches(2.1), diagram_y + Inches(1.2), color=DARK_GRAY)
    add_arrow(s, Inches(1.75), diagram_y + Inches(1.75),
              Inches(2.1), diagram_y + Inches(1.5), color=DARK_GRAY)

    # split-point marker
    split = add_rect(s, Inches(2.1), diagram_y + Inches(2.15),
                      Inches(1.5), Inches(0.45),
                      fill=ORANGE, line=DARK_GRAY)
    set_shape_text(split, "SPLIT POINT", size=10, color=WHITE)
    add_arrow(s, Inches(2.85), diagram_y + Inches(1.95),
              Inches(2.85), diagram_y + Inches(2.15), color=DARK_GRAY)

    # Compression block
    comp = add_rect(s, Inches(2.1), diagram_y + Inches(2.85),
                     Inches(1.5), Inches(1.3),
                     fill=WHITE, line=ORANGE, line_width=1.5)
    set_shape_text(comp,
                   "AE  /  ROI\nQuantize\nEntropy code",
                   size=10, color=DARK_GRAY)
    add_arrow(s, Inches(2.85), diagram_y + Inches(2.6),
              Inches(2.85), diagram_y + Inches(2.85), color=DARK_GRAY)

    # Arrow to network
    add_arrow(s, Inches(3.65), diagram_y + Inches(3.5),
              Inches(4.4), diagram_y + Inches(2.2),
              color=NAVY, line_width=2.0)

    # Network cloud (middle)
    net = add_rect(s, Inches(4.4), diagram_y + Inches(1.7),
                    Inches(1.6), Inches(1.3),
                    fill=SOFT_BG, line=NAVY, line_width=2,
                    shape=MSO_SHAPE.CLOUD)
    set_shape_text(net, "OAI 5G\nuplink (UDP)", size=11, color=NAVY,
                    bold=True)

    add_arrow(s, Inches(6.0), diagram_y + Inches(2.35),
              Inches(6.7), diagram_y + Inches(2.35),
              color=NAVY, line_width=2.0)

    # Server column (right): TWO heads (OD and SEG)
    srv = add_rect(s, Inches(6.7), diagram_y, Inches(6.1), diagram_h,
                    fill=SOFT_BG, line=NAVY, line_width=2)
    add_text_box(s, Inches(6.7), diagram_y + Inches(0.05),
                  Inches(6.1), Inches(0.4),
                  "Edge AI server", font_size=13, bold=True, color=NAVY,
                  align=PP_ALIGN.CENTER)

    # Decompression
    dq = add_rect(s, Inches(6.95), diagram_y + Inches(1.9),
                   Inches(1.4), Inches(0.9),
                   fill=WHITE, line=ORANGE)
    set_shape_text(dq, "De-quant\nDecompress", size=10)

    # OD branch
    od = add_rect(s, Inches(8.7), diagram_y + Inches(0.85),
                   Inches(1.7), Inches(1.0),
                   fill=NAVY, line=NAVY)
    set_shape_text(od, "OD head\nFaster R-CNN", size=11, color=WHITE)
    od_out = add_rect(s, Inches(10.7), diagram_y + Inches(0.95),
                       Inches(1.9), Inches(0.8),
                       fill=WHITE, line=NAVY)
    set_shape_text(od_out, "Object boxes\n+ classes", size=11,
                    color=DARK_GRAY)
    # SEG branch
    seg = add_rect(s, Inches(8.7), diagram_y + Inches(2.85),
                    Inches(1.7), Inches(1.0),
                    fill=NAVY, line=NAVY)
    set_shape_text(seg, "SEG head\nLR-ASPP", size=11, color=WHITE)
    seg_out = add_rect(s, Inches(10.7), diagram_y + Inches(2.95),
                       Inches(1.9), Inches(0.8),
                       fill=WHITE, line=NAVY)
    set_shape_text(seg_out, "Per-pixel mask\n(classes)", size=11,
                    color=DARK_GRAY)

    # connectors
    add_arrow(s, Inches(8.35), diagram_y + Inches(2.35),
              Inches(8.7), diagram_y + Inches(1.35), color=DARK_GRAY)
    add_arrow(s, Inches(8.35), diagram_y + Inches(2.35),
              Inches(8.7), diagram_y + Inches(3.35), color=DARK_GRAY)
    add_arrow(s, Inches(10.4), diagram_y + Inches(1.35),
              Inches(10.7), diagram_y + Inches(1.35), color=DARK_GRAY)
    add_arrow(s, Inches(10.4), diagram_y + Inches(3.35),
              Inches(10.7), diagram_y + Inches(3.35), color=DARK_GRAY)

    # Bottom callouts
    add_bullets(s, Inches(0.5), Inches(6.0), Inches(12.4), Inches(1.3),
                [
                    ("Same UE pipeline.", "Camera (± radar fusion) → backbone → split point → compress → UDP."),
                    ("Two different heads.", "OD ships small fixed-shape features; SEG ships much larger spatially detailed features."),
                    ("Same compression knobs.", "AE channels, ROI threshold, quantization bits, frame schedule, redundancy."),
                ], size=13)


# ============================================================
# SLIDE 10 -- Compression knobs
# ============================================================
def slide_compression_knobs():
    s = add_blank_slide()
    add_title_bar(s, "The compression knobs at the split point",
                   "What the agent will eventually control — today they are static settings")

    knobs = [
        ("AE channels (128 / 64 / 32)",
         "Bottleneck auto-encoder replaces feature channels with a smaller learned latent. Lower channels → fewer bytes, more reconstruction error.",
         ID_BLUE),
        ("ROI threshold (0.1 / 0.3 / 0.5)",
         "Saliency/objectness gate zeros out cells below the threshold. Higher threshold → smaller payload, more chance to drop weak true positives.",
         ORANGE),
        ("Quantization (8 / 6 / 4 bit)",
         "Per-channel uint8 → uint4. Big byte savings; numerical floor in feature values gradually hurts mIoU and recall.",
         GREEN),
        ("Frame send / skip",
         "Send every Nth frame; UE updates locally on skipped frames. Saves bytes and latency under congestion, increases staleness.",
         PURPLE),
        ("Redundancy (FEC / dup)",
         "Add parity or duplicate chunks when loss is high. Costs bytes; protects feature integrity under bad channel.",
         NAVY),
    ]
    x = Inches(0.45)
    y = Inches(1.55)
    w = Inches(12.45)
    each_h = Inches(1.0)
    gap = Inches(0.12)
    for i, (head, body, col) in enumerate(knobs):
        top = y + (each_h + gap) * i
        card = add_rect(s, x, top, w, each_h, fill=WHITE, line=col, line_width=1.5)
        # Color block on left
        cb = add_rect(s, x, top, Inches(2.7), each_h, fill=col, line=col,
                      shape=MSO_SHAPE.RECTANGLE)
        set_shape_text(cb, head, size=13, bold=True, color=WHITE)
        # Body
        add_text_box(s, x + Inches(2.9), top + Inches(0.08),
                     w - Inches(3.1), each_h - Inches(0.1),
                     body, font_size=13, color=DARK_GRAY,
                     anchor=MSO_ANCHOR.MIDDLE)


# ============================================================
# SLIDE 11 -- What we measured so far
# ============================================================
def slide_measurements():
    s = add_blank_slide()
    add_title_bar(s, "Baseline measurements (single-UE, our testbed)",
                   "Why OD and SEG are different network-sizing problems")

    # Left table
    table_x = Inches(0.5)
    table_y = Inches(1.6)
    table_w = Inches(7.4)
    rows = [
        ["Model config", "p50 payload", "p95 payload", "p50 RTT", "Chunks"],
        ["OD baseline", "~86.9 KB", "~88.1 KB", "~14 ms", "2"],
        ["OD ROI 0.4", "~30.4 KB", "~55.0 KB", "~13 ms", "1"],
        ["OD RD-AE 128", "~36.6 KB", "~38.2 KB", "~13 ms", "1"],
        ["SEG baseline", "~409.7 KB", "~425.2 KB", "~41 ms", "7–8"],
        ["SEG uint4", "~217.6 KB", "~222.9 KB", "~33 ms", "4"],
        ["SEG ROI 0.1", "~392.6 KB", "~415.5 KB", "~42 ms", "7–8"],
    ]
    row_h = Inches(0.55)
    col_widths = [Inches(2.4), Inches(1.4), Inches(1.4), Inches(1.0), Inches(1.2)]

    cx = table_x
    cy = table_y
    for r_idx, row in enumerate(rows):
        col_x = cx
        for c_idx, val in enumerate(row):
            cell = add_rect(s, col_x, cy, col_widths[c_idx], row_h,
                            fill=NAVY if r_idx == 0 else (
                                SOFT_BG if r_idx % 2 else WHITE),
                            line=LIGHT_GRAY, shape=MSO_SHAPE.RECTANGLE)
            set_shape_text(cell, val,
                            size=12 if r_idx == 0 else 11,
                            bold=(r_idx == 0),
                            color=WHITE if r_idx == 0 else DARK_GRAY)
            col_x += col_widths[c_idx]
        cy += row_h

    # Right: takeaways
    add_text_box(s, Inches(8.2), Inches(1.55), Inches(4.7), Inches(0.5),
                 "What this tells us", font_size=18, bold=True,
                 color=NAVY)
    add_bullets(s, Inches(8.2), Inches(2.1), Inches(4.7), Inches(4.5),
                [
                    ("SEG ≫ OD.", "SEG payload is ~4.6× OD baseline."),
                    ("Static knobs cut bytes.", "AE / ROI / uint4 reduce 1.5–4× — with different task cost."),
                    ("Knobs are not equivalent.", "Same byte budget reached different ways gives different accuracy."),
                    ("5QI exposes burst, not just latency.",
                     "Even compressed SEG exceeds MDBV for 5QI 89/90 — traffic profile is the bottleneck."),
                ], size=13)

    # Bottom hook
    add_text_box(s, Inches(0.5), Inches(6.5), Inches(12.4), Inches(0.7),
                 "These are STATIC choices today. The next step is to LEARN which knob setting to use, per scene, per network state, per task.",
                 font_size=14, bold=True, italic=True, color=NAVY,
                 align=PP_ALIGN.CENTER)


# ============================================================
# SLIDE 12 -- Bridge: single-UE -> multi-UE cooperative
# ============================================================
def slide_bridge():
    s = add_blank_slide()
    add_title_bar(s, "From single-UE split inference to multi-UE cooperative perception",
                   "The same plumbing, scaled — and now we need to coordinate")

    # LEFT panel: single UE
    lx = Inches(0.5)
    ly = Inches(1.5)
    lw = Inches(6.0)
    lh = Inches(5.0)
    add_rect(s, lx, ly, lw, lh, fill=SOFT_BG, line=LIGHT_GRAY)
    add_text_box(s, lx, ly + Inches(0.05), lw, Inches(0.4),
                  "Today: single UE", font_size=14, bold=True, color=NAVY,
                  align=PP_ALIGN.CENTER)

    car = add_rect(s, lx + Inches(0.3), ly + Inches(2.0),
                    Inches(1.3), Inches(0.8),
                    fill=ID_BLUE, line=NAVY,
                    shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    set_shape_text(car, "UE", size=12, color=WHITE)
    gnb = add_rect(s, lx + Inches(2.2), ly + Inches(2.0),
                    Inches(1.3), Inches(0.8),
                    fill=WHITE, line=NAVY)
    set_shape_text(gnb, "gNB", size=12, color=NAVY)
    srv = add_rect(s, lx + Inches(4.1), ly + Inches(2.0),
                    Inches(1.6), Inches(0.8),
                    fill=NAVY, line=NAVY)
    set_shape_text(srv, "Edge head", size=12, color=WHITE)
    add_arrow(s, lx + Inches(1.6), ly + Inches(2.4),
              lx + Inches(2.2), ly + Inches(2.4), color=NAVY)
    add_arrow(s, lx + Inches(3.5), ly + Inches(2.4),
              lx + Inches(4.1), ly + Inches(2.4), color=NAVY)

    add_bullets(s, lx + Inches(0.3), ly + Inches(3.2), lw - Inches(0.6),
                Inches(1.6),
                [
                    "One uplink path, one task at a time",
                    "Compression decisions made locally, in isolation",
                    "Network state ≈ stable",
                    "Already characterized in the AI-traffic deck",
                ], size=12)

    # RIGHT panel: multi-UE
    rx = Inches(6.85)
    ry = Inches(1.5)
    rw = Inches(6.05)
    rh = Inches(5.0)
    add_rect(s, rx, ry, rw, rh, fill=SOFT_BG, line=ID_BLUE, line_width=2)
    add_text_box(s, rx, ry + Inches(0.05), rw, Inches(0.4),
                  "Where we are going: many UEs, shared world",
                  font_size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    # 3 UEs stacked
    car_colors = [ID_BLUE, ORANGE, GREEN]
    for i in range(3):
        cy = ry + Inches(0.7 + i * 1.15)
        ci = add_rect(s, rx + Inches(0.2), cy, Inches(1.1),
                       Inches(0.65), fill=car_colors[i], line=DARK_GRAY,
                       shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        set_shape_text(ci, f"UE {i+1}", size=11, color=WHITE)
        add_arrow(s, rx + Inches(1.3), cy + Inches(0.32),
                  rx + Inches(2.4), ry + Inches(2.2),
                  color=car_colors[i], line_width=2.0)
    # gNB shared
    gnb2 = add_rect(s, rx + Inches(2.4), ry + Inches(1.8),
                     Inches(1.1), Inches(0.8),
                     fill=WHITE, line=NAVY, line_width=2)
    set_shape_text(gnb2, "gNB\n(shared)", size=10, color=NAVY)
    # Edge fusion + map
    fuse = add_rect(s, rx + Inches(3.9), ry + Inches(1.4),
                     Inches(1.9), Inches(1.6),
                     fill=NAVY, line=NAVY)
    set_shape_text(fuse, "Edge fusion\n+\nShared spatial map",
                    size=11, color=WHITE)
    add_arrow(s, rx + Inches(3.5), ry + Inches(2.2),
              rx + Inches(3.9), ry + Inches(2.2), color=NAVY, line_width=2.0)

    add_bullets(s, rx + Inches(0.3), ry + Inches(3.4), rw - Inches(0.6),
                Inches(1.6),
                [
                    "Shared radio resource pool → they compete",
                    "Different scenes → different per-UE budgets",
                    "Fusion needs confidence + freshness, not just boxes",
                    "Compression decisions must now be coordinated and network-aware",
                ], size=12)


# ============================================================
# SLIDE 13 -- The idea (SceneSense Agent)
# ============================================================
def slide_scenesense_idea():
    s = add_blank_slide()
    add_title_bar(s, "SceneSense Agent — the core idea",
                   "Learn what to share, when to share it, and how much to spend on it")

    # Quote-style big text on the left
    add_rich_text_box(s, Inches(0.5), Inches(1.5), Inches(6.7), Inches(5.4),
                      [
                          {"text": "“", "size": 60, "color": ID_BLUE, "bold": True},
                          {"text": "Can a machine learn, in real time, what visual information is worth sending over a busy wireless link so that safety-critical objects are still understood, bandwidth is not wasted, and nearby autonomous machines can be warned about hazards they cannot directly see?",
                           "size": 17, "italic": True, "color": NAVY, "space_after": 18},
                          {"text": "  ", "size": 12},
                          {"text": "— SceneSense Agent research proposal, May 2026",
                           "size": 11, "italic": True, "color": MED_GRAY},
                      ])
    # Right side: bullet of "what we are trying to do"
    add_text_box(s, Inches(7.4), Inches(1.55), Inches(5.5), Inches(0.5),
                 "What we are trying to do", font_size=20, bold=True,
                 color=NAVY)
    add_bullets(s, Inches(7.4), Inches(2.1), Inches(5.5), Inches(5.0),
                [
                    ("Learn policies, not knobs.",
                     "A constrained RL agent picks AE / ROI / quant / scheduling / redundancy live."),
                    ("Use three kinds of state.",
                     "Scene (what we see), network (link health), model (confidence)."),
                    ("Guard task precision.",
                     "Hard cap on AP, mIoU, foreground IoU, and pedestrian / cyclist recall drop."),
                    ("Feed a shared spatial map.",
                     "Accepted outputs populate a confidence-tagged map for occluded-object alerts."),
                    ("Demonstrate the value.",
                     "An autonomous vehicle uses the shared map to avoid a hazard it cannot directly see."),
                ], size=13)


# ============================================================
# SLIDE 14 -- Hypothesis & goal
# ============================================================
def slide_hypothesis():
    s = add_blank_slide()
    add_title_bar(s, "Hypothesis and goal",
                   "Concrete, testable claims that the 6-month plan is built around")

    # Two columns: Hypotheses (left), Goal (right)
    add_text_box(s, Inches(0.5), Inches(1.55), Inches(7.0), Inches(0.5),
                 "Hypotheses", font_size=20, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(2.1), Inches(7.0), Inches(5.0),
                [
                    ("H1.", "A constrained RL controller outperforms the best static knob policy at the same task fidelity."),
                    ("H2.", "Task-precision guardrails prevent byte-minimizing policies from destroying AP / mIoU / safety-class recall."),
                    ("H3.", "Controller outputs (confidence, foreground regions, freshness) are usable as physical-AI signals."),
                    ("H4.", "A learned map-sharing policy reduces avoidable collision risk in occluded intersection scenarios."),
                ], size=14)

    # Right: First implementation goal
    add_text_box(s, Inches(7.8), Inches(1.55), Inches(5.2), Inches(0.5),
                 "First implementation goal", font_size=20, bold=True,
                 color=NAVY)
    goal_box = add_rect(s, Inches(7.8), Inches(2.1), Inches(5.2),
                        Inches(2.8), fill=SOFT_BG, line=ID_BLUE,
                        line_width=2)
    set_shape_text(goal_box,
                   "Learned split-model control\nunder task-precision guardrails",
                   size=18, bold=True, color=NAVY)
    add_text_box(s, Inches(7.8), Inches(5.0), Inches(5.2), Inches(2.0),
                 "Task utility = perception value preserved AFTER compression (AP, mIoU, foreground IoU, pedestrian / cyclist / small-object recall).",
                 font_size=12, italic=True, color=DARK_GRAY)
    add_text_box(s, Inches(7.8), Inches(5.85), Inches(5.2), Inches(1.4),
                 "Bytes / latency / loss are costs, not goals. Task utility is the first-class objective.",
                 font_size=12, italic=True, color=DARK_GRAY)


# ============================================================
# SLIDE 15 -- Things to consider
# ============================================================
def slide_things_to_consider():
    s = add_blank_slide()
    add_title_bar(s, "Things we have to get right to make this work",
                   "Each axis below is a research question and an experimental risk")

    items = [
        ("Scene awareness",
         "How do we represent “what is happening in this frame” compactly? Density, foreground fraction, object scale, occlusion ratio.",
         ID_BLUE),
        ("Network awareness",
         "Which 5G signals are usable in real time? RTT, packet loss, queue delay, scheduling grants. OAI exposes some but not all.",
         ORANGE),
        ("Model awareness",
         "How confident is the model on this frame? Per-class confidence, uncertainty, foreground IoU proxy.",
         GREEN),
        ("Task-precision guardrails",
         "Reject any action that drops AP / mIoU below a configured floor or that increases small-object misses.",
         RED),
        ("Coordination scope",
         "Per-UE local agent first; later, decide whether sharing decisions stay local or coordinate across UEs.",
         PURPLE),
        ("Action latency",
         "The policy itself has to act inside the safety-critical window — it cannot be slower than the perception it controls.",
         NAVY),
    ]
    # 3 columns x 2 rows
    x = Inches(0.4)
    y = Inches(1.45)
    w = Inches(4.18)
    h = Inches(2.8)
    gx = Inches(0.13)
    gy = Inches(0.15)
    for i, (head, body, col) in enumerate(items):
        r = i // 3
        c = i % 3
        left = x + (w + gx) * c
        top = y + (h + gy) * r
        card = add_rect(s, left, top, w, h, fill=WHITE, line=col, line_width=1.5)
        band = add_rect(s, left, top, w, Inches(0.6),
                        fill=col, line=col, shape=MSO_SHAPE.RECTANGLE)
        set_shape_text(band, head, size=14, bold=True, color=WHITE)
        add_text_box(s, left + Inches(0.15), top + Inches(0.75),
                     w - Inches(0.3), h - Inches(0.85),
                     body, font_size=12, color=DARK_GRAY)


# ============================================================
# SLIDE 16 -- SceneSense Big Picture
# ============================================================
def slide_big_picture():
    s = add_blank_slide()
    add_title_bar(s, "SceneSense — the big picture",
                   "UE-side RL agent + Edge AI server + Shared Spatial Map + Map-sharing agent, all under task-precision guardrails")

    # We build a compact rendering of the architecture they had in the image.
    # 3 UEs on left, gNB+5G core in middle-left, Edge AI server middle-right,
    # Shared Spatial Map far right, Map-sharing agent below right,
    # Guardrails feedback ribbon at the bottom.

    # UE cluster
    ue_colors = [ID_BLUE, ORANGE, GREEN]
    ue_labels = ["Car 1", "Car 2", "Car 3"]
    for i in range(3):
        y0 = Inches(1.45 + i * 1.55)
        # Car icon
        car = add_rect(s, Inches(0.4), y0 + Inches(0.18),
                        Inches(0.95), Inches(0.55),
                        fill=ue_colors[i], line=DARK_GRAY,
                        shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        set_shape_text(car, ue_labels[i], size=10, color=WHITE)
        # Agent box
        agent = add_rect(s, Inches(1.45), y0, Inches(2.6),
                          Inches(1.25),
                          fill=WHITE, line=ue_colors[i], line_width=1.5)
        add_text_box(s, Inches(1.45), y0 + Inches(0.02),
                      Inches(2.6), Inches(0.32),
                      "Split-Control RL agent", font_size=11, bold=True,
                      color=NAVY, align=PP_ALIGN.CENTER)
        add_text_box(s, Inches(1.5), y0 + Inches(0.35),
                      Inches(2.5), Inches(0.85),
                      "state: scene, model, network\naction: AE / ROI / quant / sched / FEC",
                      font_size=9, color=DARK_GRAY,
                      align=PP_ALIGN.CENTER)
        # Arrow to gNB
        add_arrow(s, Inches(4.05), y0 + Inches(0.65),
                  Inches(5.0), Inches(3.75),
                  color=ue_colors[i], line_width=1.75)

    # gNB / 5G core
    gnb = add_rect(s, Inches(5.0), Inches(3.4), Inches(1.5), Inches(0.95),
                    fill=WHITE, line=NAVY, line_width=2)
    set_shape_text(gnb, "gNB\nOAI 5G", size=11, bold=True, color=NAVY)
    upf = add_rect(s, Inches(5.0), Inches(4.5), Inches(1.5), Inches(0.7),
                    fill=WHITE, line=NAVY, line_width=2)
    set_shape_text(upf, "5G Core / UPF", size=10, color=NAVY)

    # Edge AI server
    srv = add_rect(s, Inches(7.0), Inches(2.85), Inches(2.8), Inches(2.6),
                    fill=SOFT_BG, line=NAVY, line_width=2)
    add_text_box(s, Inches(7.0), Inches(2.88), Inches(2.8), Inches(0.4),
                  "Edge AI server (back half)", font_size=12, bold=True,
                  color=NAVY, align=PP_ALIGN.CENTER)
    od_b = add_rect(s, Inches(7.2), Inches(3.35),
                     Inches(2.4), Inches(0.55),
                     fill=NAVY, line=NAVY)
    set_shape_text(od_b, "OD head → boxes / pose", size=10, color=WHITE)
    seg_b = add_rect(s, Inches(7.2), Inches(4.0),
                      Inches(2.4), Inches(0.55),
                      fill=NAVY, line=NAVY)
    set_shape_text(seg_b, "SEG head → mask", size=10, color=WHITE)
    conf_b = add_rect(s, Inches(7.2), Inches(4.65),
                       Inches(2.4), Inches(0.65),
                       fill=PURPLE, line=PURPLE)
    set_shape_text(conf_b,
                   "Confidence /\nuncertainty estimator",
                   size=10, color=WHITE)

    # Arrow gNB -> server
    add_arrow(s, Inches(6.5), Inches(3.85),
              Inches(7.0), Inches(3.85), color=NAVY, line_width=2.0)

    # Shared spatial map
    smap = add_rect(s, Inches(10.1), Inches(1.45), Inches(2.85),
                     Inches(2.4),
                     fill=SOFT_BG, line=ORANGE, line_width=2)
    add_text_box(s, Inches(10.1), Inches(1.48), Inches(2.85),
                  Inches(0.45),
                  "Shared spatial map", font_size=13, bold=True,
                  color=ORANGE, align=PP_ALIGN.CENTER)
    add_text_box(s, Inches(10.2), Inches(1.95), Inches(2.7),
                  Inches(1.9),
                  "class, pose, velocity, confidence, provenance, freshness, occlusion state",
                  font_size=10, italic=True, color=DARK_GRAY,
                  align=PP_ALIGN.CENTER)

    # Map-sharing agent (below the shared map)
    msa = add_rect(s, Inches(10.1), Inches(4.0), Inches(2.85),
                    Inches(1.45),
                    fill=WHITE, line=ORANGE, line_width=2)
    add_text_box(s, Inches(10.1), Inches(4.03), Inches(2.85),
                  Inches(0.35),
                  "Map-sharing RL agent", font_size=12, bold=True,
                  color=ORANGE, align=PP_ALIGN.CENTER)
    add_text_box(s, Inches(10.15), Inches(4.4), Inches(2.75),
                  Inches(1.0),
                  "action: what / when / who / detail level\nreward: task utility − bytes − latency − stale risk",
                  font_size=10, color=DARK_GRAY, align=PP_ALIGN.CENTER)

    # Arrow server -> spatial map
    add_arrow(s, Inches(9.8), Inches(3.3),
              Inches(10.1), Inches(2.4), color=ORANGE, line_width=2.0)

    # Guardrails ribbon at bottom
    grd = add_rect(s, Inches(0.4), Inches(6.1), Inches(12.5),
                    Inches(0.85), fill=RGBColor(0xFD, 0xEC, 0xEA),
                    line=RED, line_width=1.5)
    add_text_box(s, Inches(0.55), Inches(6.15), Inches(12.2),
                  Inches(0.35),
                  "Task-precision guardrails  ·  Constraints (must hold)",
                  font_size=12, bold=True, color=RED)
    add_text_box(s, Inches(0.55), Inches(6.45), Inches(12.2),
                  Inches(0.45),
                  "AP drop ≤ ε_OD   ·   mIoU drop ≤ ε_SEG   ·   Recall_ped/cyclist ≥ τ   ·   latency ≤ L_max   →   Accept / Clamp / Reject every proposed action",
                  font_size=11, italic=True, color=DARK_GRAY)


# ============================================================
# SLIDE 17 -- Breakdown #1: UE side
# ============================================================
def slide_breakdown_ue():
    s = add_blank_slide()
    add_title_bar(s, "Breakdown · UE-side Split-Control RL Agent",
                   "Per-UE policy that picks compression actions from current state")

    # Left: state inputs
    add_text_box(s, Inches(0.5), Inches(1.55), Inches(6.0), Inches(0.5),
                 "State input  s_t", font_size=18, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(2.1), Inches(6.0), Inches(4.7),
                [
                    ("Scene  x_scene", "density, foreground fraction, occlusion ratio, scale distribution."),
                    ("Model  x_model", "head confidence, per-class uncertainty, predicted task risk."),
                    ("Network  x_net", "RTT, packet loss, throughput estimate, queue delay, scheduling grants."),
                    ("Context  q_t", "freshness, previous action context, time since last accepted frame."),
                ], size=13)

    # Right: action outputs
    add_text_box(s, Inches(7.0), Inches(1.55), Inches(6.0), Inches(0.5),
                 "Action output  a_t", font_size=18, bold=True, color=NAVY)
    add_bullets(s, Inches(7.0), Inches(2.1), Inches(6.0), Inches(2.9),
                [
                    "AE channels  (128 / 64 / 32)",
                    "ROI threshold  (0.1 / 0.3 / 0.5)",
                    "Quantization  (8 / 6 / 4 bit)",
                    "Frame send / skip",
                    "Redundancy add / drop  (FEC, duplication)",
                ], size=14)

    # Bottom card: reward
    rb = add_rect(s, Inches(0.5), Inches(5.55), Inches(12.4), Inches(1.5),
                   fill=SOFT_BG, line=ID_BLUE, line_width=2)
    add_text_box(s, Inches(0.7), Inches(5.6), Inches(12.0), Inches(0.4),
                  "Reward  R_t", font_size=14, bold=True, color=NAVY)
    add_text_box(s, Inches(0.7), Inches(5.95), Inches(12.0), Inches(1.1),
                  "R_t = task_utility (AP, mIoU, foreground IoU, recall)  −  λ_b · bytes  −  λ_l · latency  −  λ_p · loss        (subject to guardrails)",
                  font_size=14, color=DARK_GRAY)


# ============================================================
# SLIDE 18 -- Breakdown #2: Edge AI server + shared spatial map
# ============================================================
def slide_breakdown_server():
    s = add_blank_slide()
    add_title_bar(s, "Breakdown · Edge AI server + Shared Spatial Map",
                   "Where features become tagged, fused, and persisted")

    # LEFT: server pipeline
    add_text_box(s, Inches(0.5), Inches(1.55), Inches(6.0), Inches(0.5),
                 "Edge AI server", font_size=18, bold=True, color=NAVY)
    # Stack
    stages = [
        ("De-quant / decompress",
         "Reverse the per-UE compression with the agreed-upon settings.",
         WHITE, DARK_GRAY),
        ("OD head", "Faster R-CNN → boxes, classes, scores.",
         ID_BLUE, WHITE),
        ("SEG head", "LR-ASPP → per-pixel mask of classes.",
         ID_BLUE, WHITE),
        ("Confidence / uncertainty",
         "Calibrated per-detection and per-region confidence.",
         PURPLE, WHITE),
        ("Per-UE assembly",
         "Tag outputs with provenance, UE-id, timestamp, pose.",
         ORANGE, WHITE),
    ]
    y = Inches(2.1)
    for stage in stages:
        head, body, fill, fg = stage
        box = add_rect(s, Inches(0.5), y, Inches(6.0), Inches(0.85),
                        fill=fill, line=NAVY, line_width=1.0)
        add_text_box(s, Inches(0.7), y + Inches(0.05),
                      Inches(5.7), Inches(0.4),
                      head, font_size=13, bold=True, color=fg if fill != WHITE else NAVY)
        add_text_box(s, Inches(0.7), y + Inches(0.42),
                      Inches(5.7), Inches(0.4),
                      body, font_size=11, color=fg if fill != WHITE else DARK_GRAY)
        y += Inches(0.95)

    # RIGHT: shared spatial map
    add_text_box(s, Inches(7.0), Inches(1.55), Inches(6.0), Inches(0.5),
                 "Shared spatial map", font_size=18, bold=True, color=NAVY)
    map_box = add_rect(s, Inches(7.0), Inches(2.1), Inches(6.0),
                        Inches(4.85),
                        fill=SOFT_BG, line=ORANGE, line_width=2)
    fields = [
        ("class", "vehicle, pedestrian, cyclist, etc."),
        ("pose", "(x, y, yaw) in a shared coordinate frame"),
        ("velocity", "estimated or sourced from upstream tracker"),
        ("confidence", "per-object posterior, calibrated"),
        ("provenance", "which UE / sensor produced this entry"),
        ("freshness", "time since last update"),
        ("occlusion state", "visible / partially / fully occluded"),
    ]
    yy = Inches(2.3)
    for k, v in fields:
        chip = add_rect(s, Inches(7.2), yy, Inches(1.7), Inches(0.5),
                        fill=ORANGE, line=ORANGE,
                        shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        set_shape_text(chip, k, size=11, color=WHITE)
        add_text_box(s, Inches(9.0), yy + Inches(0.05),
                      Inches(4.0), Inches(0.5),
                      v, font_size=11, color=DARK_GRAY,
                      anchor=MSO_ANCHOR.MIDDLE)
        yy += Inches(0.6)
    add_text_box(s, Inches(7.2), Inches(6.6), Inches(5.6), Inches(0.4),
                 "The map is the contract between perception and downstream consumers.",
                 font_size=11, italic=True, color=NAVY)


# ============================================================
# SLIDE 19 -- Breakdown #3: Map-sharing agent + guardrails
# ============================================================
def slide_breakdown_mapsharing():
    s = add_blank_slide()
    add_title_bar(s, "Breakdown · Map-sharing RL agent and guardrails",
                   "Decide WHAT to push to which vehicle, WHEN, and at what DETAIL")

    # Left: map-sharing agent
    add_text_box(s, Inches(0.5), Inches(1.55), Inches(6.0), Inches(0.5),
                 "Map-sharing RL agent", font_size=18, bold=True,
                 color=NAVY)
    add_text_box(s, Inches(0.5), Inches(2.05), Inches(6.0), Inches(0.4),
                 "State  z_t", font_size=14, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(2.45), Inches(6.0), Inches(1.7),
                [
                    "map risk (collision-relevant objects)",
                    "object freshness",
                    "vehicle trajectory / heading",
                    "network load on the downlink",
                ], size=13)
    add_text_box(s, Inches(0.5), Inches(4.15), Inches(6.0), Inches(0.4),
                 "Action  u_t", font_size=14, bold=True, color=NAVY)
    add_bullets(s, Inches(0.5), Inches(4.55), Inches(6.0), Inches(1.7),
                [
                    "what to share (which objects)",
                    "when to share (event / scheduled)",
                    "who receives (specific UEs / broadcast)",
                    "detail level (compact / full record)",
                ], size=13)
    rbox = add_rect(s, Inches(0.5), Inches(6.1), Inches(6.0),
                     Inches(0.85), fill=SOFT_BG, line=ORANGE, line_width=1.5)
    add_text_box(s, Inches(0.7), Inches(6.15), Inches(5.6), Inches(0.75),
                  "R_t = task_utility  −  λ_b · bytes  −  λ_l · latency  −  λ_s · stale_risk",
                  font_size=13, color=DARK_GRAY,
                  anchor=MSO_ANCHOR.MIDDLE)

    # Right: guardrails
    add_text_box(s, Inches(7.0), Inches(1.55), Inches(6.0), Inches(0.5),
                 "Task-precision guardrails", font_size=18, bold=True,
                 color=RED)
    # Pipeline
    add_text_box(s, Inches(7.0), Inches(2.1), Inches(6.0), Inches(0.4),
                 "Treat the RL policy as a PROPOSER, not as the final actor.",
                 font_size=12, italic=True, color=DARK_GRAY)
    # 3-step pipeline boxes
    pipe = [
        ("Proposed action", ID_BLUE),
        ("Guardrail check", RED),
        ("Accept / Clamp / Reject", GREEN),
    ]
    yy = Inches(2.6)
    for i, (lbl, col) in enumerate(pipe):
        box = add_rect(s, Inches(7.0), yy, Inches(6.0), Inches(0.75),
                       fill=col, line=col)
        set_shape_text(box, lbl, size=14, bold=True, color=WHITE)
        if i < len(pipe) - 1:
            add_arrow(s, Inches(10.0), yy + Inches(0.75),
                      Inches(10.0), yy + Inches(0.95),
                      color=DARK_GRAY)
        yy += Inches(0.95)
    # Constraints summary
    add_text_box(s, Inches(7.0), Inches(5.45), Inches(6.0), Inches(0.4),
                 "Constraints (must hold)", font_size=14, bold=True,
                 color=RED)
    add_bullets(s, Inches(7.0), Inches(5.85), Inches(6.0), Inches(1.7),
                [
                    "AP drop ≤ ε_OD",
                    "mIoU drop ≤ ε_SEG",
                    "Recall_pedestrian / cyclist ≥ τ",
                    "End-to-end latency ≤ L_max",
                ], size=13)


# ============================================================
# SLIDE 20 -- Evaluation
# ============================================================
def slide_evaluation():
    s = add_blank_slide()
    add_title_bar(s, "Evaluation — four CARLA campaigns",
                   "Two to validate the controller, two to validate the spatial-map / override loop")

    campaigns = [
        ("A. Static knobs vs learned controller",
         "Single ego, OD + SEG split routes, repeatable network stress profiles.",
         "Task utility at lower byte / latency / loss vs the best static knob choice.",
         ID_BLUE),
        ("B. Guardrail stress test",
         "Crowded / sparse / jittery / lossy / queued scenarios.",
         "No accepted policy ever violates AP / mIoU / pedestrian / cyclist recall thresholds.",
         ORANGE),
        ("C. Physical-AI map update",
         "Occluded-object intersections; split-model outputs feed the shared spatial map.",
         "Map freshness and localization are good enough to support AV risk assessment.",
         GREEN),
        ("D. Navigation override demo",
         "AV approaches an occluded course-conflict object revealed via learned map sharing.",
         "Vehicle triggers a safe override; collision avoided across the scenario battery.",
         RED),
    ]
    x = Inches(0.4)
    y = Inches(1.5)
    w = Inches(6.25)
    h = Inches(2.65)
    gx = Inches(0.15)
    gy = Inches(0.15)
    for i, (head, setup, metric, col) in enumerate(campaigns):
        r = i // 2
        c = i % 2
        left = x + (w + gx) * c
        top = y + (h + gy) * r
        card = add_rect(s, left, top, w, h, fill=WHITE, line=col, line_width=1.5)
        band = add_rect(s, left, top, w, Inches(0.55),
                        fill=col, line=col, shape=MSO_SHAPE.RECTANGLE)
        set_shape_text(band, head, size=14, bold=True, color=WHITE)
        add_text_box(s, left + Inches(0.15), top + Inches(0.65),
                     w - Inches(0.3), Inches(0.5),
                     "Setup", font_size=11, bold=True, color=NAVY)
        add_text_box(s, left + Inches(0.15), top + Inches(1.0),
                     w - Inches(0.3), Inches(0.7),
                     setup, font_size=12, color=DARK_GRAY)
        add_text_box(s, left + Inches(0.15), top + Inches(1.65),
                     w - Inches(0.3), Inches(0.4),
                     "Headline metric", font_size=11, bold=True, color=NAVY)
        add_text_box(s, left + Inches(0.15), top + Inches(2.0),
                     w - Inches(0.3), Inches(0.6),
                     metric, font_size=12, color=DARK_GRAY)


# ============================================================
# SLIDE 21 -- Timeline
# ============================================================
def slide_timeline():
    s = add_blank_slide()
    add_title_bar(s, "Six-month plan (May 2026 – November 2026)",
                   "Months 1–3 prove controlled split inference; Months 4–6 close the spatial-map loop")

    months = [
        ("Month 1",
         "Reproduce split-inference baselines and logging in CARLA.\nFreeze RL state / action / reward schema.",
         "Repeatable OD / SEG traces with bytes, latency, loss, AP / mIoU, foreground IoU.",
         ID_BLUE),
        ("Month 2",
         "Implement constrained RL controller over AE / ROI / quantization / scheduling / redundancy.",
         "Policy trains / evaluates against static policies on the same logged metrics.",
         ID_BLUE),
        ("Month 3",
         "Evaluate task-precision guardrails under network stress.",
         "Plots show where learned control beats static knobs without guardrail violations.",
         ID_BLUE),
        ("Month 4",
         "Add spatial-map ingestion from split-model outputs.\nCARLA ground truth for validation.",
         "Map stores class, pose, velocity, confidence, provenance, freshness, occlusion.",
         ORANGE),
        ("Month 5",
         "Train / evaluate learned map-sharing policies for occluded-object updates.",
         "Policy ranks what / when / how to share under bandwidth and freshness constraints.",
         ORANGE),
        ("Month 6",
         "Intersection collision-avoidance override demo.\nPaper / figures / disclosure prep.",
         "End-to-end CARLA demo, paper outline, invention-disclosure candidate notes.",
         ORANGE),
    ]
    # Horizontal timeline track
    track_y = Inches(1.6)
    track = add_rect(s, Inches(0.5), track_y + Inches(0.25), Inches(12.35),
                     Inches(0.1), fill=LIGHT_GRAY, line=LIGHT_GRAY,
                     shape=MSO_SHAPE.RECTANGLE)

    n = len(months)
    seg_w = Inches(12.35 / n)
    for i, (label, work, exit_, col) in enumerate(months):
        cx = Inches(0.5) + seg_w * i + seg_w / 2 - Inches(0.25)
        # Marker dot
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, cx, track_y + Inches(0.05),
                                  Inches(0.5), Inches(0.5))
        dot.fill.solid()
        dot.fill.fore_color.rgb = col
        dot.line.color.rgb = WHITE
        dot.line.width = Pt(2)
        set_shape_text(dot, str(i + 1), size=14, bold=True, color=WHITE)

        # Card under the marker
        card_x = Inches(0.5) + seg_w * i + Inches(0.1)
        card_w = seg_w - Inches(0.2)
        card_y = Inches(2.4)
        card_h = Inches(4.5)
        card = add_rect(s, card_x, card_y, card_w, card_h,
                        fill=WHITE, line=col, line_width=1.5)
        # Header
        band = add_rect(s, card_x, card_y, card_w, Inches(0.5),
                        fill=col, line=col, shape=MSO_SHAPE.RECTANGLE)
        set_shape_text(band, label, size=13, bold=True, color=WHITE)
        # Work
        add_text_box(s, card_x + Inches(0.1), card_y + Inches(0.6),
                     card_w - Inches(0.2), Inches(0.35),
                     "Work", font_size=10, bold=True, color=NAVY)
        add_text_box(s, card_x + Inches(0.1), card_y + Inches(0.95),
                     card_w - Inches(0.2), Inches(1.8),
                     work, font_size=10, color=DARK_GRAY)
        # Exit
        add_text_box(s, card_x + Inches(0.1), card_y + Inches(2.7),
                     card_w - Inches(0.2), Inches(0.35),
                     "Exit criterion", font_size=10, bold=True, color=NAVY)
        add_text_box(s, card_x + Inches(0.1), card_y + Inches(3.05),
                     card_w - Inches(0.2), Inches(1.4),
                     exit_, font_size=10, color=DARK_GRAY)


# ============================================================
# SLIDE 22 -- Risks, open questions, and next steps
# ============================================================
def slide_risks():
    s = add_blank_slide()
    add_title_bar(s, "Risks, open questions, and what I am doing next",
                   "Where the proposal could fail — and how we will know early")

    # Three columns
    cols = [
        ("Risks", [
            ("Controller learns to minimize bytes only.",
             "Hard / soft guardrails on AP, mIoU, foreground IoU, safety-class recall."),
            ("Action space too broad to train.",
             "Start with AE / ROI / quant only; add scheduling and redundancy after stable."),
            ("Aggregate mIoU hides safety failures.",
             "Track foreground IoU and per-class recall (pedestrian, cyclist, small)."),
            ("Map scope expands too far.",
             "Months 4–6 limited to ingestion + sharing + one override scenario."),
            ("OAI / SionnaRT / PC5 integration consumes time.",
             "Trace-driven stress first; OAI live as follow-on validation."),
        ], RED),
        ("Open questions", [
            ("Coordination scope.",
             "Per-UE local first; decide later whether to coordinate across UEs."),
            ("Importance metric.",
             "Is L2 saliency enough, or do we need a learned / task-conditioned signal?"),
            ("Action latency.",
             "Can the policy itself meet the safety-critical decision window?"),
            ("Scaling.",
             "Where does fusion / radio resource saturate as N UEs grows?"),
        ], ORANGE),
        ("Next 4 weeks", [
            ("Lock the logging schema.",
             "Application + network CSVs aligned by run_group; already in flight."),
            ("Run static-knob baseline curves.",
             "Payload vs AP / mIoU at varying ROI / AE / quantization on OAI multi-UE."),
            ("Draft RL state/action/reward schema.",
             "Concrete feature list per UE, action discretization, reward shape."),
            ("Start CARLA occlusion scenario harness.",
             "Repeatable curbside-parked-vehicle pedestrian occlusion for campaigns B/D."),
        ], GREEN),
    ]
    x = Inches(0.4)
    y = Inches(1.45)
    cw = Inches(4.18)
    gap = Inches(0.13)
    h = Inches(5.6)
    for i, (head, items, col) in enumerate(cols):
        left = x + (cw + gap) * i
        card = add_rect(s, left, y, cw, h, fill=WHITE, line=col, line_width=1.5)
        band = add_rect(s, left, y, cw, Inches(0.55),
                        fill=col, line=col, shape=MSO_SHAPE.RECTANGLE)
        set_shape_text(band, head, size=16, bold=True, color=WHITE)
        # Items
        item_y = y + Inches(0.7)
        for (hd, body) in items:
            t1 = add_text_box(s, left + Inches(0.2), item_y,
                              cw - Inches(0.4), Inches(0.4),
                              hd, font_size=12, bold=True, color=NAVY)
            t2 = add_text_box(s, left + Inches(0.2), item_y + Inches(0.35),
                              cw - Inches(0.4), Inches(0.85),
                              body, font_size=11, color=DARK_GRAY)
            item_y += Inches(1.05)


# ============================================================
# Build all slides
# ============================================================
def main():
    slide_title()
    slide_position()
    slide_problem_occlusion()
    slide_coop_perception_concept()
    slide_use_cases()
    slide_sharing_options()
    slide_split_what()
    slide_split_why()
    slide_split_methodology()
    slide_compression_knobs()
    slide_measurements()
    slide_bridge()
    slide_scenesense_idea()
    slide_hypothesis()
    slide_things_to_consider()
    slide_big_picture()
    slide_breakdown_ue()
    slide_breakdown_server()
    slide_breakdown_mapsharing()
    slide_evaluation()
    slide_timeline()
    slide_risks()

    out_path = "SceneSense_Agent_Intro_Deck.pptx"
    prs.save(out_path)
    print(f"Wrote {out_path} with {len(prs.slides)} slides.")


if __name__ == "__main__":
    main()
