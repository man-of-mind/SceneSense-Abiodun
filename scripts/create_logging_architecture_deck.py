#!/usr/bin/env python3
"""Create the SceneSense logging architecture milestone PowerPoint deck.

This intentionally avoids external Python dependencies. It writes a simple
PowerPoint OOXML package directly with standard library modules.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence
from xml.sax.saxutils import escape


ABIODUN_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = ABIODUN_DIR / "SceneSense_Logging_Architecture_Milestone.pptx"

SLIDE_W = 13.333
SLIDE_H = 7.5
EMU_PER_IN = 914400

COLORS = {
    "navy": "14324A",
    "teal": "008C95",
    "green": "2E7D32",
    "orange": "E87722",
    "red": "B23A48",
    "purple": "6A4C93",
    "blue": "2F6FAD",
    "gray": "F3F6F8",
    "midgray": "D7DEE5",
    "dark": "1F2933",
    "muted": "5B6770",
    "white": "FFFFFF",
    "black": "000000",
}


def emu(value: float) -> int:
    return int(round(value * EMU_PER_IN))


def xml_text(value: object) -> str:
    return escape(str(value), {"\"": "&quot;", "'": "&apos;"})


def content_types(slide_count: int) -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  {slide_overrides}
</Types>'''


ROOT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''


def app_xml(slide_count: int) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>SceneSense Deck Generator</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Company>SceneSense</Company>
</Properties>'''


CORE_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>SceneSense Logging Architecture Milestone</dc:title>
  <dc:creator>SceneSense Project</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dc:description>Application, UE, and gNB logging pipeline for SceneSense over OAI 5G.</dc:description>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-05-28T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-05-28T00:00:00Z</dcterms:modified>
</cp:coreProperties>'''


def presentation_xml(slide_count: int) -> str:
    sld_ids = "\n".join(
        f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1, slide_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{emu(SLIDE_W)}" cy="{emu(SLIDE_H)}" type="wide"/>
  <p:notesSz cx="{emu(7.5)}" cy="{emu(10)}"/>
  <p:defaultTextStyle/>
</p:presentation>'''


def presentation_rels(slide_count: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for i in range(1, slide_count + 1):
        rels.append(
            f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {"".join(rels)}
</Relationships>'''


THEME_XML = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="SceneSense">
  <a:themeElements>
    <a:clrScheme name="SceneSense">
      <a:dk1><a:srgbClr val="{COLORS['dark']}"/></a:dk1>
      <a:lt1><a:srgbClr val="{COLORS['white']}"/></a:lt1>
      <a:dk2><a:srgbClr val="{COLORS['navy']}"/></a:dk2>
      <a:lt2><a:srgbClr val="{COLORS['gray']}"/></a:lt2>
      <a:accent1><a:srgbClr val="{COLORS['teal']}"/></a:accent1>
      <a:accent2><a:srgbClr val="{COLORS['orange']}"/></a:accent2>
      <a:accent3><a:srgbClr val="{COLORS['green']}"/></a:accent3>
      <a:accent4><a:srgbClr val="{COLORS['blue']}"/></a:accent4>
      <a:accent5><a:srgbClr val="{COLORS['purple']}"/></a:accent5>
      <a:accent6><a:srgbClr val="{COLORS['red']}"/></a:accent6>
      <a:hlink><a:srgbClr val="{COLORS['blue']}"/></a:hlink>
      <a:folHlink><a:srgbClr val="{COLORS['purple']}"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="SceneSense">
      <a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
      <a:minorFont><a:latin typeface="Aptos"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="SceneSense">
      <a:fillStyleLst>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
        <a:gradFill rotWithShape="1"><a:gsLst><a:gs pos="0"><a:schemeClr val="phClr"/></a:gs><a:gs pos="100000"><a:schemeClr val="phClr"><a:lumMod val="85000"/></a:schemeClr></a:gs></a:gsLst><a:lin ang="5400000" scaled="0"/></a:gradFill>
        <a:solidFill><a:schemeClr val="phClr"/></a:solidFill>
      </a:fillStyleLst>
      <a:lnStyleLst>
        <a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
        <a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
        <a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>
      </a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>'''


MASTER_XML = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{COLORS['white']}"/></a:solidFill></p:bgPr></p:bg>
    <p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>'''


MASTER_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>'''


LAYOUT_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>'''


LAYOUT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>'''


class Slide:
    def __init__(self, title: str | None = None, kicker: str | None = None):
        self.shapes: List[str] = []
        self.next_id = 2
        self.add_rect(0, 0, SLIDE_W, SLIDE_H, COLORS["white"], line=None)
        if kicker:
            self.add_text(0.55, 0.22, 6.5, 0.25, kicker.upper(), font=8, color=COLORS["teal"], bold=True, tracking=True)
        if title:
            self.add_text(0.55, 0.45, 12.1, 0.55, title, font=27, color=COLORS["navy"], bold=True)
            self.add_rect(0.55, 1.08, 1.25, 0.05, COLORS["orange"], line=None)

    def _shape_id(self) -> int:
        sid = self.next_id
        self.next_id += 1
        return sid

    def add_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        line: str | None = COLORS["midgray"],
        radius: bool = False,
    ) -> None:
        sid = self._shape_id()
        line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        prst = "roundRect" if radius else "rect"
        self.shapes.append(f'''
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}</p:spPr></p:sp>''')

    def add_arrow(self, x: float, y: float, w: float, h: float, fill: str = COLORS["orange"]) -> None:
        sid = self._shape_id()
        self.shapes.append(f'''
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Arrow {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rightArrow"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>''')

    def add_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        font: int = 16,
        color: str = COLORS["dark"],
        bold: bool = False,
        align: str = "l",
        fill: str | None = None,
        line: str | None = None,
        radius: bool = False,
        margin: float = 0.08,
        tracking: bool = False,
    ) -> None:
        sid = self._shape_id()
        fill_xml = "<a:noFill/>" if fill is None else f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        prst = "roundRect" if radius else "rect"
        paragraphs = []
        for raw_line in text.split("\n"):
            line_text = raw_line.rstrip()
            if not line_text:
                paragraphs.append('<a:p/>')
                continue
            ppr = f'<a:pPr algn="{align}"/>'
            kern = ' spc="500"' if tracking else ""
            bold_attr = ' b="1"' if bold else ""
            rpr = f'<a:rPr lang="en-US" sz="{font * 100}"{bold_attr}{kern}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr>'
            paragraphs.append(f'<a:p>{ppr}<a:r>{rpr}<a:t>{xml_text(line_text)}</a:t></a:r></a:p>')
        tx = "".join(paragraphs)
        self.shapes.append(f'''
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
<p:txBody><a:bodyPr wrap="square" lIns="{emu(margin)}" rIns="{emu(margin)}" tIns="{emu(margin)}" bIns="{emu(margin)}"><a:spAutoFit/></a:bodyPr><a:lstStyle/>{tx}</p:txBody></p:sp>''')

    def add_footer(self, slide_num: int) -> None:
        self.add_text(0.55, 7.12, 4.0, 0.2, "SceneSense logging milestone", font=7, color=COLORS["muted"])
        self.add_text(12.2, 7.12, 0.55, 0.2, str(slide_num), font=7, color=COLORS["muted"], align="r")

    def xml(self, slide_num: int) -> str:
        self.add_footer(slide_num)
        shapes = "\n".join(self.shapes)
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {shapes}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def slide_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>'''


def card(slide: Slide, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    slide.add_text(x, y, w, h, f"{title}\n{body}", font=13, color=COLORS["dark"], fill="FFFFFF", line=color, radius=True, margin=0.12)
    slide.add_rect(x, y, 0.08, h, color, line=None)


def metric_card(slide: Slide, x: float, y: float, w: float, h: float, label: str, value: str, note: str, color: str) -> None:
    slide.add_text(x, y, w, h, f"{value}\n{label}\n{note}", font=12, color=COLORS["dark"], fill=COLORS["gray"], line=color, radius=True, align="ctr", margin=0.08)


def build_slides() -> List[Slide]:
    slides: List[Slide] = []

    s = Slide()
    s.add_text(0.65, 0.7, 11.8, 0.7, "SceneSense Logging Architecture", font=34, color=COLORS["navy"], bold=True)
    s.add_rect(0.65, 1.55, 1.55, 0.06, COLORS["orange"], line=None)
    s.add_text(0.65, 1.82, 10.8, 0.55, "Validated application, UE-side, and gNB-side telemetry for OAI 5G split perception", font=18, color=COLORS["dark"])
    s.add_text(0.65, 2.75, 3.0, 0.55, "Application metrics", font=15, color=COLORS["white"], bold=True, fill=COLORS["teal"], radius=True, align="ctr")
    s.add_text(3.95, 2.75, 3.0, 0.55, "UE decoded grants", font=15, color=COLORS["white"], bold=True, fill=COLORS["orange"], radius=True, align="ctr")
    s.add_text(7.25, 2.75, 3.0, 0.55, "gNB RAN metrics", font=15, color=COLORS["white"], bold=True, fill=COLORS["green"], radius=True, align="ctr")
    s.add_text(1.05, 4.0, 10.9, 1.25, "Milestone: repeatable logging pipeline over OAI 5G that can feed both research analysis and future RL agents.", font=24, color=COLORS["navy"], bold=True, align="ctr")
    s.add_text(8.9, 6.55, 3.5, 0.28, "Validated run: exp07_full_logging_validation", font=10, color=COLORS["muted"], align="r")
    slides.append(s)

    s = Slide("Milestone Summary", "What changed")
    card(s, 0.7, 1.45, 3.75, 4.25, "Built", "- OAI 5G multi-UE fusion pipeline\n- UE NR grant T-tracer event\n- gNB T-tracer/stdout parsers\n- Analysis + validation helpers", COLORS["teal"])
    card(s, 4.8, 1.45, 3.75, 4.25, "Validated", "- UE grant TBS vs payload bits\n- UE-vs-gNB MAC/PHY totals\n- App goodput vs scheduled capacity\n- gNB SNR/RSRP/BLER/HARQ summaries", COLORS["green"])
    card(s, 8.9, 1.45, 3.75, 4.25, "Why It Matters", "- Creates trusted network state\n- Separates app, UE, and RAN views\n- Enables reproducible experiments\n- Provides RL-ready features", COLORS["orange"])
    slides.append(s)

    s = Slide("End-to-End Logging Architecture", "System view")
    s.add_text(0.7, 1.35, 2.2, 0.75, "Pole / UE 1\nfront half\n10.0.0.2", font=13, color=COLORS["white"], fill=COLORS["teal"], radius=True, align="ctr")
    s.add_text(0.7, 3.25, 2.2, 0.75, "Pole / UE 2\nfront half\n10.0.0.3", font=13, color=COLORS["white"], fill=COLORS["teal"], radius=True, align="ctr")
    s.add_arrow(3.1, 1.55, 1.0, 0.35, COLORS["orange"])
    s.add_arrow(3.1, 3.45, 1.0, 0.35, COLORS["orange"])
    s.add_text(4.25, 1.95, 2.2, 1.25, "OAI 5G\nRAN + Core\nmulti-UE", font=16, color=COLORS["white"], fill=COLORS["navy"], radius=True, align="ctr")
    s.add_arrow(6.65, 2.4, 1.0, 0.35, COLORS["orange"])
    s.add_text(7.85, 1.45, 2.2, 0.8, "Back-half\nfusion container\n192.168.70.140", font=13, color=COLORS["white"], fill=COLORS["green"], radius=True, align="ctr")
    s.add_text(7.85, 3.1, 2.2, 0.8, "Spatial-map\nserver", font=13, color=COLORS["white"], fill=COLORS["purple"], radius=True, align="ctr")
    s.add_arrow(8.45, 2.35, 0.6, 0.45, COLORS["purple"])
    s.add_text(0.8, 5.25, 2.7, 0.85, "App CSVs\npayload, RTT,\ntimeouts, FPS", font=12, color=COLORS["dark"], fill=COLORS["gray"], line=COLORS["teal"], radius=True, align="ctr")
    s.add_text(4.1, 5.25, 2.7, 0.85, "UE T-tracer\nNRUE_MAC_DCI_GRANT", font=12, color=COLORS["dark"], fill=COLORS["gray"], line=COLORS["orange"], radius=True, align="ctr")
    s.add_text(7.4, 5.25, 2.7, 0.85, "gNB T-tracer\nMAC/PHY/RLC/PDCP", font=12, color=COLORS["dark"], fill=COLORS["gray"], line=COLORS["green"], radius=True, align="ctr")
    s.add_text(10.7, 5.25, 1.8, 0.85, "stdout\nSNR/BLER", font=12, color=COLORS["dark"], fill=COLORS["gray"], line=COLORS["blue"], radius=True, align="ctr")
    slides.append(s)

    s = Slide("Three Metric Planes", "Mental model")
    card(s, 0.65, 1.45, 3.8, 4.5, "App-Centric", "- What did perception experience?\n- Feature bytes\n- Round-trip latency\n- Timeout / receive rate\n- Approximate FPS\n- Task quality later", COLORS["teal"])
    card(s, 4.75, 1.45, 3.8, 4.5, "UE-Centric", "- What did the UE observe locally?\n- Decoded NR grants\n- MCS, RBs, symbols, TBS\n- HARQ / NDI / RV\n- Scheduled bitrate windows", COLORS["orange"])
    card(s, 8.85, 1.45, 3.8, 4.5, "gNB-Centric", "- What did the network schedule/receive?\n- MAC/PHY TBS + MCS\n- PRBs and SNR-like stats\n- BLER / HARQ / DTX\n- RLC/PDCP/LCID bytes", COLORS["green"])
    slides.append(s)

    s = Slide("Application-Centric Metrics", "Perception view")
    metric_card(s, 0.7, 1.45, 2.65, 1.25, "fusion_tl_14 receive rate", "93.0%", "timeout 7.0%", COLORS["teal"])
    metric_card(s, 3.65, 1.45, 2.65, 1.25, "fusion_tl_14 RTT", "249 ms", "p95 378 ms", COLORS["teal"])
    metric_card(s, 6.6, 1.45, 2.65, 1.25, "fusion_tl_14 goodput", "19.08 Mbps", "feature payload", COLORS["teal"])
    metric_card(s, 9.55, 1.45, 2.65, 1.25, "view_2 goodput", "16.06 Mbps", "feature payload", COLORS["teal"])
    card(s, 0.9, 3.35, 5.35, 2.3, "Core Formulas", "feature_goodput = 8 * sum(feature_payload_bytes) / duration\nreceive_rate = received_frames / total_frames\ntimeout_rate = missed_results / total_frames", COLORS["blue"])
    card(s, 6.75, 3.35, 5.35, 2.3, "Why This Plane Matters", "- Measures user-facing perception performance\n- Gives RL latency and payload pressure\n- Later joins with mIoU, recall, localization error\n- Shows if network decisions protect task utility", COLORS["orange"])
    slides.append(s)

    s = Slide("UE-Centric Metrics", "Decoded NR grant state")
    card(s, 0.65, 1.35, 4.0, 4.8, "Local UE Observation", "- direction: UL or DL\n- rnti and DCI format\n- mcs and mcs_table\n- rb_start and rb_size\n- start_symbol and nr_symbols\n- tbs, harq_pid, ndi, rv, round\n- qam_mod_order and target_code_rate", COLORS["orange"])
    card(s, 4.95, 1.35, 3.7, 4.8, "Derived Features", "TBS_bytes_norm:\n  UL = tbs\n  DL = tbs / 8\n\nscheduled_mbps = 8 * sum(TBS_bytes_norm) / Delta_t / 1e6\n\nretx_rate = count(round > 0 or rv > 0) / grants", COLORS["blue"])
    card(s, 8.95, 1.35, 3.6, 4.8, "Important Wording", "- This is UE-visible scheduled state\n- It is not raw SNR/CQI\n- MCS acts as link-adaptation proxy\n- Raw UE CSI/CQI can be added later if needed", COLORS["purple"])
    slides.append(s)

    s = Slide("gNB-Centric Metrics", "Network scheduler + validation view")
    card(s, 0.7, 1.35, 3.0, 4.75, "Scheduler", "- GNB_MAC_UL/DL\n- MCS and TBS\n- Per-RNTI scheduling\n- PRB-related fields", COLORS["green"])
    card(s, 3.95, 1.35, 3.0, 4.75, "PHY / Power", "- GNB_PHY_UL_PAYLOAD_RX_BITS\n- PUSCH power control\n- PUCCH power control\n- SNR-like fields", COLORS["blue"])
    card(s, 7.2, 1.35, 2.65, 4.75, "Upper Layers", "- LCID UL/DL bytes\n- RLC UL/DL\n- RLC-MAC\n- PDCP UL/DL", COLORS["orange"])
    card(s, 10.1, 1.35, 2.55, 4.75, "stdout", "- RSRP\n- SNR-like values\n- BLER\n- HARQ rounds/errors\n- DTX\n- MAC byte totals", COLORS["purple"])
    slides.append(s)

    s = Slide("Validation Chain", "How we know the numbers are sane")
    s.add_text(0.7, 1.35, 2.8, 0.85, "UE grant event\nNRUE_MAC_DCI_GRANT", font=13, color=COLORS["white"], fill=COLORS["orange"], radius=True, align="ctr")
    s.add_arrow(3.65, 1.55, 0.85, 0.35, COLORS["orange"])
    s.add_text(4.65, 1.35, 2.8, 0.85, "UE payload trace\nUE_PHY_UL_PAYLOAD_TX_BITS", font=13, color=COLORS["white"], fill=COLORS["blue"], radius=True, align="ctr")
    s.add_arrow(7.6, 1.55, 0.85, 0.35, COLORS["orange"])
    s.add_text(8.6, 1.35, 3.05, 0.85, "Exact check\nTBS * 8 == payload bits", font=13, color=COLORS["white"], fill=COLORS["green"], radius=True, align="ctr")
    s.add_text(0.7, 3.05, 2.8, 0.85, "UE grant totals\nper RNTI/window", font=13, color=COLORS["white"], fill=COLORS["orange"], radius=True, align="ctr")
    s.add_arrow(3.65, 3.25, 0.85, 0.35, COLORS["orange"])
    s.add_text(4.65, 3.05, 2.8, 0.85, "gNB MAC/PHY\nscheduler totals", font=13, color=COLORS["white"], fill=COLORS["green"], radius=True, align="ctr")
    s.add_arrow(7.6, 3.25, 0.85, 0.35, COLORS["orange"])
    s.add_text(8.6, 3.05, 3.05, 0.85, "Cross-check\nratios near 1.0", font=13, color=COLORS["white"], fill=COLORS["green"], radius=True, align="ctr")
    card(s, 1.0, 5.0, 10.7, 0.95, "Key Insight", "The UE-side state is not guessed from IP throughput. It is decoded from NR grants, validated against existing OAI payload traces, then cross-checked against gNB MAC/PHY scheduling.", COLORS["teal"])
    slides.append(s)

    s = Slide("Validated Run: exp07_full_logging_validation", "Results")
    metric_card(s, 0.7, 1.35, 2.9, 1.15, "UE grant vs payload", "1.000", "both RNTIs", COLORS["green"])
    metric_card(s, 3.9, 1.35, 2.9, 1.15, "UL UE/gNB ratio", "0.994 / 0.981", "RNTI 0x0847 / 0xf107", COLORS["green"])
    metric_card(s, 7.1, 1.35, 2.9, 1.15, "DL UE/gNB ratio", "0.993 / 0.971", "after unit normalization", COLORS["green"])
    metric_card(s, 10.3, 1.35, 2.35, 1.15, "Retx rate", "0.0", "clean run", COLORS["green"])
    card(s, 0.7, 3.0, 5.7, 2.7, "UE Grant Summary", "0x0847 UL: 19.71 Mbps, avg MCS 13.07, avg RBs 100.64\n0xf107 UL: 17.39 Mbps, avg MCS 13.83, avg RBs 99.58\nDL is small: roughly 0.18-0.22 Mbps scheduled", COLORS["orange"])
    card(s, 6.9, 3.0, 5.7, 2.7, "gNB stdout Summary", "RNTI 0x0847 -> CU-UE-ID 1\nRNTI 0xf107 -> CU-UE-ID 2\nAvg UL SNR-like values around 18 dB\nBLER trends near zero after warm-up\nRSRP stable around -43 dBm", COLORS["blue"])
    slides.append(s)

    s = Slide("Artifacts and Reproducibility", "How to rerun")
    card(s, 0.65, 1.35, 3.85, 4.8, "Run Group", "Use one label everywhere:\n\nexp07_full_logging_validation\n\nfront halves, tunnel sampler, UE T-tracer, gNB T-tracer, and analysis all join on run_group.", COLORS["teal"])
    card(s, 4.75, 1.35, 3.85, 4.8, "Main Outputs", "metrics_logs/scenesense_analysis/<run_group>/\n\nmetrics_logs/scenesense_ttracer/<run_group>/ue/analysis/\n\nmetrics_logs/scenesense_ttracer/<run_group>/analysis/", COLORS["orange"])
    card(s, 8.85, 1.35, 3.85, 4.8, "One Post-Run Command", "scripts/run_logging_validation_analysis.sh \\\n  --run-group <label> \\\n  --window-s 1.0\n\nProduces app summary, UE grant windows, UE payload validation, UE-gNB comparison, and stdout parse.", COLORS["green"])
    slides.append(s)

    s = Slide("UE-Side RL Agent Formulation", "Fast local compression control")
    card(s, 0.65, 1.35, 4.15, 4.8, "State s_t^UE", "- app payload trend\n- RTT / timeout trend\n- scene complexity later\n- UL MCS, RBs, symbols\n- scheduled Mbps\n- HARQ/retx indicators\n- last action", COLORS["orange"])
    card(s, 4.95, 1.35, 3.35, 4.8, "Action a_t^UE", "- AE channels\n- quantization mode\n- entropy coder\n- ROI threshold\n- feature/drop policy\n- frame/stream rate", COLORS["teal"])
    card(s, 8.45, 1.35, 4.05, 4.8, "Reward", "r_t = Q_task\n      - lambda_L * latency\n      - lambda_B * bytes\n      - lambda_D * drops\n      - lambda_T * timeout\n\nGoal: send just enough perception information under changing network capacity.", COLORS["green"])
    slides.append(s)

    s = Slide("Spatial-Map Sharing Agent Formulation", "Server-side orchestration")
    card(s, 0.65, 1.35, 4.1, 4.8, "State s_t^map(i)", "- target UE identity\n- object/update age\n- occlusion / risk score\n- expected payload bytes\n- DL MCS / TBS / PRBs\n- BLER / HARQ / DTX\n- map freshness", COLORS["purple"])
    card(s, 4.95, 1.35, 3.35, 4.8, "Action a_t^map", "- which UE to send to\n- what update to send\n- when to send/defer\n- object subset vs mask\n- payload budget\n- priority class", COLORS["orange"])
    card(s, 8.45, 1.35, 4.05, 4.8, "Reward", "r_t = safety_value\n      + freshness_gain\n      - lambda_L * delivery_delay\n      - lambda_B * bytes\n      - lambda_S * stale_updates\n\nGoal: deliver the right spatial-map update to the right UE before it becomes stale.", COLORS["green"])
    slides.append(s)

    s = Slide("Two-Agent View", "How the pieces fit")
    s.add_text(0.7, 1.35, 3.1, 0.9, "UE Agent\ncompression + feature policy", font=14, color=COLORS["white"], fill=COLORS["orange"], radius=True, align="ctr")
    s.add_text(4.95, 1.35, 3.1, 0.9, "5G Transport\nOAI RAN/Core", font=14, color=COLORS["white"], fill=COLORS["navy"], radius=True, align="ctr")
    s.add_text(9.2, 1.35, 3.1, 0.9, "Spatial Agent\nsharing + orchestration", font=14, color=COLORS["white"], fill=COLORS["purple"], radius=True, align="ctr")
    s.add_arrow(3.95, 1.62, 0.75, 0.32, COLORS["orange"])
    s.add_arrow(8.2, 1.62, 0.75, 0.32, COLORS["orange"])
    card(s, 0.75, 3.0, 3.0, 2.1, "UE Inputs", "Local grants\nApp latency\nPayload trend\nTimeouts", COLORS["orange"])
    card(s, 5.05, 3.0, 3.0, 2.1, "Network Evidence", "gNB MAC/PHY\nBLER/HARQ\nRLC/PDCP\nstdout summaries", COLORS["green"])
    card(s, 9.25, 3.0, 3.0, 2.1, "Server Inputs", "Target UE DL feasibility\nMap freshness\nOcclusion/risk\nPayload budget", COLORS["purple"])
    s.add_text(1.0, 5.8, 11.3, 0.55, "Principle: fast UE decisions use local decoded grants; server-side sharing decisions use richer gNB/network context without pushing every metric back to every UE.", font=15, color=COLORS["navy"], bold=True, align="ctr")
    slides.append(s)

    s = Slide("What Is Closed vs. What Comes Next", "Project direction")
    card(s, 0.65, 1.35, 3.85, 4.8, "Closed Now", "- OAI multi-UE fusion transport\n- App logging\n- UE decoded-grant logging\n- gNB radio logging\n- Validation scripts\n- Analysis runbook", COLORS["green"])
    card(s, 4.75, 1.35, 3.85, 4.8, "Useful Later", "- Clean NR UE CSI/CQI event if needed\n- 5QI/QFI/SDAP labeling\n- Wider stress scenarios\n- Raw channel impairment sweeps", COLORS["blue"])
    card(s, 8.85, 1.35, 3.85, 4.8, "Next Checklist Item", "- Controlled CARLA scenario harness\n- Object density and occlusion controls\n- Repeatable scene labels\n- Task-quality metrics: mIoU, recall, localization error\n- Then RL prototype", COLORS["orange"])
    slides.append(s)

    return slides


def write_pptx(slides: Sequence[Slide], out_path: Path) -> None:
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides)))
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("docProps/app.xml", app_xml(len(slides)))
        zf.writestr("docProps/core.xml", CORE_XML)
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        zf.writestr("ppt/theme/theme1.xml", THEME_XML)
        zf.writestr("ppt/slideMasters/slideMaster1.xml", MASTER_XML)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", MASTER_RELS)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT_XML)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", LAYOUT_RELS)
        for i, slide in enumerate(slides, 1):
            zf.writestr(f"ppt/slides/slide{i}.xml", slide.xml(i))
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels())


def main() -> int:
    slides = build_slides()
    write_pptx(slides, OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print(f"Slides: {len(slides)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
