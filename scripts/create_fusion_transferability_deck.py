#!/usr/bin/env python3
"""Create a SceneSense fusion transferability PowerPoint deck.

The environment does not provide python-pptx, so this script writes a small
PowerPoint OOXML package directly. It also generates OD transferability plots
from the summary numbers reported by the latest experiment.
"""

from __future__ import annotations

import csv
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", "/tmp/scenesense_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ABIODUN_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ABIODUN_DIR / "metrics_logs" / "scenesense_analysis" / "fusion_transferability_presentation"
OUT_PPTX = ABIODUN_DIR / "SceneSense_Fusion_Model_Transferability_OD_SEG.pptx"
SEG_DIR = ABIODUN_DIR / "metrics_logs" / "scenesense_analysis" / "pole_vs_ego_transfer_presentation"
SEG_CSV = SEG_DIR / "pole_vs_ego_transfer_iou_summary.csv"

SLIDE_W = 13.333
SLIDE_H = 7.5
EMU_PER_IN = 914400

COLORS = {
    "navy": "17324D",
    "teal": "007C89",
    "green": "3D7A46",
    "orange": "D56A21",
    "red": "B23A48",
    "blue": "2F6FAD",
    "purple": "7057A3",
    "gray": "F3F6F8",
    "midgray": "D8E0E7",
    "dark": "202A33",
    "muted": "637381",
    "white": "FFFFFF",
    "black": "000000",
}


OD_2M = [
    {
        "stream": "Ego S1",
        "stream_id": "fusion_ego_front",
        "platform": "Parked Ego",
        "gt": 2049,
        "pred": 5091,
        "recall": 0.025,
        "mean_xy": 1.301,
    },
    {
        "stream": "Ego S2",
        "stream_id": "fusion_ego_front_view_2",
        "platform": "Parked Ego",
        "gt": 1322,
        "pred": 3901,
        "recall": 0.024,
        "mean_xy": 1.412,
    },
    {
        "stream": "Pole S1",
        "stream_id": "fusion_tl_14",
        "platform": "Pole",
        "gt": 1158,
        "pred": 9894,
        "recall": 0.282,
        "mean_xy": 1.088,
    },
    {
        "stream": "Pole S2",
        "stream_id": "fusion_tl_14_view_2",
        "platform": "Pole",
        "gt": 1586,
        "pred": 9499,
        "recall": 0.354,
        "mean_xy": 1.151,
    },
]

OD_5M = [
    {
        "stream": "Ego S1",
        "stream_id": "fusion_ego_front",
        "platform": "Parked Ego",
        "gt": 2049,
        "pred": 5091,
        "recall": 0.136,
        "mean_xy": 3.361,
    },
    {
        "stream": "Ego S2",
        "stream_id": "fusion_ego_front_view_2",
        "platform": "Parked Ego",
        "gt": 1322,
        "pred": 3901,
        "recall": 0.108,
        "mean_xy": 3.054,
    },
    {
        "stream": "Pole S1",
        "stream_id": "fusion_tl_14",
        "platform": "Pole",
        "gt": 1158,
        "pred": 9894,
        "recall": 0.523,
        "mean_xy": 2.103,
    },
    {
        "stream": "Pole S2",
        "stream_id": "fusion_tl_14_view_2",
        "platform": "Pole",
        "gt": 1586,
        "pred": 9499,
        "recall": 0.637,
        "mean_xy": 2.019,
    },
]


def emu(value: float) -> int:
    return int(round(value * EMU_PER_IN))


def xml_text(value: object) -> str:
    return escape(str(value), {"\"": "&quot;", "'": "&apos;"})


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def platform_average(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for platform in ("Pole", "Parked Ego"):
        result[platform] = mean(float(row[key]) for row in rows if row["platform"] == platform)
    return result


def retention(ego: float, pole: float) -> float:
    return ego / pole if pole else 0.0


def read_segmentation_rows() -> List[Dict[str, object]]:
    with SEG_CSV.open("r", newline="", encoding="utf-8") as handle:
        return [
            {
                **row,
                "miou_binary": float(row["miou_binary"]),
                "miou_3class_macro": float(row["miou_3class_macro"]),
                "miou_vehicle_iou": float(row["miou_vehicle_iou"]),
                "miou_person_iou": float(row["miou_person_iou"]),
                "frames": int(row["frames"]),
            }
            for row in csv.DictReader(handle)
        ]


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8e0e7", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, bars: Sequence[object], fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            float(bar.get_x()) + float(bar.get_width()) / 2.0,
            height + 0.015,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202a33",
        )


def plot_od_recall() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "fusion_od_recall_by_stream.png"
    labels = [row["stream"] for row in OD_2M]
    x = np.arange(len(labels))
    width = 0.36
    colors = ["#007C89" if row["platform"] == "Pole" else "#D56A21" for row in OD_2M]
    fig, ax = plt.subplots(figsize=(10.2, 5.3))
    bars_2m = ax.bar(x - width / 2, [row["recall"] for row in OD_2M], width, color=colors, alpha=0.92, label="Recall@2m")
    bars_5m = ax.bar(x + width / 2, [row["recall"] for row in OD_5M], width, color=colors, alpha=0.45, hatch="//", label="Recall@5m")
    annotate_bars(ax, bars_2m)
    annotate_bars(ax, bars_5m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Recall")
    ax.set_title("Fusion Object Detection Recall by Stream")
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_od_platform() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "fusion_od_platform_summary.png"
    rec_2m = platform_average(OD_2M, "recall")
    rec_5m = platform_average(OD_5M, "recall")
    xy_2m = platform_average(OD_2M, "mean_xy")
    xy_5m = platform_average(OD_5M, "mean_xy")
    labels = ["Pole", "Parked Ego"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    b1 = axes[0].bar(x - 0.18, [rec_2m[label] for label in labels], 0.36, color="#007C89", label="Recall@2m")
    b2 = axes[0].bar(x + 0.18, [rec_5m[label] for label in labels], 0.36, color="#D56A21", label="Recall@5m")
    annotate_bars(axes[0], b1)
    annotate_bars(axes[0], b2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 0.68)
    axes[0].set_ylabel("Platform-average recall")
    axes[0].set_title("OD Recall Transfer")
    axes[0].legend(frameon=False, loc="upper right")
    style_axes(axes[0])

    b3 = axes[1].bar(x - 0.18, [xy_2m[label] for label in labels], 0.36, color="#007C89", label="2m matches")
    b4 = axes[1].bar(x + 0.18, [xy_5m[label] for label in labels], 0.36, color="#D56A21", label="5m matches")
    annotate_bars(axes[1], b3)
    annotate_bars(axes[1], b4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 3.75)
    axes[1].set_ylabel("Mean XY error on matched objects (m)")
    axes[1].set_title("Localization Error")
    axes[1].legend(frameon=False, loc="upper left")
    style_axes(axes[1])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_transfer_drop(seg_rows: Sequence[Dict[str, object]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "fusion_transfer_retention_summary.png"
    seg_binary = platform_average(seg_rows, "miou_binary")
    seg_vehicle = platform_average(seg_rows, "miou_vehicle_iou")
    od_2m = platform_average(OD_2M, "recall")
    od_5m = platform_average(OD_5M, "recall")
    metrics = [
        ("SEG binary IoU", retention(seg_binary["Parked Ego"], seg_binary["Pole"])),
        ("SEG vehicle IoU", retention(seg_vehicle["Parked Ego"], seg_vehicle["Pole"])),
        ("OD recall@2m", retention(od_2m["Parked Ego"], od_2m["Pole"])),
        ("OD recall@5m", retention(od_5m["Parked Ego"], od_5m["Pole"])),
    ]
    labels = [item[0] for item in metrics]
    values = [item[1] for item in metrics]
    colors = ["#3D7A46", "#3D7A46", "#B23A48", "#D56A21"]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    bars = ax.bar(labels, values, color=colors)
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            float(bar.get_x()) + float(bar.get_width()) / 2.0,
            height + 0.025,
            f"{height * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#202a33",
        )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Parked-ego performance / pole performance")
    ax.set_title("Transfer Retention: Pole-Trained Fusion Model")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_od_csvs() -> Tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stream_path = OUT_DIR / "fusion_od_transfer_summary_from_reported_results.csv"
    with stream_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["threshold_m", "stream", "stream_id", "platform", "gt", "pred", "recall", "mean_xy_error_m"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for threshold_m, rows in ((2, OD_2M), (5, OD_5M)):
            for row in rows:
                writer.writerow(
                    {
                        "threshold_m": threshold_m,
                        "stream": row["stream"],
                        "stream_id": row["stream_id"],
                        "platform": row["platform"],
                        "gt": row["gt"],
                        "pred": row["pred"],
                        "recall": row["recall"],
                        "mean_xy_error_m": row["mean_xy"],
                    }
                )

    platform_path = OUT_DIR / "fusion_od_platform_summary_from_reported_results.csv"
    with platform_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["threshold_m", "platform", "mean_recall", "mean_xy_error_m"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for threshold_m, rows in ((2, OD_2M), (5, OD_5M)):
            for platform in ("Pole", "Parked Ego"):
                selected = [row for row in rows if row["platform"] == platform]
                writer.writerow(
                    {
                        "threshold_m": threshold_m,
                        "platform": platform,
                        "mean_recall": mean(float(row["recall"]) for row in selected),
                        "mean_xy_error_m": mean(float(row["mean_xy"]) for row in selected),
                    }
                )
    return stream_path, platform_path


def generate_assets() -> Dict[str, Path]:
    seg_rows = read_segmentation_rows()
    paths = {
        "seg_platform": SEG_DIR / "pole_vs_ego_platform_average_iou.png",
        "seg_stream": SEG_DIR / "pole_vs_ego_stream_iou_core_metrics.png",
        "seg_all": SEG_DIR / "pole_vs_ego_stream_iou_all_metrics.png",
        "od_recall": plot_od_recall(),
        "od_platform": plot_od_platform(),
        "retention": plot_transfer_drop(seg_rows),
    }
    write_od_csvs()
    return paths


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
  <Default Extension="png" ContentType="image/png"/>
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
  <Application>SceneSense Transfer Deck Generator</Application>
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
  <dc:title>SceneSense Fusion Model Transferability</dc:title>
  <dc:creator>SceneSense Project</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dc:description>Pole-trained RGB+radar fusion model transfer from traffic-light poles to parked ego vehicles.</dc:description>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-06-04T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-04T00:00:00Z</dcterms:modified>
</cp:coreProperties>'''


def presentation_xml(slide_count: int) -> str:
    slide_ids = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, slide_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
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
            f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {"".join(rels)}
</Relationships>'''


THEME_XML = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="SceneSense Transfer">
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
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="25400"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="38100"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
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
        self.images: List[Tuple[str, Path]] = []
        self.next_id = 2
        self.add_rect(0, 0, SLIDE_W, SLIDE_H, COLORS["white"], line=None)
        self.add_rect(0, 0, 0.18, SLIDE_H, COLORS["teal"], line=None)
        if kicker:
            self.add_text(0.55, 0.22, 7.5, 0.25, kicker.upper(), font=8, color=COLORS["teal"], bold=True)
        if title:
            self.add_text(0.55, 0.46, 12.0, 0.58, title, font=25, color=COLORS["navy"], bold=True)
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
        prst = "roundRect" if radius else "rect"
        line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        self.shapes.append(f'''
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}</p:spPr></p:sp>''')

    def add_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        font: int = 14,
        color: str = COLORS["dark"],
        bold: bool = False,
        align: str = "l",
        fill: str | None = None,
        line: str | None = None,
        radius: bool = False,
        margin: float = 0.08,
    ) -> None:
        sid = self._shape_id()
        fill_xml = "<a:noFill/>" if fill is None else f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        line_xml = '<a:ln><a:noFill/></a:ln>' if line is None else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        prst = "roundRect" if radius else "rect"
        paragraphs = []
        for raw_line in text.split("\n"):
            line_text = raw_line.rstrip()
            if not line_text:
                paragraphs.append("<a:p/>")
                continue
            bold_attr = ' b="1"' if bold else ""
            rpr = f'<a:rPr lang="en-US" sz="{font * 100}"{bold_attr}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr>'
            paragraphs.append(f'<a:p><a:pPr algn="{align}"/><a:r>{rpr}<a:t>{xml_text(line_text)}</a:t></a:r></a:p>')
        self.shapes.append(f'''
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
<p:txBody><a:bodyPr wrap="square" lIns="{emu(margin)}" rIns="{emu(margin)}" tIns="{emu(margin)}" bIns="{emu(margin)}"><a:spAutoFit/></a:bodyPr><a:lstStyle/>{"".join(paragraphs)}</p:txBody></p:sp>''')

    def add_image(self, path: Path, x: float, y: float, w: float, h: float) -> None:
        sid = self._shape_id()
        rel_id = f"rId{len(self.images) + 2}"
        self.images.append((rel_id, path))
        self.shapes.append(f'''
<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{xml_text(path.name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>''')

    def add_footer(self, slide_num: int) -> None:
        self.add_text(0.55, 7.12, 5.2, 0.2, "SceneSense fusion transferability", font=7, color=COLORS["muted"])
        self.add_text(12.2, 7.12, 0.55, 0.2, str(slide_num), font=7, color=COLORS["muted"], align="r")

    def xml(self, slide_num: int) -> str:
        self.add_footer(slide_num)
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {"".join(self.shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def slide_rels(slide: Slide, media_map: Dict[Path, str]) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    for rel_id, path in slide.images:
        rels.append(
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_map[path]}"/>'
        )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {"".join(rels)}
</Relationships>'''


def card(slide: Slide, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    slide.add_text(x, y, w, h, f"{title}\n{body}", font=12, color=COLORS["dark"], fill="FFFFFF", line=color, radius=True, margin=0.12)
    slide.add_rect(x, y, 0.08, h, color, line=None)


def metric_card(slide: Slide, x: float, y: float, w: float, h: float, label: str, value: str, note: str, color: str) -> None:
    slide.add_text(x, y, w, h, f"{value}\n{label}\n{note}", font=12, color=COLORS["dark"], fill=COLORS["gray"], line=color, radius=True, align="ctr", margin=0.08)


def build_slides(paths: Dict[str, Path]) -> List[Slide]:
    seg_rows = read_segmentation_rows()
    seg_binary = platform_average(seg_rows, "miou_binary")
    seg_vehicle = platform_average(seg_rows, "miou_vehicle_iou")
    od_2m = platform_average(OD_2M, "recall")
    od_5m = platform_average(OD_5M, "recall")
    xy_2m = platform_average(OD_2M, "mean_xy")
    xy_5m = platform_average(OD_5M, "mean_xy")

    slides: List[Slide] = []

    s = Slide()
    s.add_text(0.65, 0.65, 11.8, 0.72, "SceneSense Fusion Model Transferability", font=32, color=COLORS["navy"], bold=True)
    s.add_rect(0.65, 1.5, 1.5, 0.06, COLORS["orange"], line=None)
    s.add_text(0.65, 1.82, 11.2, 0.52, "Traffic-light-pole-trained RGB+radar fusion model tested on parked ego-vehicle sensor placement", font=17, color=COLORS["dark"])
    metric_card(s, 0.8, 3.0, 2.55, 1.25, "SEG binary IoU retention", f"{retention(seg_binary['Parked Ego'], seg_binary['Pole']) * 100:.1f}%", "parked ego / pole", COLORS["green"])
    metric_card(s, 3.75, 3.0, 2.55, 1.25, "SEG vehicle IoU retention", f"{retention(seg_vehicle['Parked Ego'], seg_vehicle['Pole']) * 100:.1f}%", "parked ego / pole", COLORS["green"])
    metric_card(s, 6.7, 3.0, 2.55, 1.25, "OD recall@2m retention", f"{retention(od_2m['Parked Ego'], od_2m['Pole']) * 100:.1f}%", "precise localization", COLORS["red"])
    metric_card(s, 9.65, 3.0, 2.55, 1.25, "OD recall@5m retention", f"{retention(od_5m['Parked Ego'], od_5m['Pole']) * 100:.1f}%", "loose sensitivity", COLORS["orange"])
    s.add_text(1.0, 5.25, 11.2, 0.62, "Bottom line: segmentation transfers partially; object detection/localization drops sharply when moving from pole cameras to parked ego cameras.", font=18, color=COLORS["navy"], bold=True, align="ctr")
    slides.append(s)

    s = Slide("Experiment Design", "Setup")
    card(s, 0.65, 1.35, 3.75, 4.95, "Question", "The fusion model was trained with cameras/radar mounted near traffic-light poles. We moved the same checkpoint to parked ego vehicles and ask how much task accuracy transfers.", COLORS["teal"])
    card(s, 4.65, 1.35, 3.75, 4.95, "Streams", "Pole S1: fusion_tl_14\nPole S2: fusion_tl_14_view_2\n\nEgo S1: fusion_ego_front\nEgo S2: fusion_ego_front_view_2\n\nRuns were collected as pole pair first, parked-ego pair second.", COLORS["orange"])
    card(s, 8.65, 1.35, 3.75, 4.95, "Controls", "Same checkpoint:\ncheckpoints/fusion_object_best.pt\n\nSame loopback split transport, run logging, CARLA actor/semantic GT path, and comparable traffic settings.", COLORS["green"])
    slides.append(s)

    s = Slide("Ground Truth and Metric Definitions", "How accuracy is measured")
    card(s, 0.65, 1.35, 3.95, 4.95, "Segmentation GT", "A co-located CARLA semantic-segmentation camera produces per-pixel class labels. CARLA tags are mapped to three classes: background, vehicle, and person.\n\nIoU = TP / (TP + FP + FN)", COLORS["green"])
    card(s, 4.9, 1.35, 3.95, 4.95, "Object GT", "CARLA vehicle actors provide world pose and bounding boxes. 3D boxes are projected into the RGB camera. Evaluation filters require visible center, minimum box area, and max distance for the strict OD comparison. The parked ego vehicle itself is excluded.", COLORS["orange"])
    card(s, 9.15, 1.35, 3.05, 4.95, "OD Metrics", "Recall@d = matched GT / selected GT\n\nA match is a greedy nearest-neighbor pair within d meters in global XY.\n\nMean XY error is the average distance over matched object centers.", COLORS["teal"])
    slides.append(s)

    s = Slide("Segmentation Transfer Result", "fusion_as_segmentation")
    s.add_image(paths["seg_platform"], 0.65, 1.28, 6.1, 3.25)
    s.add_image(paths["seg_stream"], 6.95, 1.28, 5.8, 3.25)
    card(s, 0.85, 5.0, 3.8, 1.2, "Binary IoU", f"Pole {seg_binary['Pole']:.3f} -> Ego {seg_binary['Parked Ego']:.3f}\nRetention {retention(seg_binary['Parked Ego'], seg_binary['Pole']) * 100:.1f}%", COLORS["green"])
    card(s, 4.95, 5.0, 3.8, 1.2, "Vehicle IoU", f"Pole {seg_vehicle['Pole']:.3f} -> Ego {seg_vehicle['Parked Ego']:.3f}\nRetention {retention(seg_vehicle['Parked Ego'], seg_vehicle['Pole']) * 100:.1f}%", COLORS["green"])
    card(s, 9.05, 5.0, 3.0, 1.2, "Interpretation", "Foreground transfer is strong, but vehicle segmentation degrades from the new viewpoint.", COLORS["orange"])
    slides.append(s)

    s = Slide("Object Detection Evaluation", "fusion_as_od")
    card(s, 0.65, 1.35, 4.0, 4.95, "Prediction Path", "The fusion object head decodes center heatmap peaks into world XYZ, yaw, size, parked score, and radar-support score. Predictions are logged per frame and stream.", COLORS["teal"])
    card(s, 4.95, 1.35, 3.65, 4.95, "Strict OD View", "Recall@2m asks whether the object is found with useful localization precision. This is the main transfer metric for spatial-map usefulness.", COLORS["red"])
    card(s, 8.9, 1.35, 3.65, 4.95, "Sensitivity View", "Recall@5m asks whether the model is roughly finding the object even if localization is loose. It helps separate center-detection failure from regression precision failure.", COLORS["orange"])
    slides.append(s)

    s = Slide("Object Recall by Stream", "OD result")
    s.add_image(paths["od_recall"], 0.75, 1.3, 7.1, 3.75)
    card(s, 8.2, 1.35, 4.0, 1.35, "Pole In-Domain", "Recall@2m: 0.282 / 0.354\nRecall@5m: 0.523 / 0.637", COLORS["teal"])
    card(s, 8.2, 3.0, 4.0, 1.35, "Parked Ego Transfer", "Recall@2m: 0.025 / 0.024\nRecall@5m: 0.136 / 0.108", COLORS["orange"])
    card(s, 8.2, 4.65, 4.0, 1.25, "Reading", "The pole-trained object head remains much stronger on pole views. Parked-ego recall remains low even with a loose 5 m match gate.", COLORS["red"])
    slides.append(s)

    s = Slide("OD Platform Summary", "Recall and localization")
    s.add_image(paths["od_platform"], 0.75, 1.25, 7.35, 3.75)
    card(s, 8.45, 1.35, 3.8, 1.2, "Average Recall@2m", f"Pole {od_2m['Pole']:.3f}\nParked Ego {od_2m['Parked Ego']:.3f}\nRetention {retention(od_2m['Parked Ego'], od_2m['Pole']) * 100:.1f}%", COLORS["red"])
    card(s, 8.45, 2.85, 3.8, 1.2, "Average Recall@5m", f"Pole {od_5m['Pole']:.3f}\nParked Ego {od_5m['Parked Ego']:.3f}\nRetention {retention(od_5m['Parked Ego'], od_5m['Pole']) * 100:.1f}%", COLORS["orange"])
    card(s, 8.45, 4.35, 3.8, 1.2, "Mean XY Error", f"2m gate: Pole {xy_2m['Pole']:.2f} m, Ego {xy_2m['Parked Ego']:.2f} m\n5m gate: Pole {xy_5m['Pole']:.2f} m, Ego {xy_5m['Parked Ego']:.2f} m", COLORS["teal"])
    slides.append(s)

    s = Slide("Transfer Retention Across Tasks", "Pole-trained checkpoint")
    s.add_image(paths["retention"], 0.7, 1.25, 7.15, 3.35)
    card(s, 8.15, 1.35, 4.05, 1.35, "Segmentation", "Binary foreground segmentation nearly transfers. Vehicle IoU transfers at about 64% of pole performance.", COLORS["green"])
    card(s, 8.15, 3.0, 4.05, 1.35, "Object Detection", "OD recall transfers poorly: about 8% retention at 2 m, about 21% retention at 5 m.", COLORS["red"])
    card(s, 8.15, 4.65, 4.05, 1.35, "Why It Matters", "Spatial maps need object centers and pose, not just foreground pixels. OD transfer is the blocker for parked-ego map sharing.", COLORS["orange"])
    slides.append(s)

    s = Slide("Interpretation", "What the numbers imply")
    card(s, 0.65, 1.35, 3.85, 4.95, "Not a Total Coordinate Failure", "Matched OD predictions have reasonable XY error: about 1.1 m on pole streams at the 2 m gate. The model can localize objects it detects.", COLORS["teal"])
    card(s, 4.75, 1.35, 3.85, 4.95, "Likely Failure Mode", "The object-center detector and confidence distribution do not transfer from high pole views to parked-ego front views. The segmentation head is more robust than the object head.", COLORS["orange"])
    card(s, 8.85, 1.35, 3.85, 4.95, "Checkpoint Caveat", "Checkpoint metadata stores best_miou and selection score. It does not store object AP/recall, so the saved best checkpoint may be segmentation-selected rather than OD-selected.", COLORS["purple"])
    slides.append(s)

    s = Slide("Conclusion and Next Steps", "Decision")
    s.add_text(0.75, 1.45, 11.9, 0.82, "Conclusion: the pole-trained RGB+radar fusion model partially transfers for segmentation, but does not transfer well enough for parked-ego object detection/localization.", font=20, color=COLORS["navy"], bold=True, align="ctr")
    card(s, 0.85, 2.65, 3.75, 3.25, "Use As-Is?", "Segmentation: usable for baseline transfer analysis.\n\nOD/localization: not sufficient for parked-ego spatial-map experiments.", COLORS["red"])
    card(s, 4.85, 2.65, 3.75, 3.25, "Recommended Fix", "Collect parked-ego RGB/radar/object-label samples and fine-tune or retrain the object head. Keep pole data mixed in to avoid forgetting.", COLORS["green"])
    card(s, 8.85, 2.65, 3.75, 3.25, "Evaluation Next", "Pull full OD CSVs if we want yaw/dimension plots, score calibration, false positives per frame, and confidence-threshold sweeps.", COLORS["orange"])
    slides.append(s)

    return slides


def write_pptx(slides: Sequence[Slide], out_path: Path) -> None:
    image_paths: List[Path] = []
    for slide in slides:
        image_paths.extend(path for _rel_id, path in slide.images)
    media_map: Dict[Path, str] = {}
    for index, path in enumerate(dict.fromkeys(image_paths), start=1):
        media_map[path] = f"image{index}.png"

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
        for source_path, media_name in media_map.items():
            zf.write(source_path, f"ppt/media/{media_name}")
        for index, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{index}.xml", slide.xml(index))
            zf.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", slide_rels(slide, media_map))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = generate_assets()
    slides = build_slides(paths)
    write_pptx(slides, OUT_PPTX)
    shutil.copy2(OUT_PPTX, OUT_DIR / OUT_PPTX.name)
    print(f"Wrote {OUT_PPTX}")
    print(f"Copied deck to {OUT_DIR / OUT_PPTX.name}")
    print(f"Slides: {len(slides)}")
    print(f"Assets: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
