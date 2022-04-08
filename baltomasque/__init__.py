import os
import glob
import logging
from math import ceil
from statistics import median
from collections import namedtuple
from typing import List, Optional, Iterable, Any, Dict, Tuple, Union
import base64
from io import BytesIO

# Kraken deps
import numpy
import scipy.stats
from PIL import Image
from kraken.lib.xml import parse_xml
from kraken.lib.segmentation import extract_polygons
import lxml.etree as ET
# Web deps
from flask import Flask, current_app, request, render_template, Response

Logger = logging.getLogger()
Logger.setLevel(logging.INFO)


class ConfigurationError(Exception):
    """ Error raised when the XML environment variable is not found """


class NoXMLFound(Exception):
    """ Error raised when no XML is found"""


def get_xml(app: Optional[Flask] = None) -> List[str]:
    if app:
        return app.config["XML"]
    with current_app.app_context():
        return current_app.config["XML"]


app = Flask("balto-masqué", template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "statics"))
app.config["DATA-ENV"] = os.getenv("XML")

if app.config["DATA-ENV"] is None:
    raise ConfigurationError("No XML environment variable found. Did you run XML=PATH ... run ?")

app.config["XML"] = [file for path in app.config["DATA-ENV"].split("|") for file in glob.glob(path)]

if not get_xml(app):
    raise ConfigurationError(f"No XML found. Did you check {app.config['DATA-ENV']} ?")

print(f" * [Details] Found {len(get_xml(app))} XML files")


@app.template_filter("nicename")
def nicename(filename):
    return os.path.basename(filename)


BoundaryBaselineY = namedtuple("BoundaryBaselineY", ["max_y", "min_y", "base_y_min", "base_y_max"])
Score = namedtuple("Score", ["median", "iqr"])
Outlier = namedtuple("Outlier", ["idx", "value", "score"])


def get_min_max_y(lines: List[Dict[str, Any]]) -> Iterable[BoundaryBaselineY]:
    for line in lines:
        _, y_pol = list(zip(*[points for points in line["boundary"]]))
        _, y_base = list(zip(*[points for points in line["baseline"]]))
        max_y = max(y_pol)
        min_y = min(y_pol)
        yield BoundaryBaselineY(max_y, min_y, min(y_base), max(y_base))


def get_diff(bby: Union[Tuple[int, int], BoundaryBaselineY], mode: str = "min_y"):
    if mode == "min_y":
        return bby.base_y_min - bby.min_y
    else:
        return bby.max_y - bby.base_y_max


def get_outliers_iqr(lines: List[BoundaryBaselineY], score: Score, attr: str = "min_y") -> Iterable[Outlier]:
    for line_idx, line in enumerate(lines):
        diff_y = get_diff(line, mode=attr)
        diff = abs(score.median - diff_y)
        if diff > score.iqr:
            yield Outlier(line_idx, diff_y, diff)


def compute_cuttings(boundaries: List[BoundaryBaselineY], qrt_bot: int = 10) -> Tuple[Score, Score]:
    qrt_top: int = 100 - qrt_bot
    diff_y_max, diff_y_min = list(zip(
        *[
            (bby.max_y-bby.base_y_max, bby.base_y_min-bby.min_y)
            for bby in boundaries
        ]
    ))
    min_score = Score(median(diff_y_min), scipy.stats.iqr(diff_y_min, rng=(qrt_bot, qrt_top)))
    max_score = Score(median(diff_y_max), scipy.stats.iqr(diff_y_max, rng=(qrt_bot, qrt_top)))
    return max_score, min_score


def img_to_base64(img: Tuple[Image.Image, Any]) -> bytes:
    img, line = img
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return (bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str).decode()


def _apply(
        x_y: Iterable[Tuple[int, int]],
        bby: BoundaryBaselineY,
        min_y: Optional[Score] = None,
        max_y: Optional[Score] = None,
        factor_min_y: float = 1,
        factor_max_y: float = 0,
        margin_min_y: int = 0,
        margin_max_y: int = 0
):
    baseline_y = (bby.base_y_max + bby.base_y_min) / 2
    for (x, y) in x_y:
        if max_y and y > baseline_y:
            diff_y = y - baseline_y
            diff = abs(max_y.median - diff_y)
            if diff > max_y.iqr:
                yield x, ceil(baseline_y + max_y.median)+(margin_max_y or 0)
            else:
                yield x, y
        elif min_y and y < baseline_y:
            diff_y = baseline_y - y
            diff = abs(min_y.median - diff_y)
            if diff > min_y.iqr:
                yield x, ceil(baseline_y - min_y.median)-(margin_min_y or 0)
            else:
                yield x, y
        else:
            yield x, y


def apply_iqr(
    outliers: Dict[int, Dict[str, Optional[Outlier]]],
    kept_lines: List[Dict[str, Any]],
    min_iqr: Score,
    max_iqr: Score,
    margins: Optional[Dict[int, Dict[str, int]]] = None
) -> List[Dict[str, Any]]:
    new_lines = []
    margins = margins or {}
    for line in kept_lines:
        details = outliers[line["idx"]]
        if details["max"] and details["min"]:
            scores = dict(min_y=min_iqr, max_y=max_iqr)
        elif details["max"]:
            scores = dict(max_y=max_iqr)
        else:
            scores = dict(min_y=max_iqr)
        if line["idx"] in margins:
            scores.update(margins[line["idx"]])
        new_lines.append({
            **line,
            "boundary": list(_apply(line["boundary"], line["bby"], **scores))
        })

    return new_lines


@app.route("/", methods=["GET", "POST"])
def get_page():
    xmls = get_xml()
    page = request.args.get("page", xmls[0])
    qrt = request.args.get("qrt", 10, type=int)
    # ToDo: Allow for linetype ignoring
    ignore_zone = request.args.get("ignore_zone", "", type=str)
    content = parse_xml(page)
    lines = content["lines"]

    masks_extremes = list(get_min_max_y(lines))
    max_cuttings, min_cuttings = compute_cuttings(masks_extremes, qrt_bot=qrt)
    max_outliers = list(get_outliers_iqr(masks_extremes, max_cuttings, "max_y"))
    min_outliers = list(get_outliers_iqr(masks_extremes, min_cuttings, "min_y"))

    outliers: Dict[int, Dict[str, Optional[Outlier]]] = {
        line.idx: {"max": line, "min": None}
        for line in max_outliers
    }
    for line in min_outliers:
        if line.idx not in outliers:
            outliers[line.idx] = {"max": None, "min": None}
        outliers[line.idx]["min"] = line

    image = Image.open(content["image"])

    kept_lines = [
        {"idx": idx, "bby": masks_extremes[idx], **line}
        for idx, line in enumerate(lines)
        if idx in outliers
    ]
    orig_images = {
        poly[1]["idx"]: img_to_base64(poly)
        for poly in extract_polygons(image, {"lines": kept_lines, "type": "baselines"})
    }

    # Don't forget it goes from y=0 at the top, so max_y is the bottomest thing !

    margins = {}
    if request.method == "POST":
        margins = {
            int(field_name.split("_")[-1]): {"margin_max_y": None, "margin_min_y": None}
            for field_name in request.form
            if field_name.startswith("update_") and request.form[field_name] == "on"
        }
        for field_name in request.form:
            if request.form[field_name] != "0" and field_name.startswith("custom_margin"):
                idx = int(field_name.split("_")[-1])
                if "max" in field_name and request.form.get(f"update_max_{idx}", "off") == "on":
                    margins[idx]["margin_max_y"] = int(request.form[field_name])
                elif "min" in field_name and request.form.get(f"update_min_{idx}", "off") == "on":
                    margins[idx]["margin_min_y"] = int(request.form[field_name])

    changes = apply_iqr(outliers, kept_lines, min_iqr=min_cuttings, max_iqr=max_cuttings, margins=margins)
    preview = {
        poly[1]["idx"]: {"img": img_to_base64(poly), "height": poly[0].height}
        for poly in extract_polygons(image, {"lines": changes, "type": "baselines"})
    }
    doc = None
    if request.form.get("serialize", "off") == "on":
        def map_coords(line_dict, attribute: str) -> str:
            points = line_dict[attribute]
            return " ".join([f"{x} {y}" for (x, y) in points])

        changed_lines = {
            map_coords(line, "baseline"): line
            for line in changes
            if line["idx"] in margins
        }
        doc = ET.parse(page)
        for line in doc.findall("//{*}TextLine"):
            if line.attrib["BASELINE"] in changed_lines:
                for poly in line.findall(".//{*}Polygon"):
                    poly.attrib["POINTS"] = map_coords(changed_lines[line.attrib["BASELINE"]], "boundary")
        return Response(ET.tostring(doc, encoding=str), mimetype="text/xml")
    return render_template(
        "container.html",
        doc=doc,
        content=outliers,
        orig_images=orig_images,
        medians={"top": max_cuttings, "bot": min_cuttings},
        lines=lines,
        pages=xmls,
        current_page=page,
        qrt=qrt, preview=preview, margins=margins
    )
