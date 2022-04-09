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
import numpy as np
import numpy.typing
import scipy.stats
from PIL import Image
from kraken.lib.xml import parse_xml
from kraken.lib.segmentation import extract_polygons
import lxml.etree as ET
# Web deps
from flask import Flask, current_app, request, render_template, Response, send_file
from shapely.geometry import LineString

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


app = Flask(
    "balto-masqué",
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "statics")
)
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


BoundaryBaselineY = namedtuple("BoundaryBaselineY", ["max_y", "min_y", "base_y_max", "base_y_min",
                                                     "dist_min", "dist_max"])
Score = namedtuple("Score", ["median", "iqr"])
Outlier = namedtuple("Outlier", ["idx", "value", "score"])


def get_min_max_y(lines: List[Dict[str, Any]]) -> Iterable[BoundaryBaselineY]:
    for line in lines:
        ys = []
        dists = []
        baselines = []
        for (x_pol, y_pol) in line["boundary"]:
            _, base_y = get_closest_points((x_pol, y_pol), line["advanced_baseline"])
            dist = abs(base_y - y_pol)
            if dist != 0:
                ys.append(y_pol)
                dists.append(dist)
                baselines.append(base_y)
        max_idx = dists.index(max(dists))
        min_idx = dists.index(min(dists))
        yield BoundaryBaselineY(
            ys[max_idx],
            ys[min_idx],
            baselines[max_idx],
            baselines[min_idx],
            dists[max_idx],
            dists[min_idx]
        )


def get_diff(bby: Union[Tuple[int, int], BoundaryBaselineY], mode: str = "min_y"):
    if mode == "min_y":
        return bby.dist_min
    else:
        return bby.dist_max


def is_outlier(bby: Union[Tuple[int, int], BoundaryBaselineY], score: Score, mode: str = "min_y"):
    diff_y = get_diff(bby, mode=mode)
    if diff_y > score.iqr:
        return diff_y, abs(score.iqr - diff_y)
    return False


def is_up_outlier(p_y: int, b_y: int, score: Score):
    diff_y = abs(p_y - b_y)
    if diff_y < score.iqr:
        return False
    return True


def get_outliers_iqr(lines: List[BoundaryBaselineY], score: Score, attr: str = "min_y") -> Iterable[Outlier]:
    for line_idx, line in enumerate(lines):
        was_outlier = is_outlier(line, score, mode=attr)
        if was_outlier:
            yield Outlier(line_idx, *was_outlier)


def compute_cuttings(boundaries: List[BoundaryBaselineY], qrt_bot: int = 10, qrt_top: int = 10) -> Tuple[Score, Score]:
    diff_y_max, diff_y_min = list(zip(
        *[
            (bby.dist_max, bby.dist_min)
            for bby in boundaries
        ]
    ))
    min_score = Score(
        median(diff_y_min),
        scipy.stats.scoreatpercentile(diff_y_min, 100-qrt_top)#+median(diff_y_min)
    )
    max_score = Score(
        median(diff_y_max),
        scipy.stats.scoreatpercentile(diff_y_max, 100-qrt_bot)#+median(diff_y_max)
    )
    return max_score, min_score


def img_to_base64(img: Tuple[Image.Image, Any]) -> bytes:
    img, line = img
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return (bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str).decode()


def redraw_polygon(
        x_y: Iterable[Tuple[int, int]],
        baseline: numpy.typing.ArrayLike,
        min_y: Optional[Score] = None,
        max_y: Optional[Score] = None,
        margin_min_y: int = 0,
        margin_max_y: int = 0
):
    for (x, y) in x_y:
        _, baseline_y = get_closest_points((x, y), baseline)
        if max_y and y > baseline_y and is_up_outlier(y, baseline_y, max_y):
            yield x, ceil(baseline_y + max_y.median) + (margin_max_y or 0)
        elif min_y and y < baseline_y and is_up_outlier(y, baseline_y, min_y):
            yield x, ceil(baseline_y - min_y.median) - (margin_min_y or 0)
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
        baseline = line["advanced_baseline"]
        details = outliers[line["idx"]]
        if details["max"] and details["min"]:
            scores = dict(min_y=min_iqr, max_y=max_iqr)
        elif details["max"]:
            scores = dict(max_y=max_iqr)
        else:
            scores = dict(min_y=min_iqr)
        if line["idx"] in margins:
            scores.update(margins[line["idx"]])
        new_lines.append({
            **line,
            "boundary": list(redraw_polygon(line["boundary"], baseline=baseline, **scores))
        })

    return new_lines


def get_closest_points(current_point: Tuple[int, int], baseline: np.typing.ArrayLike) -> Tuple[int, int]:
    distances = np.linalg.norm(baseline - np.array(current_point), axis=1)
    min_index = np.argmin(distances)
    return tuple(baseline[min_index])


def get_all_points(baseline: List[Tuple[int, int]]) -> np.typing.ArrayLike:
    # https://gis.stackexchange.com/questions/263859/fast-way-to-get-all-points-as-integer-of-a-linestring-in-shapely
    ls = LineString(baseline)
    xy = []
    for f in range(0, int(ceil(ls.length)) + 1, ceil(ls.length/50)):
        p = ls.interpolate(f).coords[0]
        pr = tuple(map(round, p))
        if pr not in xy:
            xy.append(pr)
    return np.array(xy)


def add_advanced_baseline(line: Dict[str, Any]) -> Dict[str, Any]:
    line["advanced_baseline"] = get_all_points(line["baseline"])
    return line


@app.route('/image/<path:path>')
def get_image(path):
    return send_file("/"+path if path.startswith("home") else path)


@app.route("/", methods=["GET", "POST"])
def index():
    xmls = get_xml()
    details = {
        page: {**parse_xml(page), "basename": os.path.basename(page)}
        for page in xmls
    }
    return render_template("index.html", details=details, pages=xmls)


@app.route("/approach/stats", methods=["GET", "POST"])
def page_iqr():
    xmls = get_xml()
    page = request.args.get("page", xmls[0])
    qrt_bot = request.args.get("qrt_bot", 20, type=int)
    qrt_top = request.args.get("qrt_top", 10, type=int)
    # ToDo: Allow for linetype ignoring
    ignore_zone = request.args.get("ignore_zone", "", type=str)
    content = parse_xml(page)
    lines = content["lines"]
    lines = [add_advanced_baseline(line) for line in lines]

    masks_extremes = list(get_min_max_y(lines))
    max_cuttings, min_cuttings = compute_cuttings(masks_extremes, qrt_bot=qrt_bot, qrt_top=qrt_top)
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
                if "min" in field_name and request.form.get(f"update_min_{idx}", "off") == "on":
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
        return Response(ET.tostring(doc, encoding=str), mimetype="text/xml", headers={
            "Content-Disposition": f"attachment; filename={os.path.basename(page)}"
        })
    return render_template(
        "over_percentile.html",
        doc=doc,
        content=outliers,
        orig_images=orig_images,
        medians={"top": max_cuttings, "bot": min_cuttings},
        lines=lines,
        pages=xmls,
        current_page=page,
        qrt_bot=qrt_bot,
        qrt_top=qrt_top,
        preview=preview, margins=margins
    )
