import os
import glob
import logging
from math import ceil
from dataclasses import dataclass
from statistics import median
from collections import namedtuple
from lss.utils import simplify_mask
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
from kraken.lib.exceptions import KrakenInputException
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


@dataclass
class BoundaryBaselineY:
    # All values are ABSOLUTE values
    max_y_under: int
    max_y_above: int
    base_y_max: int
    base_y_min: int
    max_under_distance: int
    max_above_distance: int


@dataclass
class Score:
    # All values are ABSOLUTE values
    median: float
    prct: float


Outlier = namedtuple("Outlier", ["idx", "value", "score"])


def get_min_max_y(lines: List[Dict[str, Any]]) -> Iterable[BoundaryBaselineY]:
    for line in lines:
        ys = []
        dists = []
        baselines = []
        for (x_pol, y_pol) in line["boundary"]:
            # If dist is negative, that means that the polygon is above the baseline, as Y starts from the top at 0
            #   If a polygon above the line was at 0 and the baseline at 5, -5 would be the dist
            _, base_y = get_closest_points((x_pol, y_pol), line["advanced_baseline"])
            dist = y_pol - base_y  # If negative, the line is after the polygon, it's the min idx
            if dist != 0:
                ys.append(y_pol)
                dists.append(dist)
                baselines.append(base_y)
        under_baseline = dists.index(max(dists))
        above_baseline = dists.index(min(dists))
        yield BoundaryBaselineY(
            ys[under_baseline],
            ys[above_baseline],
            baselines[under_baseline],
            baselines[above_baseline],
            dists[under_baseline],
            abs(dists[above_baseline])  # All should be absolute values
        )


def get_max_dist(bby: Union[Tuple[int, int], BoundaryBaselineY], mode: str = "under"):
    if mode == "under":
        return bby.max_under_distance
    else:
        return bby.max_above_distance


def is_outlier(bby: Union[Tuple[int, int], BoundaryBaselineY], score: Score, mode: str = "under"):
    diff_y = get_max_dist(bby, mode=mode)
    if diff_y > score.prct:
        return diff_y, abs(score.prct - diff_y)
    return False


def get_outliers_iqr(lines: List[BoundaryBaselineY], score: Score, attr: str = "under") -> Iterable[Outlier]:
    for line_idx, line in enumerate(lines):
        was_outlier = is_outlier(line, score, mode=attr)
        if was_outlier:
            yield Outlier(line_idx, *was_outlier)


def compute_cuttings(
        boundaries: List[BoundaryBaselineY],
        prct_under: int = 10,
        prct_above: int = 10) -> Tuple[Score, Score]:
    """ This function computes the percentile and median
    """
    above_distances, under_distances = list(zip(
        *[
            (bby.max_above_distance, bby.max_under_distance)
            for bby in boundaries
        ]
    ))
    under_score = Score(
        median(under_distances),
        scipy.stats.scoreatpercentile(under_distances, 100 - prct_under)
    )
    above_score = Score(
        median(above_distances),
        scipy.stats.scoreatpercentile(above_distances, 100 - prct_above)
    )
    return above_score, under_score


def img_to_base64(img: Tuple[Image.Image, Any]) -> str:
    """ Transform an Image to a base64 string """
    img, line = img
    if img is None:
        return "data:image/png;base64,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"
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return (bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str).decode()


def redraw_polygon(
        x_y: Iterable[Tuple[int, int]],
        baseline: numpy.typing.ArrayLike,
        max_under: Optional[Score] = None,
        max_above: Optional[Score] = None,
        margin_under: int = 0,
        margin_above: int = 0
):
    """ Redraw a polygon by removing outliers and replacing them with margin+median"""
    for (x, y) in x_y:
        _, baseline_y = get_closest_points((x, y), baseline)
        # If Y is greater than baseline, then it's part of the bottom of the mask
        if max_under and y > baseline_y and abs(y - baseline_y) > max_under.prct:
            # Because it's under, max_under is added to the baseline
            yield x, ceil(baseline_y + max_under.median) + (margin_under or 0)
        # If Y is smaller than baseline, then it's part of the top of the mask
        elif max_above and y < baseline_y and abs(y - baseline_y) > max_above.prct:
            # Because it's under, max_under is substracted to the baseline
            yield x, ceil(baseline_y - max_above.median) - (margin_above or 0)
        else:
            yield x, y


def apply_prct(
    outliers: Dict[int, Dict[str, Optional[Outlier]]],
    kept_lines: List[Dict[str, Any]],
    under_score: Score,
    above_score: Score,
    margins: Optional[Dict[int, Dict[str, int]]] = None
) -> List[Dict[str, Any]]:
    """ Apply the percentile max distance """
    new_lines = []
    margins = margins or {}
    for line in kept_lines:
        baseline = line["advanced_baseline"]
        details = outliers[line["idx"]]
        if details["above"] and details["under"]:
            scores = dict(max_under=under_score, max_above=above_score)
        elif details["above"]:
            scores = dict(max_above=above_score)
        else:
            scores = dict(max_under=under_score)
        if line["idx"] in margins:
            scores.update(margins[line["idx"]])
        new_lines.append({
            **line,
            "boundary": list(redraw_polygon(line["boundary"], baseline=baseline, **scores))
        })

    return new_lines


def get_closest_points(current_point: Tuple[int, int], baseline: np.typing.ArrayLike) -> Tuple[int, int]:
    """ Get the closest point to CURRENT_POINT in the redrawn baseline"""
    distances = np.linalg.norm(baseline - np.array(current_point), axis=1)
    min_index = np.argmin(distances)
    return tuple(baseline[min_index])


def get_all_points(baseline: List[Tuple[int, int]], number_of_points: int = 50) -> np.typing.ArrayLike:
    """ Get all points for a given baseline"""
    # https://gis.stackexchange.com/questions/263859/fast-way-to-get-all-points-as-integer-of-a-linestring-in-shapely
    ls = LineString(baseline)
    xy = []
    for f in range(0, int(ceil(ls.length)) + 1, ceil(ls.length/number_of_points)):
        p = ls.interpolate(f).coords[0]
        pr = tuple(map(round, p))
        if pr not in xy:
            xy.append(pr)
    return np.array(xy)


def add_advanced_baseline(line: Dict[str, Any]) -> Dict[str, Any]:
    """ Redraw a baseline by adding points """
    line["advanced_baseline"] = get_all_points(line["baseline"])
    return line


def get_medians(boundaries: Iterable[BoundaryBaselineY]) -> Tuple[float, float]:
    """ Get medians of distances for under and above baseline distances
    """
    dists_under, dists_above = list(zip(
        *[
            (bby.max_under_distance, bby.max_above_distance)
            for bby in boundaries
        ]
    ))
    return abs(median(dists_under)), abs(median(dists_above))


def apply_min_max(
        lines: List[Dict[str, Any]],
        under_baseline_max: int,
        above_baseline_max: int) -> Iterable[Dict[str, Any]]:
    """ Apply to a maximum height or a maximum depth to each line
    """
    for line in lines:
        new_boundaries = []
        for (x_pol, y_pol) in line["boundary"]:
            _, base_y = get_closest_points((x_pol, y_pol), line["advanced_baseline"])
            # If dist is negative, that means that the polygone is above the baseline, as Y starts from the top at 0
            #   If a polygon above the line was at 0 and the baseline at 5, -5 would be the dist
            dist = y_pol - base_y
            if dist < 0:
                # As such, we want the maximum value of Y(Polygon) vs Y(Baseline)+Margin,
                #  because if the maximum distance between baseline and top max was 3,
                #  we would want Y(Baseline) - Y(Max-Above-Baseline) = 2 as the top polygon Y
                new_boundaries.append((x_pol, max(0, y_pol, base_y-above_baseline_max)))
            else:
                # On the contrary, we want the minimum value of Y(Polygon) vs Y(Baseline)+Margin,
                #  because if the maximum distance between Y(Baseline) and Y(BottomOfPolygon) was 3,
                #  but Dist(Y(Polygon), Y(Baseline) was 5 where Y(Baseline) = 10
                #  we would want Y(Baseline) + Y(Max-under-Baseline) = 3 as the bottom polygon Y
                new_boundaries.append((x_pol, min(y_pol, base_y+under_baseline_max)))

        yield {**line, "boundary": new_boundaries}


def extract_polygon_wrapper(*args, **kwargs) -> Iterable[Optional[Tuple[Image.Image, Dict[str, Any]]]]:
    try:
        return extract_polygons(*args, **kwargs)
    except KrakenInputException:
        return []


def normalize_points(points, max_y: int, max_x: int):
    return [
        (max(0, min(x, max_x)), max(0, min(y, max_y)))
        for (x, y) in points
    ]


def normalize_lines(lines, max_y: int, max_x: int):
    return [
        {
            **l,
            "baseline": normalize_points(l["baseline"], max_y=max_y, max_x=max_x),
            "boundary": normalize_points(l["boundary"], max_y=max_y, max_x=max_x),
        }
        for l in lines
    ]


def get_boundary_height(boundaries: List[Tuple[int, int]]) -> int:
    _, ys = list(zip(*boundaries))
    return abs(max(ys) - min(ys))


def get_simpler_mask(line: Dict[str, Any]):
    tolerance: float = .15 * (get_boundary_height(line["boundary"]))
    return {**line, "boundary": simplify_mask(line["boundary"], tolerance=tolerance)}


@app.route('/image')
def get_image():
    path = request.args.get("path", None)
    if not path:
        return None
    return send_file(path)


@app.route("/", methods=["GET", "POST"])
def index():
    xmls = get_xml()
    details = {
        page: {**parse_xml(page), "basename": os.path.basename(page)}
        for page in xmls
    }
    return render_template("index.html", details=details, pages=xmls)


@app.route("/approach/margins", methods=["GET", "POST"])
def page_margins():
    xmls = get_xml()
    page = request.args.get("page", xmls[0])
    content = parse_xml(page)
    lines = content["lines"]

    image = Image.open(content["image"])
    lines = normalize_lines(lines, max_y=image.height, max_x=image.width)

    lines = [add_advanced_baseline(line) for line in lines]
    lines = [{"idx": idx, **line} for idx, line in enumerate(lines)]

    under_median, above_median = get_medians(get_min_max_y(lines))
    max_under = ceil(request.args.get("max_under", under_median, type=int)) or under_median
    max_above = ceil(request.args.get("max_above", above_median, type=int)) or above_median

    orig_images = {
        poly[1]["idx"]: img_to_base64(poly)
        for poly in extract_polygon_wrapper(image, {"lines": lines, "type": "baselines"})
    }

    new_lines = list(apply_min_max(
        lines,
        under_baseline_max=max_under,
        above_baseline_max=max_above
    ))
    preview = {
        poly[1]["idx"]: img_to_base64(poly)
        for poly in extract_polygon_wrapper(image, {"lines": new_lines, "type": "baselines"})
    }

    return render_template(
        "baseline_mode.html",
        content={l["idx"]: l for l in lines},
        orig_images=orig_images,
        medians={"above": above_median, "under": under_median},
        form={"max_under": request.args.get("max_above", ""), "max_above": request.args.get("max_above", "")},
        pages=xmls,
        preview=preview,
        current_page=page
    )


@app.route("/approach/stats", methods=["GET", "POST"])
def page_prct():
    xmls = get_xml()
    page = request.args.get("page", xmls[0])

    under_prct = request.args.get("under_prct", 5, type=int)
    above_prct = request.args.get("above_prct", 10, type=int)

    # ToDo: Allow for linetype ignoring
    ignore_zone = request.args.get("ignore_zone", "", type=str)
    content = parse_xml(page)
    lines = content["lines"]

    image = Image.open(content["image"])
    lines = normalize_lines(lines, max_y=image.height, max_x=image.width)

    lines = [add_advanced_baseline(line) for line in lines]
    lines = [{"idx": idx, **line} for idx, line in enumerate(lines)]

    masks_extremes = list(get_min_max_y(lines))
    above_score, under_score = compute_cuttings(masks_extremes, prct_under=under_prct, prct_above=above_prct)
    above_outliers = list(get_outliers_iqr(masks_extremes, above_score, "above"))
    under_outliers = list(get_outliers_iqr(masks_extremes, under_score, "under"))

    outliers: Dict[int, Dict[str, Optional[Outlier]]] = {
        line.idx: {"above": line, "under": None}
        for line in above_outliers
    }
    for line in under_outliers:
        if line.idx not in outliers:
            outliers[line.idx] = {"above": None, "under": None}
        outliers[line.idx]["under"] = line

    image = Image.open(content["image"])

    kept_lines = [
        {"idx": idx, "bby": masks_extremes[idx], **line}
        for idx, line in enumerate(lines)
        if idx in outliers
    ]
    orig_images = {
        poly[1]["idx"]: img_to_base64(poly)
        for poly in extract_polygon_wrapper(image, {"lines": kept_lines, "type": "baselines"})
    }

    # Don't forget it goes from y=0 at the top, so max_y is the bottomest thing !

    margins = {}
    if request.method == "POST":
        margins = {
            int(field_name.split("_")[-1]): {"margin_above": None, "margin_under": None}
            for field_name in request.form
            if field_name.startswith("update_") and request.form[field_name] == "on"
        }
        for field_name in request.form:
            if request.form[field_name] != "0" and field_name.startswith("custom_margin"):
                idx = int(field_name.split("_")[-1])
                if "margin_above" in field_name and request.form.get(f"update_above_{idx}", "off") == "on":
                    margins[idx]["margin_above"] = int(request.form[field_name])
                if "margin_under" in field_name and request.form.get(f"update_under_{idx}", "off") == "on":
                    margins[idx]["margin_under"] = int(request.form[field_name])

    changes = apply_prct(
        outliers,
        kept_lines,
        under_score=under_score,
        above_score=above_score,
        margins=margins
    )
    simply = False
    if request.method == "POST" and request.form.get("simplify"):
        changes = [get_simpler_mask(l) for l in changes]
        simply = True

    preview = {
        poly[1]["idx"]: {"img": img_to_base64(poly), "height": poly[0].height}
        for poly in extract_polygon_wrapper(image, {"lines": changes, "type": "baselines"})
    }
    doc = None
    overwrite = False
    if request.form.get("download", "off") == "on" or request.form.get("overwrite", "off") == "on":
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
        overwrite = request.form.get("overwrite", "off") == "on"
        if overwrite:
            with open(page, "w") as out:
                out.write(ET.tostring(doc, encoding=str))
        if request.form.get("download", "off") == "on":
            return Response(ET.tostring(doc, encoding=str), mimetype="text/xml", headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(page)}"
            })
    return render_template(
        "over_percentile.html",
        content=outliers,
        overwrite=overwrite,
        new_content={l["idx"]: l for l in changes},
        orig_images=orig_images,
        scores={"above": above_score, "under": under_score},
        lines=lines,
        simply=simply,
        pages=xmls,
        current_page=page,
        above_prct=above_prct,
        under_prct=under_prct,
        preview=preview, margins=margins
    )
