#!/usr/bin/env python3
"""Known answers for teaser_map and for the time stamps of render_teaser: no dataset, no map file, no Blender.

Usage: python research/src/test_teaser_map.py

The fit and the time stamps run with numpy alone. The geometry needs shapely >= 2.1 and is skipped with a note
without it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "figures"))
from render_teaser import BODY_AHEAD, source_of, thin, vehicle_marks  # noqa: E402
from teaser_map import MOTION_HZ, SCENE_HZ, ambiguous, fit_window, parse_path, rigid, window_frames  # noqa: E402


def rotation(deg, reflect=False):
    a = np.radians(deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return r @ np.diag([1, -1]) if reflect else r


def test_fit():
    rng = np.random.default_rng(0)
    # rigid: a known rotation, with and without a flip of handedness, is recovered exactly
    a = rng.normal(size=(50, 2)) * 10
    for reflect in (False, True):
        r, t = rotation(37, reflect), np.array([3.0, -8.0])
        r_hat, t_hat = rigid(a, a @ r.T + t, reflect)
        assert np.allclose(r_hat, r) and np.allclose(t_hat, t), (reflect, r_hat, t_hat)
        assert np.isclose(np.linalg.det(r_hat), -1 if reflect else 1)

    # fit_window: a curved vehicle path and a walking pedestrian, window cut 2.3 s into the scene, teaser frame mirrored
    frames = 200
    time = np.arange(frames) / SCENE_HZ
    ego_world = np.stack([100 + 6 * time, 50 + 0.15 * time ** 2], 1)
    ped_world = np.stack([120 + 0.2 * time, 58 - 1.2 * time], 1)
    start, length = 2.3, 196
    t = start + np.arange(length) / MOTION_HZ
    at = lambda track: np.stack([np.interp(t, time, track[:, k]) for k in range(2)], 1)
    r, shift = rotation(-64, reflect=True), np.array([-5.0, 12.0])
    fit = fit_window(ego_world, ped_world, at(ego_world) @ r.T + shift, at(ped_world) @ r.T + shift)
    assert abs(fit["start_s"] - start) < 1e-6, fit["start_s"]
    assert fit["reflect"] and fit["rms"] < 1e-6, fit
    assert np.allclose(fit["R"], r, atol=1e-6) and np.allclose(fit["t"], shift, atol=1e-5)
    assert fit["rms_other_handedness"] > 1.0, "the wrong handedness must fit visibly worse"
    assert fit["rms_elsewhere"] > 0.05 and ambiguous(fit) is None, fit

    # a placement that is not unique is named as such: a vehicle at constant speed on a straight line beside a
    # standing pedestrian fits every window, and a pedestrian standing on the vehicle's line fits mirrored as well
    straight = np.stack([100 + 6 * time, np.full(frames, 50.0)], 1)
    standing = np.tile([[130.0, 53.0]], (frames, 1))
    sliding = fit_window(straight, standing + (straight - straight[0]), at(straight), at(standing + (straight - straight[0])))
    assert "window" in ambiguous(sliding), ambiguous(sliding)
    on_line = np.tile([[130.0, 50.0]], (frames, 1))
    mirrored = fit_window(straight, on_line, at(straight), at(on_line))
    assert "mirrored" in ambiguous(mirrored), ambiguous(mirrored)

    # frames of the window and the path after it: row k of the path is k frames after `last`, cut at a tracking gap
    track = np.stack([np.arange(200.0), np.zeros(200)], 1)
    first, last, tail = window_frames(1.15, 196, track)  # 1.15 s to 10.9 s
    assert (first, last) == (11, 109) and len(tail) == 91 and tail[0, 0] == 109, (first, last, len(tail))
    first, last, tail = window_frames(0.0, 196, track)  # 0 s to 9.75 s: the first frame at or after the end is 98
    assert (first, last) == (0, 98) and tail[0, 0] == 98
    gap = track.copy(); gap[120:126] = np.nan
    assert len(window_frames(1.15, 196, gap)[2]) == 11, "the path stops at the gap instead of jumping over it"
    lost = track.copy(); lost[109:] = np.nan
    assert len(window_frames(1.15, 196, lost)[2]) == 0

    # a pedestrian who is not tracked at the start: only windows inside the track are tried
    late = ped_world.copy(); late[:30] = np.nan
    fit = fit_window(ego_world, late, at(ego_world) @ r.T + shift, at(ped_world) @ r.T + shift)
    assert fit["start_s"] >= 3.0 - 1e-9 and fit["rms"] > 1e-3, fit
    never = np.full_like(ped_world, np.nan)
    try:
        fit_window(ego_world, never, at(ego_world), at(ped_world))
        raise AssertionError("an untracked pedestrian must be refused")
    except ValueError:
        pass

    assert parse_path("nuscenes/train/146_11.json") == ("nuscenes", "146", "11")
    assert parse_path("/data/diffusion/waymo/train/093_60.json") == ("waymo", "093", "60")
    try:
        parse_path("waymo/train/M093_60.json")
        raise AssertionError("a mirrored copy has no map of its own")
    except ValueError:
        pass


def test_time_stamps():
    """Which time stamps of the time-lapse are drawn, and which span of time a drawn one stands for."""
    frames = [0, 48, 96, 144, 195]
    along = lambda xs: np.stack([np.interp(np.arange(196), frames, xs), np.zeros(196), np.zeros(196)], 1)

    # every time stamp far enough from the next: all drawn, each for itself
    assert thin(frames, along([0, 5, 10, 15, 20]), 3.5, .5) == {f: f for f in frames}
    # a vehicle that waits until 4.8 s and drives off: one mark for 0 to 4.8 s
    assert thin(frames, along([0, 0, 0, 5, 12]), 3.5, .5) == {96: 0, 144: 144, 195: 195}
    # one that stands for the whole window, and one that arrives and then stands
    assert thin(frames, along([0, 0, 0, 0, 0]), 3.5, .5) == {195: 0}
    assert thin(frames, along([0, 5, 10, 10, 10]), 3.5, .5) == {0: 0, 48: 48, 195: 96}
    # a vehicle that creeps 3 m is drawn once, and the mark does not claim that it stood there
    assert thin(frames, along([0, 1, 2, 3, 4]), 3.5, .5) == {0: 0, 195: 195}
    assert thin(frames, along([0, 0, 0, 3, 3.2]), 3.5, .5) == {195: 144}, "the span starts where the standing starts"
    # away and back between two time stamps is not standing: the frames in between count, not only the stamps
    there_and_back = along([0, 0, 0, 0, 0])
    there_and_back[110:130, 0] = 2.0
    assert thin(frames, there_and_back, 3.5, .5) == {195: 144}
    # the vehicle's own time stamps, as --car-times gives them
    assert thin([0, 96, 195], along([0, 0, 0, 5, 12]), 3.5, .5) == {96: 0, 195: 195}
    assert thin([195], along([0, 5, 10, 15, 20]), 3.5, .5) == {195: 195}
    # stands, steps 2 m aside for a while, stands again: the last mark does not reach back over the step aside
    assert thin(frames, along([0, 2, 0, 0, 0]), 3.5, .5) == {195: 96}
    # the limits count as far enough and as still, the order and repeats of the time stamps do not matter
    assert thin(frames, along([0, 3.5, 7, 10.5, 14]), 3.5, .5) == {f: f for f in frames}
    assert thin([0, 195], along([0.5, 0.5, 0.5, 0.5, 0]), 3.5, .5) == {195: 0}
    assert thin([195, 0, 96, 96, 0], along([0, 0, 0, 5, 12]), 3.5, .5) == {96: 0, 195: 195}
    assert thin(frames, along([0, 0, 0, 5, 12])[:, :2], 3.5, .5) == {96: 0, 144: 144, 195: 195}

    # where the vehicle body goes: its middle lies ahead of the point whose path the data holds, along the heading
    drive = np.stack([np.zeros(196), -np.interp(np.arange(196), frames, [0, 5, 10, 15, 20]), np.full(196, .04)], 1)
    marks = vehicle_marks(frames, drive, .45)  # driving along -y
    assert [m[:2] for m in marks] == [(f, f) for f in frames]
    assert all(np.isclose(m[2], -np.pi / 2) and np.allclose(m[3], drive[m[0], :2] + [0, -.45]) for m in marks), marks
    # a vehicle that waits and then drives off along +x: one mark for the wait, turned the way it sets off
    wait = vehicle_marks([0, 96, 195], along([0, 0, 0, 5, 12]), 1.45)
    assert [m[:2] for m in wait] == [(96, 0), (195, 195)] and np.isclose(wait[0][2], 0) and np.allclose(wait[0][3], [1.45, 0])
    assert np.allclose(wait[1][3], [13.45, 0]) and np.allclose(vehicle_marks([195], along([0, 0, 0, 0, 0]), .45)[0][3], [.45, 0])
    assert all(0 <= a < 2 for a in BODY_AHEAD.values()), "the offset is a distance in metres within the vehicle"

    assert source_of("nuscenes/train/260_16.json") == "nuscenes" and source_of(np.array("/data/x/waymo/train/093_60.json")) == "waymo"
    assert source_of(b"nuscenes/train/260_16.json") == "nuscenes" and source_of("260_16.json") == "" and source_of("a/b.json") == ""


def test_geometry():
    import shapely
    from shapely.geometry import LineString, Point, Polygon, box
    from shapely.ops import unary_union

    from teaser_map import build_row, dashes, marking, road_from_edges, stripes, triangles

    # zebra: bars across the short side of a 10 m x 3 m crosswalk, stripe and gap of 0.6 m, all inside it
    walk = box(0, 0, 10, 3)
    bars = stripes(walk, stripe=0.6)
    assert len(bars) == 8, len(bars)
    assert all(np.isclose(b.area, 0.6 * 3) and walk.buffer(1e-9).contains(b) for b in bars)
    assert all(b.bounds[3] - b.bounds[1] > b.bounds[2] - b.bounds[0] for b in bars), "bars run across the short side"
    turned = stripes(shapely.affinity.rotate(walk, 90, origin=(0, 0)), stripe=0.6)
    assert len(turned) == 8 and all(b.bounds[2] - b.bounds[0] > b.bounds[3] - b.bounds[1] for b in turned)

    # markings: a solid line is one strip, a broken one is cut into dashes, a double one is two strips
    line = LineString([(0, 0), (30, 0)])
    assert len(marking(line, "SOLID_WHITE", width=0.1)) == 1 and np.isclose(marking(line, "SOLID_WHITE", width=0.1)[0].area, 3.0)
    assert len(dashes(line, dash=3, gap=4.5)) == 4 and len(marking(line, "TYPE_BROKEN_SINGLE_YELLOW")) == 4
    double = marking(line, "TYPE_SOLID_DOUBLE_YELLOW", width=0.1)
    assert len(double) == 2 and not double[0].intersects(double[1])

    # road from its edges: two edges 8 m apart with a lane centre between them, in a 40 m clip
    clip = box(-20, -20, 20, 20)
    edges = [LineString([(-4, -30), (-4, 30)]), LineString([(4, -30), (4, 30)])]
    road = road_from_edges(edges, [LineString([(0, -30), (0, 30)])], clip)
    assert np.isclose(road.area, 8 * 40), road.area
    assert road.contains(Point(0, 0)) and not road.contains(Point(10, 0))
    assert road_from_edges(edges, [], clip).is_empty, "no lane, no road"
    try:  # an edge that stops short lets the surface run around its end: refused, not drawn
        road_from_edges([LineString([(-16, -30), (-16, 30)]), LineString([(-8, -30), (-8, 5)])], [LineString([(-12, -30), (-12, 30)])], clip)
        raise AssertionError("a leaking road must be refused")
    except SystemExit:
        pass

    # triangles cover the polygon exactly and leave a hole open
    ring = Polygon(box(0, 0, 10, 10).exterior.coords, [box(4, 4, 6, 6).exterior.coords])
    tris = triangles(ring)
    covered = unary_union([Polygon(t) for t in tris])
    assert np.isclose(covered.area, 96) and not covered.contains(Point(5, 5)) and all(len(t) == 3 for t in tris)
    assert triangles(Polygon()) == []

    # one panel end to end: a straight road with a crosswalk, a centre line and a stop sign, moved by a known transform
    r, shift = rotation(90), np.array([100.0, 0.0])
    to_teaser = lambda pts: np.asarray(pts, dtype=float)[:, :2] @ r.T + shift
    raw = dict(layers=dict(
        road_edge=[dict(line=[(-30, -4), (30, -4)]), dict(line=[(-30, 4), (30, 4)])], lane=[dict(line=[(-30, 0), (30, 0)])],
        road_line=[dict(line=[(-30, 0), (30, 0)], kind="TYPE_SOLID_SINGLE_WHITE")],
        crosswalk=[dict(exterior=[(10, -4), (13, -4), (13, 4), (10, 4)])], stop_sign=[dict(position=[9, -5])]))
    row = build_row(raw, to_teaser, (80, -20, 120, 20), "waymo")
    assert row["road"].contains(Point(100, 0)) and not row["road"].contains(Point(110, 0))  # the road now runs along z at x = 100
    assert np.allclose(row["signs"], [[105, 9]])
    zebra = unary_union([Polygon(t) for t in row["triangles"]["crosswalk"]])
    assert 0 < zebra.area < 3 * 8 and zebra.within(box(96, 10, 104, 13).buffer(1e-6))
    centre = unary_union([Polygon(t) for t in row["triangles"]["marking_white"]])
    assert centre.intersects(Point(100, 0).buffer(.2)) and not centre.intersects(zebra.buffer(-1e-6)), "the centre line stops at the crosswalk"
    assert row["triangles"]["marking_yellow"] == []

    # the same panel through a mirroring transform, as the real teaser frame has: the sign changes side
    flip = rotation(90, reflect=True)
    mirror = build_row(raw, lambda pts: np.asarray(pts, dtype=float)[:, :2] @ flip.T + shift, (80, -20, 120, 20), "waymo")
    assert np.allclose(mirror["signs"], [[95, 9]]) and mirror["road"].contains(Point(100, 0))

    # nuScenes layers: a road polygon with an island, a walkway that overlaps the road, a dashed double divider
    nus = dict(layers=dict(
        drivable_area=[dict(exterior=[(-30, -4), (30, -4), (30, 4), (-30, 4)], holes=[[(-2, -1), (2, -1), (2, 1), (-2, 1)]])],
        walkway=[dict(exterior=[(-30, 3), (30, 3), (30, 7), (-30, 7)], holes=[])], ped_crossing=[],
        road_divider=[dict(line=[(-30, -2), (30, -2)], kind="SOLID_WHITE")], lane_divider=[dict(line=[(-30, 2), (30, 2)], kind="DOUBLE_DASHED_WHITE")]))
    row = build_row(nus, to_teaser, (80, -20, 120, 20), "nuscenes")
    assert not row["road"].contains(Point(100, 0)) and row["road"].contains(Point(102.5, 5)), "the island stays open"
    walk = unary_union([Polygon(t) for t in row["triangles"]["walkway"]])
    assert np.isclose(walk.area, 3 * 40), "the walkway is drawn only where it is not road"
    assert row["triangles"]["crosswalk"] == [] and row["signs"] == [] and len(row["triangles"]["marking_white"]) > 8

    # nothing mapped inside the section: empty layers, no error
    empty = build_row(dict(layers=dict(road_edge=[], lane=[], road_line=[], crosswalk=[], stop_sign=[])), to_teaser, (80, -20, 120, 20), "waymo")
    assert all(v == [] for v in empty["triangles"].values()) and empty["signs"] == []


def main():
    test_fit()
    test_time_stamps()
    try:
        import shapely
        ok = hasattr(shapely, "constrained_delaunay_triangles")
    except ImportError:
        ok = False
    if ok:
        test_geometry()
        print("all checks passed")
    else:
        print("fit and time stamp checks passed; geometry checks skipped (shapely >= 2.1 not installed in this environment)")


if __name__ == "__main__":
    main()
