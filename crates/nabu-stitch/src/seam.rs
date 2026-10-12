//! Where one photo ends and the next begins.
//!
//! Wherever two photos overlap, the seam between them is placed by a minimum
//! graph cut (Kwatra et al. 2003) whose costs are the colour difference
//! between the photos normalised by the local gradient: cutting is cheap
//! where the photos agree, and cheaper still along edges, where a small
//! mismatch is invisible. People who moved between shots, branches in the
//! wind and parallax from a hand-held camera all end up on one side of the
//! seam instead of appearing twice or half-transparent.

use crate::warp::Warped;

const TERMINAL_COST: f32 = 10_000.0;
const BAD_REGION: f32 = 1_000.0;
const EPS: f32 = 1.0 / 255.0;
const GAP: i64 = 10;

/// Boykov–Kolmogorov maximum flow ("An Experimental Comparison of
/// Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision", 2004).
/// On grid graphs like these it is many times faster than the general
/// algorithms, because the two search trees are reused between augmentations
/// instead of being rebuilt.
struct Flow {
    // Nodes.
    first: Vec<u32>,
    parent: Vec<u32>,
    is_sink: Vec<bool>,
    tr_cap: Vec<f32>,
    ts: Vec<u32>,
    dist: Vec<u32>,
    queued: Vec<bool>,
    // Arcs, stored in sister pairs (a ^ 1 is the reverse arc).
    head: Vec<u32>,
    next: Vec<u32>,
    r_cap: Vec<f32>,
}

const NONE: u32 = u32::MAX;
const TERMINAL: u32 = u32::MAX - 1;
const ORPHAN: u32 = u32::MAX - 2;

impl Flow {
    fn new(nodes: usize, arcs_hint: usize) -> Self {
        Self {
            first: vec![NONE; nodes],
            parent: vec![NONE; nodes],
            is_sink: vec![false; nodes],
            tr_cap: vec![0.0; nodes],
            ts: vec![0; nodes],
            dist: vec![0; nodes],
            queued: vec![false; nodes],
            head: Vec::with_capacity(arcs_hint),
            next: Vec::with_capacity(arcs_hint),
            r_cap: Vec::with_capacity(arcs_hint),
        }
    }

    /// Capacity from the source (`source`) and to the sink (`sink`).
    fn terminal(&mut self, v: usize, source: f32, sink: f32) {
        self.tr_cap[v] += source - sink;
    }

    /// Arc u→v with capacity `c`, and v→u with capacity `rc`.
    fn add(&mut self, u: usize, v: usize, c: f32, rc: f32) {
        for (from, to, cap) in [(u, v, c), (v, u, rc)] {
            self.head.push(to as u32);
            self.r_cap.push(cap);
            self.next.push(self.first[from]);
            self.first[from] = (self.head.len() - 1) as u32;
        }
    }

    fn run(&mut self) {
        let n = self.first.len();
        let mut active: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let mut orphans: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        for v in 0..n {
            if self.tr_cap[v] > 0.0 {
                self.is_sink[v] = false;
                self.parent[v] = TERMINAL;
                self.dist[v] = 1;
                self.queued[v] = true;
                active.push_back(v as u32);
            } else if self.tr_cap[v] < 0.0 {
                self.is_sink[v] = true;
                self.parent[v] = TERMINAL;
                self.dist[v] = 1;
                self.queued[v] = true;
                active.push_back(v as u32);
            }
        }
        let mut time = 0u32;
        let mut current: Option<usize> = None;

        loop {
            // Next active node (the current one stays while it still finds paths).
            let i = match current.take() {
                Some(i) if self.parent[i] != NONE => i,
                _ => loop {
                    match active.pop_front() {
                        None => return,
                        Some(v) => {
                            let v = v as usize;
                            self.queued[v] = false;
                            if self.parent[v] != NONE {
                                break v;
                            }
                        }
                    }
                },
            };

            // Grow.
            let mut middle = NONE;
            if !self.is_sink[i] {
                let mut a = self.first[i];
                while a != NONE {
                    let au = a as usize;
                    if self.r_cap[au] > 0.0 {
                        let j = self.head[au] as usize;
                        if self.parent[j] == NONE {
                            self.is_sink[j] = false;
                            self.parent[j] = a ^ 1;
                            self.ts[j] = self.ts[i];
                            self.dist[j] = self.dist[i] + 1;
                            if !self.queued[j] {
                                self.queued[j] = true;
                                active.push_back(j as u32);
                            }
                        } else if self.is_sink[j] {
                            middle = a;
                            break;
                        } else if self.ts[j] <= self.ts[i] && self.dist[j] > self.dist[i] {
                            self.parent[j] = a ^ 1;
                            self.ts[j] = self.ts[i];
                            self.dist[j] = self.dist[i] + 1;
                        }
                    }
                    a = self.next[au];
                }
            } else {
                let mut a = self.first[i];
                while a != NONE {
                    let au = a as usize;
                    if self.r_cap[au ^ 1] > 0.0 {
                        let j = self.head[au] as usize;
                        if self.parent[j] == NONE {
                            self.is_sink[j] = true;
                            self.parent[j] = a ^ 1;
                            self.ts[j] = self.ts[i];
                            self.dist[j] = self.dist[i] + 1;
                            if !self.queued[j] {
                                self.queued[j] = true;
                                active.push_back(j as u32);
                            }
                        } else if !self.is_sink[j] {
                            middle = a ^ 1;
                            break;
                        } else if self.ts[j] <= self.ts[i] && self.dist[j] > self.dist[i] {
                            self.parent[j] = a ^ 1;
                            self.ts[j] = self.ts[i];
                            self.dist[j] = self.dist[i] + 1;
                        }
                    }
                    a = self.next[au];
                }
            }
            time += 1;
            if middle == NONE {
                continue;
            }
            current = Some(i);

            // Augment.
            let m = middle as usize;
            let mut bottleneck = self.r_cap[m];
            let mut v = self.head[m ^ 1] as usize;
            loop {
                let a = self.parent[v];
                if a == TERMINAL {
                    break;
                }
                bottleneck = bottleneck.min(self.r_cap[a as usize ^ 1]);
                v = self.head[a as usize] as usize;
            }
            bottleneck = bottleneck.min(self.tr_cap[v]);
            let mut v = self.head[m] as usize;
            loop {
                let a = self.parent[v];
                if a == TERMINAL {
                    break;
                }
                bottleneck = bottleneck.min(self.r_cap[a as usize]);
                v = self.head[a as usize] as usize;
            }
            bottleneck = bottleneck.min(-self.tr_cap[v]);

            self.r_cap[m ^ 1] += bottleneck;
            self.r_cap[m] -= bottleneck;
            let mut v = self.head[m ^ 1] as usize;
            loop {
                let a = self.parent[v];
                if a == TERMINAL {
                    break;
                }
                let au = a as usize;
                self.r_cap[au] += bottleneck;
                self.r_cap[au ^ 1] -= bottleneck;
                if self.r_cap[au ^ 1] <= 0.0 {
                    self.parent[v] = ORPHAN;
                    orphans.push_front(v as u32);
                }
                v = self.head[au] as usize;
            }
            self.tr_cap[v] -= bottleneck;
            if self.tr_cap[v] <= 0.0 {
                self.parent[v] = ORPHAN;
                orphans.push_front(v as u32);
            }
            let mut v = self.head[m] as usize;
            loop {
                let a = self.parent[v];
                if a == TERMINAL || a == ORPHAN {
                    break;
                }
                let au = a as usize;
                self.r_cap[au ^ 1] += bottleneck;
                self.r_cap[au] -= bottleneck;
                if self.r_cap[au] <= 0.0 {
                    self.parent[v] = ORPHAN;
                    orphans.push_front(v as u32);
                }
                v = self.head[au] as usize;
            }
            if self.parent[v] == TERMINAL {
                self.tr_cap[v] += bottleneck;
                if self.tr_cap[v] >= 0.0 {
                    self.parent[v] = ORPHAN;
                    orphans.push_front(v as u32);
                }
            }

            // Adopt orphans.
            while let Some(o) = orphans.pop_front() {
                let o = o as usize;
                self.adopt(o, time, &mut active, &mut orphans);
            }
        }
    }

    fn adopt(
        &mut self,
        i: usize,
        time: u32,
        active: &mut std::collections::VecDeque<u32>,
        orphans: &mut std::collections::VecDeque<u32>,
    ) {
        let sink = self.is_sink[i];
        let mut best = NONE;
        let mut best_d = u32::MAX;
        let mut a0 = self.first[i];
        while a0 != NONE {
            let a0u = a0 as usize;
            // Residual capacity in the direction of the tree's flow.
            let cap = if sink { self.r_cap[a0u] } else { self.r_cap[a0u ^ 1] };
            let j0 = self.head[a0u] as usize;
            if cap > 0.0 && self.is_sink[j0] == sink && self.parent[j0] != NONE {
                // Does j0 still lead to the terminal?
                let mut j = j0;
                let mut d = 0u32;
                let valid = loop {
                    if self.ts[j] == time {
                        d += self.dist[j];
                        break true;
                    }
                    let a = self.parent[j];
                    d += 1;
                    if a == TERMINAL {
                        self.ts[j] = time;
                        self.dist[j] = 1;
                        break true;
                    }
                    if a == ORPHAN {
                        break false;
                    }
                    j = self.head[a as usize] as usize;
                };
                if valid {
                    if d < best_d {
                        best = a0;
                        best_d = d;
                    }
                    let mut j = j0;
                    let mut dd = d;
                    while self.ts[j] != time {
                        self.ts[j] = time;
                        self.dist[j] = dd;
                        dd -= 1;
                        j = self.head[self.parent[j] as usize] as usize;
                    }
                }
            }
            a0 = self.next[a0u];
        }

        if best != NONE {
            self.parent[i] = best;
            self.ts[i] = time;
            self.dist[i] = best_d + 1;
            return;
        }
        self.parent[i] = NONE;
        let mut a0 = self.first[i];
        while a0 != NONE {
            let a0u = a0 as usize;
            let j = self.head[a0u] as usize;
            let a = self.parent[j];
            if self.is_sink[j] == sink && a != NONE {
                let cap = if sink { self.r_cap[a0u] } else { self.r_cap[a0u ^ 1] };
                if cap > 0.0 && !self.queued[j] {
                    self.queued[j] = true;
                    active.push_back(j as u32);
                }
                if a != TERMINAL && a != ORPHAN && self.head[a as usize] as usize == i {
                    self.parent[j] = ORPHAN;
                    orphans.push_back(j as u32);
                }
            }
            a0 = self.next[a0u];
        }
    }

    /// True for nodes on the source side of the cut (free nodes included).
    fn source_side(&self, v: usize) -> bool {
        self.parent[v] == NONE || !self.is_sink[v]
    }
}

fn gray(p: [f32; 3]) -> f32 {
    0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
}

/// Sobel magnitudes of a photo over a window, in x and y.
fn gradients(img: &[[f32; 3]], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    let at = |x: isize, y: isize| -> f32 {
        let x = x.clamp(0, w as isize - 1) as usize;
        let y = y.clamp(0, h as isize - 1) as usize;
        gray(img[y * w + x])
    };
    for y in 0..h as isize {
        for x in 0..w as isize {
            let dx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            let dy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1))
                - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
            gx[y as usize * w + x as usize] = dx.abs();
            gy[y as usize * w + x as usize] = dy.abs();
        }
    }
    (gx, gy)
}

/// Cuts the masks of every overlapping pair so each pixel belongs to one
/// photo. Masks are changed in place.
pub fn find_seams(warped: &mut [Warped]) {
    let n = warped.len();
    for i in 0..n {
        for j in i + 1..n {
            seam_pair(warped, i, j);
        }
    }
}

fn seam_pair(warped: &mut [Warped], i: usize, j: usize) {
    let Some(overlap) = warped[i].rect.intersect(&warped[j].rect) else { return };
    let x0 = overlap.x - GAP;
    let y0 = overlap.y - GAP;
    let w = overlap.width + 2 * GAP as usize;
    let h = overlap.height + 2 * GAP as usize;

    // Copy both photos and masks into the window, empty outside them.
    let window = |img: &Warped| -> (Vec<[f32; 3]>, Vec<bool>) {
        let mut rgb = vec![[0.0f32; 3]; w * h];
        let mut mask = vec![false; w * h];
        for y in 0..h as i64 {
            let sy = y0 + y - img.rect.y;
            if sy < 0 || sy >= img.rect.height as i64 {
                continue;
            }
            for x in 0..w as i64 {
                let sx = x0 + x - img.rect.x;
                if sx < 0 || sx >= img.rect.width as i64 {
                    continue;
                }
                let k = sy as usize * img.rect.width + sx as usize;
                let o = y as usize * w + x as usize;
                rgb[o] = img.rgb[k];
                mask[o] = img.mask[k] != 0;
            }
        }
        (rgb, mask)
    };
    let (img1, mask1) = window(&warped[i]);
    let (img2, mask2) = window(&warped[j]);
    if !mask1.iter().zip(&mask2).any(|(a, b)| *a && *b) {
        return;
    }
    let (dx1, dy1) = gradients(&img1, w, h);
    let (dx2, dy2) = gradients(&img2, w, h);

    let diff = |k: usize| -> f32 {
        let (a, b) = (img1[k], img2[k]);
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };

    let mut flow = Flow::new(w * h, w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let v = y * w + x;
            let cs = if mask1[v] { TERMINAL_COST } else { 0.0 };
            let ct = if mask2[v] { TERMINAL_COST } else { 0.0 };
            flow.terminal(v, cs, ct);
            if x + 1 < w {
                let u = v + 1;
                let mut weight = (diff(v) + diff(u) + EPS) / (dx1[v] + dx1[u] + dx2[v] + dx2[u] + EPS);
                if !mask1[v] || !mask1[u] || !mask2[v] || !mask2[u] {
                    weight += BAD_REGION;
                }
                flow.add(v, u, weight, weight);
            }
            if y + 1 < h {
                let u = v + w;
                let mut weight = (diff(v) + diff(u) + EPS) / (dy1[v] + dy1[u] + dy2[v] + dy2[u] + EPS);
                if !mask1[v] || !mask1[u] || !mask2[v] || !mask2[u] {
                    weight += BAD_REGION;
                }
                flow.add(v, u, weight, weight);
            }
        }
    }
    let started = std::time::Instant::now();
    flow.run();
    crate::debug(&format!("seam {i}-{j}: {w}x{h} window, flow {:.2}s", started.elapsed().as_secs_f64()));

    // Apply the cut inside the overlap (not the gap around it).
    let (r1, r2) = (warped[i].rect, warped[j].rect);
    for y in GAP as usize..GAP as usize + overlap.height {
        for x in GAP as usize..GAP as usize + overlap.width {
            let v = y * w + x;
            let (px, py) = (x0 + x as i64, y0 + y as i64);
            let k1 = (py - r1.y) as usize * r1.width + (px - r1.x) as usize;
            let k2 = (py - r2.y) as usize * r2.width + (px - r2.x) as usize;
            if flow.source_side(v) {
                if mask1[v] {
                    warped[j].mask[k2] = 0;
                }
            } else if mask2[v] {
                warped[i].mask[k1] = 0;
            }
        }
    }
}
