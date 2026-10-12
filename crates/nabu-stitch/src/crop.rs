//! The largest upright rectangle that holds only picture, for cropping away
//! the ragged transparent border a projection leaves.

/// Largest axis-aligned rectangle of `true` cells, as (x, y, width, height).
/// Classic maximal-rectangle-in-histogram sweep, linear in the pixel count.
pub fn largest_rectangle(mask: &[bool], width: usize, height: usize) -> Option<(usize, usize, usize, usize)> {
    let mut heights = vec![0usize; width];
    let mut best = (0usize, 0usize, 0usize, 0usize);
    let mut best_area = 0usize;
    let mut stack: Vec<usize> = Vec::with_capacity(width + 1);

    for y in 0..height {
        for x in 0..width {
            heights[x] = if mask[y * width + x] { heights[x] + 1 } else { 0 };
        }
        stack.clear();
        for x in 0..=width {
            let h = if x < width { heights[x] } else { 0 };
            while let Some(&top) = stack.last() {
                if heights[top] < h {
                    break;
                }
                stack.pop();
                let bar = heights[top];
                let left = stack.last().map(|&l| l + 1).unwrap_or(0);
                let area = bar * (x - left);
                if area > best_area {
                    best_area = area;
                    best = (left, y + 1 - bar, x - left, bar);
                }
            }
            stack.push(x);
        }
    }
    (best_area > 0).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_inner_rectangle() {
        // A plus-shaped region: the 3×5 vertical bar beats the 7×2 one.
        let (w, h) = (7, 5);
        let mut m = vec![false; w * h];
        for y in 0..h {
            for x in 2..5 {
                m[y * w + x] = true;
            }
        }
        for x in 0..w {
            m[2 * w + x] = true;
            m[3 * w + x] = true;
        }
        assert_eq!(largest_rectangle(&m, w, h), Some((2, 0, 3, 5)));
    }
}
