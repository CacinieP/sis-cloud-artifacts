# Side-pinning cores: records vs their exact-side alternates (10-18)

Question (idea 2 of the 10-18 review): are records "a rigid core + loose filler", with the 123 exact-side alternates of
the shove runs (`runs/shove/ties1018`, 46 n) rearranging only the filler?  `coremap.py`, ~0 CPU (LPs, seconds).

* **Dual cores (one LP vertex, `--mode dual`; core1)**: 24-34 squares at n 200-300, ~2s (about one wall-to-wall chain
  per direction).  Record vs alternate overlap is low (J 0.02-0.3) even when nearly every square is pose-identical
  (233 / _2: 24 of 25 record core squares identical in the alternate, 5 carry force there): the force solution is
  degenerate, so a single vertex is not a fingerprint.
* **Locked sets (`fixlp.mobility`: no first-order motion at the fixed side over 12 random-objective LPs; = the union of
  all force supports; core2)**: median 57 % of a record's squares are locked (alternates 50 %).  Between record and
  alternate (best of 8 symmetries + Hungarian), median 49 % of all squares change pose but only 16.5 % of the record's
  locked squares do (median 20 squares); Jaccard of locked-and-identical median 0.68 (quartiles 0.39-0.82).  Lowest
  overlap where the alternate is a different arrangement altogether (257, 292, 170: md 0.15-0.27).
* Reading: yes, mostly.  Equal-side alternates rearrange the loose half; the locked half (the side-pinning skeleton) is
  largely shared, but not fixed (~20 locked squares per alternate do move, probably as rigid blocks).  For idea 2 this
  only halves the target (locked ~ half the packing), and the minimal force chains (~2s) are not unique.  The test that
  would decide whether skeleton-targeted moves are worth building: kicks restricted to locked vs loose vs all squares
  from the record, same quench, compare how often the side changes (not ties) and how far.

## What the exact-side alternates are (10-18; Evan: "a common deterministic program we're finding inefficiently, like
149, or different packings with a related core and different fill?"; `altshape.py`, figures in the session scratchpad)
132 alternates (ties1018 + DUP3's refined ties), aligned to their record:
* 55 "loose only": >= 80 % of the squares that differ were loose (mobility > 0) in the record: the same packing with
  rattlers / slack squares elsewhere.  Not distinct in any useful sense.
* 77 "mixed": locked squares differ too.  Drawn (257, 153, 174, 207, 298, 125): every one is a discrete rearrangement
  of the record's own pieces: where a 45-degree strip jogs or breaks (125, 174), a strip vs a diamond block of 45-degree
  squares (257), which diamond cells are vacant (298), which wall row / column carries the fractional slack and where
  the upright blocks sit along the strip (153, 207).  Same piece set, same side, different placement: the 149 situation
  (`search/goebel_alt`), i.e. a combinatorial family a placement model / ILP (Ellsworth's 45-degree enumerator, goebel_alt)
  would list in full.  No rigid-translation test catches it (blocks shifted by whole lattice units match onto their
  neighbours), but the pictures are unambiguous at these n.
* Reading: the plateau is the record's combinatorial family; shove / delete-L / duplicate-half sample it.  Beating
  these records needs a different piece set (a construction change), not a rearrangement.

## The s(174) bend (10-18; Evan: "that wonky bend is interesting ... can we place it anywhere? move pieces locally?")
`bend174.py`, `stripslack.py` (~1 CPU-h).
* In the record (Göbel, 13 + 1/sqrt 2) all 16 strip squares are loose (fixlp.mobility >= 0.2); the upright staircases
  carry the side.  The alternates' bend squares (39 / 67 degrees, slightly tipped uprights at staircase notches) are
  loose too: a bend is the strip's slack parked somewhere.
* E1 (insert one square at each of the 17-20 strip gaps, record + 3 alternates, r0 0.1 / 0.2, box held / free, 608
  runs): best 175 = 13.8267 (s(175) record 13.7641743; 174's side 13.7071).  No room for another square.
* E2 (melt a window r 1.5 / 2.5 around each strip square, re-insert at random, box held, 432 runs): 8 return to the
  record side, every one with a bend (5-9 non-45 squares), at windows k = 0, 1, 3, 6, 7, 14 (lower left .. upper right);
  0 below; median outcome +0.20.  So bends sit almost anywhere along the strip, at zero cost and zero gain.
* Consolidated slack (LP: slide range of the top strip square along (1, 1), side fixed, all else free) in the
  13 + 1/sqrt 2 family: 27 0.243, 38 0.657, 52 0.071, 67 0.328, 104 0.157, 125 0.400, 149 0.142, 174 0.328, 201
  0.571, 231 0.228, 262 0.400, 296 0.157.  Values of the form a + b sqrt 2 (174: 2 sqrt 2 - 5/2), set by where the
  strip ends meet the staircases; no growth with k, never near 1.
* Reading: the bend is slack spreading, not structure.  Nothing to collect at these n.
