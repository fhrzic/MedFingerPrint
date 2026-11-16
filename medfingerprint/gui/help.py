"""In-app help: feature explanations, "why flagged" sentences and help pages."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QToolButton, QVBoxLayout

from ..core.explain import (  # noqa: F401  (re-exported for the GUI)
    AXES_2D, AXES_3D, FEATURES, explain_flag, feature_title, feature_tooltip, obvious_problems,
    parse_feature, severity, verdict,
)


# ---------------------------------------------------------------------- help pages
CSS = "<style>body{font-size:10.5pt} h2{margin-bottom:4px} td,th{padding:3px 8px} " \
      "th{background:#eef} .g{color:#555}</style>"

OUTLIERS = CSS + """
<h2>Understanding outliers</h2>
<p>An <b>outlier</b> is an image whose numbers differ strongly from most images in the same dataset.
It is <i>not automatically bad</i>: it is unusual and worth a look.</p>

<h3>1. Features: what is measured per image</h3>
<table border=1 cellspacing=0>
<tr><th>Feature</th><th>Meaning</th><th>Typical cause when unusual</th></tr>
<tr><td>mean, p50</td><td>average / median brightness (HU) of the body</td><td>inverted image, wrong windowing, not in HU</td></tr>
<tr><td>std</td><td>contrast / spread</td><td>blank image, noise, sharp kernel</td></tr>
<tr><td>p00_5, p99_5</td><td>darkest / brightest 0.5 %</td><td>padding, metal, clipping</td></tr>
<tr><td>skewness</td><td>asymmetry of the intensity distribution</td><td>unusual content or processing</td></tr>
<tr><td>entropy_bits</td><td>grey-level variety</td><td>blank, posterized, compressed</td></tr>
<tr><td>frac_at_image_min / max</td><td>share of pure black / white pixels</td><td>borders, saturation</td></tr>
<tr><td>body_fraction</td><td>how much of the image is body</td><td>empty image, tiny anatomy, crop</td></tr>
<tr><td>spacing, shape</td><td>voxel size, image size</td><td>different protocol (5 mm vs 0.6 mm)</td></tr>
<tr><td>histogram</td><td>whole histogram vs the typical one</td><td>any change of overall look</td></tr>
</table>
<p class=g>Hover over a feature name in the table to see its explanation.</p>

<h3>2. The score</h3>
<p><code>score = (value − median of all images) / typical spread</code></p>
<ul>
<li><b>0</b> = perfectly typical; <b>±3.5</b> = default cutoff (flagged beyond it);
    <b>±10 and more</b> = very far off, usually something really wrong.</li>
<li><b>+</b> = higher than typical, <b>−</b> = lower than typical.</li>
<li>Median and spread are <i>robust</i>: a few broken images cannot distort them, which is why one
    blank image can score 700 while normal images stay within ±2.</li>
</ul>

<h3>3. Reading the table</h3>
<ul>
<li><b># features</b>: on how many features the image is unusual. <b>Most important.</b>
    Unusual on 4–5 features → almost certainly broken. Unusual on 1 → often just a bit different.</li>
<li><b>max |score|</b>: how extreme the worst feature is.</li>
<li>Rows are ranked by # features, then score. Start at the top. Click a row to see the preview
    and a plain-language explanation.</li>
</ul>

<h3>4. Three kinds of outliers</h3>
<ol>
<li><b>Broken</b>: blank, noise, corrupted, inverted → remove or fix.</li>
<li><b>Different protocol</b>: kernel, slice thickness, scanner, contrast → keep, but report it.
    This is exactly what a dataset fingerprint should reveal.</li>
<li><b>Normal anatomical variation</b>: child, large patient, different field of view → keep.</li>
</ol>

<h3>5. Controls</h3>
<ul>
<li><b>Method</b>: <i>robust_z</i> (default, median/MAD, most reliable), <i>zscore</i>
    (mean/std, distorted by the outliers themselves), <i>iqr</i> (boxplot rule).</li>
<li><b>Threshold</b>: higher (e.g. 5) = only extreme cases; lower (e.g. 3) = more cases.</li>
<li><b>Max share</b> (1 %): each feature flags at most the 1 % most extreme images (minimum 5).
    Without it, huge datasets flag tens of thousands of images because real data has heavy tails.
    0 = no cap.</li>
<li><b>Group by</b>: compare each image only with similar ones:
  <ul><li><code>level:1</code>: 2nd folder level of the path, e.g. MURA <code>train/XR_WRIST/…</code>
          → wrists compared with wrists.</li>
      <li><code>meta:institute</code>, <code>meta:ConvolutionKernel</code>,
          <code>meta:Manufacturer</code>: a metadata column.</li></ul></li>
</ul>

<h3>6. Suggested workflow</h3>
<ol><li>Click through the top 20–50 rows and look at each preview.</li>
<li>Read <i>why flagged</i> in the detail panel.</li>
<li>If many images are flagged on the same feature, the dataset has subgroups: try
    <b>Group by</b> or look at that feature in <i>Histogram &amp; bins</i>.</li>
<li>Keep a list of broken files. For a paper, report broken vs. different-protocol counts.</li></ol>
"""

HISTOGRAM = CSS + """
<h2>Histogram &amp; bins</h2>
<p>The histogram shows how often each intensity occurs, pooled over <b>all images</b> of the dataset.
For CT the x-axis is in Hounsfield units (HU): air −1000, fat ≈ −100, water 0, soft tissue 20–80,
bone 300+.</p>
<ul>
<li><b>Region</b>: <i>all</i> = every voxel; <i>body</i> = body/foreground found from intensities
    (no masks needed); <i>foreground</i> = voxels inside the annotation masks.</li>
<li><b>log scale</b>: shows rare intensities (metal, padding) that are invisible on a linear axis.</li>
<li><b>Bins</b>: <i>quantiles</i> = n bins with equal voxel counts; <i>equal width</i> = n bins
    of equal size; CT presets: <i>ct_tissue</i> (air, lung, fat, soft tissue, bone…),
    <i>ct_body_composition</i> (Alberta protocol fat/muscle ranges), <i>ct_coarse</i>.</li>
<li><b>Custom edges</b>: type e.g. <code>0 120 180 216</code> and press Enter.</li>
<li><b>Drag the red lines</b> with the mouse to move edges. The table updates live and edges snap
    to the histogram resolution, so the percentages are exact.</li>
<li><b>open ends</b>: adds the bins &lt; first edge and ≥ last edge so the table sums to 100 %.</li>
</ul>
<p class=g>Preset HU ranges are approximate textbook values; verify before citing.</p>
"""

SCAN = CSS + """
<h2>Scanning a dataset</h2>
<ol>
<li><b>Dataset folder</b>: the folder with the images. Layout, file types and modality are
    detected automatically; check them.</li>
<li><b>Layout</b>:
  <ul><li><i>files</i>: every image file under the folder is one case (DICOM: one per series).</li>
      <li><i>paired</i>: separate image and mask folders matched by name.</li>
      <li><i>nnunet</i>: nnU-Net raw dataset (dataset.json, imagesTr, labelsTr).</li>
      <li><i>totalsegmentator</i>: case/ct.nii.gz + case/segmentations/*.nii.gz + meta.csv.</li>
      <li><i>saros</i>: the SAROS dataset (TCIA). Choose the folder holding the DICOM collection
      folders, the unzipped NIfTI masks (case_XXX/body-regions.nii.gz, body-parts.nii.gz) and
      Segmentation-Info*.csv; it is detected automatically. Pick <i>body-regions</i> (13 tissue
      classes: fat, muscle, bone, lungs, …) or <i>body-parts</i> (torso, head, arms, legs); scan
      twice for both. SAROS masks are stored on a 5 mm grid with only every 5th slice annotated
      (value 255 = not annotated). Each annotated slice is placed on the nearest CT slice by its
      physical position; all other slices are left out of the label statistics. Label intensities
      therefore come from the annotated slices only, and organ volumes are <i>estimates</i>: each
      annotated slice is scaled to the stretch of scan it represents (about 25 mm). Collection,
      patient, sex, body region, breast implant and the official fold/test split are attached as
      metadata (use <i>split</i> to group or compare).</li></ul></li>
<li><b>Modality</b>: <i>CT</i> (values in HU), <i>2D</i> (X-ray, photos), <i>3D</i> (other volumes, e.g. MRI).</li>
<li><b>Label names (optional)</b>: names for label numbers in masks
    (1 = liver, &hellip;). Usually leave empty; see <i>Help &rarr; Label names JSON</i>.</li>
<li><b>Metadata CSV (optional)</b>: one row per image with extra columns (body part, view, split&hellip;);
    see <i>Help &rarr; Metadata CSV</i>.</li>
<li><b>Annotations (optional)</b>: COCO/VGG/YOLO/VOC objects for 2D images; see <i>Help &rarr; Annotations</i>.</li>
<li><b>Body region</b>: how the body is found without masks: <i>auto</i> (CT: above −500 HU,
    otherwise Otsu), <i>hu</i>, <i>otsu</i>, <i>none</i>, or type a threshold value.</li>
<li><b>Parallel workers</b>: CPU processes. More = faster, but each needs memory (≈1.5 GB for a
    large CT with 117 masks).</li>
<li><b>Limit cases</b>: scan only the first N for a quick test.</li>
<li><b>Anonymize</b>: case names become case_00001…, paths are removed; the mapping is saved
    separately (.idmap.csv). Use for fingerprints you share.</li>
</ol>
<h3>A dataset already split into train / valid / test</h3>
<p>To scan all parts as <b>one dataset</b>, choose the folder that <i>contains</i> train, valid and
test. Alternatively, use <b>Add folder&hellip;</b> to list several folders (separated by <b>;</b>);
they do not have to be next to each other. The app recognises the usual download layouts and fills
in the fields for you:</p>
<ul>
<li><code>train/images</code> + <code>train/masks</code> (and the same for valid, test): the layout
    is set to <i>paired</i>. Each images folder is matched with the masks folder of the same split.</li>
<li><code>train/labels/*.txt</code> (YOLO) or <code>train/_annotations.coco.json</code> (COCO, e.g.
    Roboflow): all annotation sources are added to <i>Annotations</i>.</li>
<li><code>images/</code> + <code>masks/</code> with train/test sub-folders inside: <i>paired</i>.
    Images and masks are matched within the same sub-folder.</li>
</ul>
<p>The same file name may occur in several splits: each image is matched with the mask or
annotation of its own split. The split name is the first folder of every case name
(<code>train/images/a.png</code>), so you can:</p>
<ul>
<li>group outliers by it (<i>level:0</i>)</li>
<li>split the fingerprint per split and compare train vs test (<i>Compare &rarr; Split open
    dataset</i>)</li>
<li>see copies between train and test, marked &#9888; as <b>data leakage</b> on the
    <i>Quality &amp; duplicates</i> tab</li>
</ul>
<p><b>Cancel</b> keeps finished cases in the cache; pressing <b>Start</b> again resumes.
Scanning the same data again with the same settings is instant.</p>
"""

SUMMARY = CSS + """
<h2>Summary</h2>
<ul>
<li><b>spacing / shape median</b>: typical voxel size (mm) and image size, in z, y, x order.</li>
<li><b>Region table</b>: pooled statistics over all voxels of all images: count, mean, std,
    min, percentiles (p00_5 = 0.5th percentile…), max, skewness, kurtosis, entropy.</li>
<li><b>nnU-Net (labels)</b>: identical to nnU-Net's dataset fingerprint (intensities inside the masks).
    nnU-Net uses <i>percentile_00_5</i> and <i>percentile_99_5</i> to clip CT and
    <i>mean</i>/<i>std</i> to normalise.</li>
<li><b>nnU-Net-style (body mask)</b>: the same computed inside the body region; available without masks.
    Not nnU-Net's own definition.</li>
<li><b>Label table</b>: per structure: in how many cases it is present, median volume (ml),
    intensity mean/std/median.</li>
<li><b>WARNING lines</b>: e.g. cases whose values do not look like HU.</li>
</ul>
<p><b>Export PDF report…</b> (also File menu, Ctrl+P): everything in one PDF for the paper: overview,
geometry, histograms, bins, nnU-Net values and predicted plan, labels, annotations, quality checks,
duplicates, metadata and outliers with thumbnails. The outlier part uses the current settings of the
Outliers tab.</p>
"""

COMPARE = CSS + """
<h2>Comparing datasets</h2>
<p>Load two or more fingerprint JSONs (<b>Add fingerprints…</b>, or <b>Add currently open</b>).
Nothing is rescanned: everything comes from the JSON files. Each dataset keeps its colour in all plots.</p>
<ul>
<li><b>Region</b>: which voxels are compared (only regions present in all datasets are offered).</li>
<li><b>Scale</b>: compare datasets stored with different value ranges (e.g. 16-bit 0&ndash;65535
    and 8-bit 0&ndash;255) on one <b>0&ndash;255 scale</b>. Each dataset is rescaled linearly with its
    <i>own</i> range, then everything (statistics, histograms, CDF, distances, bins, PDF) uses the
    common scale:
    <ul><li><i>0&ndash;255 (p0.5&ndash;p99.5)</i>: recommended; ignores rare extreme pixels, so
        about 1&nbsp;% of the values end up slightly below 0 or above 255.</li>
        <li><i>0&ndash;255 (min&ndash;max)</i>: the full range; a single extreme pixel changes the scale.</li>
        <li><i>0&ndash;255 (bit depth)</i>: like converting the files to 8 bit (16-bit &rarr; divide by
        257; 12-bit data uses 0&ndash;4095).</li></ul>
    The mapping of each dataset is shown above the tables and in the statistics table. Spreads (std,
    IQR) are scaled; shape measures (skewness, kurtosis, entropy) do not change. Use <i>original</i>
    for CT: HU values are already comparable, and the HU bin sets are only offered there.</li>
<li><b>Statistics</b>: side by side: geometry, pooled intensity statistics, per-case variability
    (how much cases differ <i>within</i> each dataset), body size, nnU-Net blocks and metadata.
    With exactly two datasets, <b>Δ</b> (second − first) and <b>Δ %</b> are shown; differences
    of 20 % or more are red.</li>
<li><b>Histograms</b>: normalised intensity distributions overlaid (area = 1, so dataset size does not
    matter). <i>cumulative</i> shows the fraction of voxels below each value: a horizontal gap is a shift,
    a different slope is a different spread.</li>
<li><b>Distances</b>: one number per pair:
  <ul><li><i>wasserstein</i>: average intensity shift in HU / intensity units. Easiest to interpret
          ("the datasets differ by about 40 HU").</li>
      <li><i>ks</i>: largest gap between the cumulative curves, 0–1.</li>
      <li><i>js</i>: Jensen-Shannon divergence, 0 (identical) to 1 (no overlap); sensitive to shape.</li></ul></li>
<li><b>Bins</b>: share of voxels per bin for every dataset (CT presets, quartiles/deciles of the first
    dataset, or your own edges). Δ is in percentage points (pp).</li>
<li><b>Labels</b>: for structures present in several datasets: presence, volume, mean/median HU.</li>
<li><b>Split open dataset…</b>: one fingerprint per group of the dataset that is open in the other tabs,
    by folder (<i>level:0</i> = first folder of the path, e.g. Fractured / Non_fractured) or by a metadata
    column (<i>meta:hand</i>). The groups are saved next to the fingerprint and added here. Nothing is
    rescanned (the scan cache is used).</li>
<li><b>Export CSV…</b>: all tables in one file for the paper.</li>
<li><b>Export PDF…</b>: statistics, histograms, distances, the current bin set and labels as a PDF.</li>
</ul>
<p class=g>Tip: compare datasets of the same modality. CT is comparable in HU; 8-bit X-rays only
roughly, because their grey values depend on each dataset's processing.</p>
"""

LABELS = CSS + """
<h2>Label names JSON (optional)</h2>
<p><b>In most cases leave this field empty.</b> It only gives <i>names</i> to label numbers;
it does not change any statistic.</p>

<h3>Why it exists</h3>
<p>Some datasets store masks as images where every pixel holds a number:
0 = background, 1 = liver, 2 = spleen, &hellip; Nothing inside the mask file says that
&ldquo;1&rdquo; means liver. Without a names file the results show <code>label 1</code>,
<code>label 2</code>, &hellip;; with it they show <code>liver</code>, <code>spleen</code>, &hellip;</p>

<h3>What the file looks like</h3>
<p>Any of these three forms is accepted (label 0 / background is always ignored):</p>
<pre>{ "1": "liver", "2": "spleen", "3": "kidney" }</pre>
<pre>{ "liver": 1, "spleen": 2, "kidney": 3 }</pre>
<pre>{
  "name": "MyDataset",
  "labels": { "background": 0, "liver": 1, "spleen": 2 }
}</pre>
<p class=g>The last one is an nnU-Net <code>dataset.json</code>; its <code>"labels"</code> entry is used.
You can write the file yourself in any text editor and save it as <code>.json</code>.</p>

<h3>Do I need it?</h3>
<table border=1 cellspacing=0>
<tr><th>Your data</th><th>Label names JSON</th></tr>
<tr><td>No masks at all (e.g. CQ500, MURA, CheXpert)</td><td>not needed</td></tr>
<tr><td>One file per organ, e.g. TotalSegmentator <code>segmentations/liver.nii.gz</code></td>
    <td>not needed: names come from the file names</td></tr>
<tr><td>nnU-Net folder (<code>dataset.json</code>, <code>imagesTr</code>, <code>labelsTr</code>)</td>
    <td>not needed: names are read from <code>dataset.json</code></td></tr>
<tr><td>Masks with numbers 1, 2, 3&hellip; and no names file (layout <i>paired</i>)</td>
    <td><b>useful</b>: write one as above</td></tr>
<tr><td>Annotation files (COCO JSON, VOC XML, YOLO), e.g. FracAtlas</td>
    <td>not this field: such files contain the annotations themselves and are not read yet</td></tr>
</table>
"""

METADATA = CSS + """
<h2>Metadata CSV (optional)</h2>
<p>A table with <b>one row per image</b> and extra information about it: body part, view,
scanner, hospital, train/test split, diagnosis&hellip; Example (FracAtlas <code>dataset.csv</code>):</p>
<pre>image_id,hand,leg,hip,shoulder,fractured,fracture_count,frontal,lateral
IMG0000000.jpg,0,1,0,0,0,0,1,1
IMG0000019.jpg,1,0,0,0,1,1,1,0</pre>
<ul>
<li>The column with the image names is <b>found automatically</b>: it may contain the file name
    (<code>IMG0000019.jpg</code>), the name without extension, the path relative to the dataset folder,
    or a folder name (e.g. a patient id).</li>
<li>Separators <code>,</code> <code>;</code> or tab are all fine.</li>
<li>A CSV named <code>dataset.csv</code>, <code>meta.csv</code>, <code>metadata.csv</code>&hellip; in the
    dataset folder or the folder above it is filled in automatically.</li>
<li><b>Several dataset folders</b> (e.g. CheXpert <code>train</code> and <code>valid</code>): add one CSV
    per folder with <b>Add CSV&hellip;</b> (or separate them with <code>;</code>). Each CSV is matched on
    its own, with its own id column. A CSV named like the folder (<code>train.csv</code> next to
    <code>train/</code>) is filled in automatically.</li>
</ul>
<h3>What it is used for</h3>
<ul>
<li><b>Summary / report</b>: distribution of every column.</li>
<li><b>Outliers &rarr; group by</b>: compare each image only with images of the same group
    (e.g. <code>meta:hand</code>), so a hand X-ray is not flagged just because most images are legs.</li>
<li><b>Compare &rarr; Split open dataset&hellip;</b>: one fingerprint per group (hand vs leg,
    frontal vs lateral, train vs test) and compare them like separate datasets.</li>
<li><b>Annotations</b>: a column with "count" in its name (e.g. <code>fracture_count</code>) is checked
    against the number of annotated objects.</li>
</ul>
<p class=g>TotalSegmentator's own <code>meta.csv</code> is read automatically; leave this field empty there.</p>
"""

ANNOTATIONS = CSS + """
<h2>Annotations (optional, 2D images)</h2>
<p>Files that describe <b>objects drawn on the images</b> (polygons or boxes), e.g. fractures, nodules,
lesions. Supported formats:</p>
<table border=1 cellspacing=0>
<tr><th>Format</th><th>What to choose</th></tr>
<tr><td>COCO</td><td>the <code>.json</code> file (e.g. FracAtlas <code>COCO_fracture_masks.json</code>) &ndash; polygons
    and boxes. <b>Recommended.</b></td></tr>
<tr><td>VGG Image Annotator</td><td>the <code>.json</code> file</td></tr>
<tr><td>YOLO</td><td>the <b>folder</b> with one <code>.txt</code> per image (boxes or YOLO-seg polygons);
    class names from <code>classes.txt</code> or <code>data.yaml</code></td></tr>
<tr><td>Pascal VOC</td><td>the <b>folder</b> with one <code>.xml</code> per image (boxes)</td></tr>
</table>
<p>Images are matched by file name. Images that are not in the annotation file (e.g. FracAtlas
<code>Non_fractured</code>) are treated as having <b>no objects</b>.</p>
<h3>What you get</h3>
<ul>
<li><b>Per category</b> (like the per-label statistics for masks): how many images contain it, pixel
    intensities inside the objects.</li>
<li><b>Per object</b>: area (pixels, mm&sup2; when the pixel size is known, share of the image), box width,
    height and aspect ratio, number of polygon points (how detailed the annotation is), and
    <b>contrast</b> = mean intensity inside minus a ring of pixels around it (how visible the object is).</li>
<li><b>Position heatmap</b>: where objects usually are in the image.</li>
<li><b>Consistency checks</b>: polygons with fewer than 3 points, objects outside the image, tiny objects,
    duplicated objects, annotated size &ne; real image size, annotated images that do not exist,
    and a metadata count column (e.g. <code>fracture_count</code>) that disagrees with the objects.</li>
<li><b>Unusual objects</b>: extremely small/large or elongated objects compared with the others.</li>
<li>The nnU-Net fingerprint is computed with the objects as foreground.</li>
</ul>
<p class=g>Prefer COCO polygons over boxes: boxes include background pixels, so intensity and contrast
measurements are less precise.</p>
"""

QUALITY = CSS + """
<h2>Quality &amp; duplicates</h2>
<p>Rule-based checks of every image. They are <b>hints</b>: click a row and look at the image.</p>
<h3>Image checks</h3>
<table border=1 cellspacing=0>
<tr><th>Check</th><th>Rule</th><th>Typical cause</th></tr>
<tr><td>Blank / flat</td><td>no contrast at all</td><td>empty export, failed conversion</td></tr>
<tr><td>Very low dynamic range</td><td>p0.5&ndash;p99.5 range &lt; 2 % of the data type range</td><td>under-exposure, wrong windowing</td></tr>
<tr><td>Bright background</td><td>border clearly brighter than the image centre (2D)</td><td>inverted grey scale (MONOCHROME1 not handled), white collimation</td></tr>
<tr><td>Large uniform borders</td><td>&gt; 25 % of the image in constant rows/columns at the edges</td><td>padding to a square, black bars, scanned film</td></tr>
<tr><td>Colour image</td><td>RGB channels differ</td><td>photo of a monitor, coloured overlays/annotations</td></tr>
<tr><td>Blurry</td><td>sharpness &lt; 1/3 of the dataset median</td><td>motion, smoothing, upscaled thumbnails</td></tr>
<tr><td>Noisy</td><td>noise &gt; 3&times; the dataset median</td><td>low dose, grain, sharp kernel</td></tr>
<tr><td>Saturated</td><td>&gt; 20 % of pixels at the data-type maximum</td><td>over-exposure, burned-out areas</td></tr>
<tr><td>Bright marks in corners</td><td>small very bright areas in the corners</td><td>burned-in text, L/R markers. <b>Information only</b>, not counted as a problem: almost every
    radiograph has them (59&nbsp;% of CheXpert)</td></tr>
</table>
<p class=g>Sharpness and noise are measured relative to each image's own contrast, so they are comparable
between 8-bit and 16-bit images. They are also used as outlier features.</p>
<h3>Duplicates</h3>
<ul>
<li><b>Exact duplicate</b>: identical pixel values (the same image stored twice, maybe under another name).</li>
<li><b>Near-duplicate</b>: the same picture after re-saving as JPEG, resizing or a brightness change
    (a 256-bit difference hash of the image shrunk to 17&times;16 pixels; at most 14 of the 256 bits
    may differ). The rule is strict on purpose: in CheXpert, re-saved copies differ in at most 6 bits
    and +10&nbsp;% brightness in at most 12 bits, while two <i>different</i> chest X-rays almost never
    come closer than 32 bits. Cropped or edited copies are not found, and strongly recompressed copies
    of images with large flat, noisy areas (where re-saving flips many hash bits) can be missed. For CT, three views of the volume are compared,
    so e.g. a thin and a thick reconstruction of the same scan are found.</li>
<li><b>&#9888; across folders</b>: the two copies are in different top-level folders (e.g. train and
    valid, or Fractured and Non_fractured): possible <b>data leakage</b> or a labelling error.</li>
<li>Blank images are excluded (they would all look identical).</li>
</ul>
<p class=g>Follow-up studies of the same patient can look almost the same and are legitimately similar;
check whether such pairs end up in different splits.</p>
<h3>Annotation checks</h3>
<p>Shown when the scan used an annotation file (see <i>Help &rarr; Annotations</i>).</p>
"""

PLAN = CSS + """
<h2>Predicted nnU-Net plan</h2>
<p>What <code>nnUNetv2_plan_and_preprocess</code> (default planner, 8 GB GPU) would choose for this
dataset, computed from the fingerprint with nnU-Net v2's own rules:</p>
<ul>
<li><b>Target spacing</b>: median spacing; for strongly anisotropic data (e.g. 5 mm slices with 0.7 mm
    pixels) the 10th percentile of the coarse axis.</li>
<li><b>Median shape</b>: the typical image size after resampling to the target spacing.</li>
<li><b>Patch size</b>: as large as the 8 GB memory budget allows, shaped like the images, divisible
    by 2 for every pooling step.</li>
<li><b>Batch size</b>: from the remaining memory; at least 2; one batch covers at most 5 % of the dataset.</li>
<li><b>3d_lowres / cascade</b>: only when the images are much larger than the patch.</li>
<li><b>Normalisation</b>: CT: clip to the foreground 0.5&ndash;99.5 percentiles, then z-score with the
    foreground mean and std. Other modalities: z-score per image.</li>
</ul>
<p>The planning rules are implemented exactly. The GPU-memory model is a simplified version, so patch and
batch sizes are marked approximate (checked against nnU-Net's plans for MSD Hippocampus, Heart and
Liver: same patch sizes).</p>
<p class=g>Without masks, the foreground is the intensity body region, so CT normalisation values are an
approximation.</p>
"""

PAGES = {"outliers": ("Understanding outliers", OUTLIERS), "histogram": ("Histogram & bins", HISTOGRAM),
         "scan": ("Scanning a dataset", SCAN), "summary": ("Summary", SUMMARY),
         "compare": ("Comparing datasets", COMPARE), "labels": ("Label names JSON", LABELS),
         "metadata": ("Metadata CSV", METADATA), "annotations": ("Annotations", ANNOTATIONS),
         "quality": ("Quality & duplicates", QUALITY), "plan": ("Predicted nnU-Net plan", PLAN)}


def show_help(parent, topic: str):
    title, html = PAGES[topic]
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"Help: {title}")
    dlg.resize(820, 760)
    lay = QVBoxLayout(dlg)
    tb = QTextBrowser()
    tb.setOpenExternalLinks(True)
    tb.setHtml(html)
    lay.addWidget(tb)
    bb = QDialogButtonBox(QDialogButtonBox.Close)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    dlg.setModal(False)
    dlg.show()
    return dlg


def help_button(parent, topic: str) -> QToolButton:
    b = QToolButton(parent)
    b.setText("?")
    b.setToolTip(f"Help: {PAGES[topic][0]}")
    b.setStyleSheet("QToolButton{font-weight:bold; border:1px solid #888; border-radius:10px;"
                    "min-width:20px; min-height:20px}")
    b.clicked.connect(lambda: setattr(parent, "_help_dlg", show_help(parent, topic)))
    return b
