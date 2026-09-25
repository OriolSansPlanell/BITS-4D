"""The PDF report: it must run on current matplotlib and draw the
histogram the right way round.

Regressions: ``matplotlib.cm.get_cmap`` (removed in matplotlib 3.9) made
every report fail; and the histogram, stored as [xray_bin, neutron_bin], was
transposed before drawing, so it appeared mirrored about the diagonal and
the ROI outlines did not sit on the data they select.
"""

import numpy as np

from utils.pdf_reporter import PDFReporter


class _Selection:
    def __init__(self, name, mask, roi):
        self.name = name
        self.spatial_mask = mask
        self.histogram_roi = roi
        self.color = None
        self.visible = True
        self.cluster_id = None


class _CapturePdf:
    def __init__(self):
        self.figures = []

    def savefig(self, figure, *args, **kwargs):
        self.figures.append(figure)


def _selections():
    mask = np.zeros((8, 8), dtype=bool)
    mask[:3, :3] = True
    roi = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 5.0]])
    return [_Selection("A", mask, roi)]


def test_histogram_page_keeps_neutron_on_x():
    # Asymmetric: 4 X-ray bins x 6 neutron bins, one hot bin
    hist = np.zeros((4, 6))
    hist[3, 1] = 100.0          # high X-ray, low neutron
    n_edges = np.linspace(0, 6, 7)
    x_edges = np.linspace(0, 40, 5)
    pdf = _CapturePdf()
    PDFReporter._page_histogram(pdf, (hist, n_edges, x_edges), _selections())
    image = pdf.figures[0].axes[0].images[0]
    np.testing.assert_allclose(image.get_array(), np.log10(hist + 1))
    assert tuple(image.get_extent()) == (0.0, 6.0, 0.0, 40.0)


def test_whole_report_is_written(tmp_path):
    rng = np.random.default_rng(0)
    neutron = rng.normal(500, 20, (8, 8))
    xray = rng.normal(500, 20, (8, 8))
    hist = rng.integers(0, 50, (16, 16)).astype(float)
    edges = np.linspace(400, 600, 17)
    path = PDFReporter.generate_report(
        str(tmp_path / "report.pdf"), _selections(), neutron, xray,
        (hist, edges, edges),
    )
    assert (tmp_path / "report.pdf").stat().st_size > 0
