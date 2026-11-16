from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetricsF, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from engine import chart, draw
from engine.draw import Rect
from ui import theme


# Draws what the modules describe (see engine/draw.py) with Qt: on screen, in a ChartWidget,
# and into a picture file. Text is measured with the real font from here on, so charts are
# laid out for the letters as they are drawn


def qt_font(font):
    return theme.font(font.points, font.bold)


def measure(font, text):
    metrics = QFontMetricsF(qt_font(font))
    return metrics.horizontalAdvance(text), metrics.height()


draw.measure = measure


class Painter:
    # The painter a chart function is handed, drawing through a QPainter
    def __init__(self, qpainter):
        self.qpainter = qpainter
        qpainter.setRenderHint(QPainter.RenderHint.Antialiasing)
        qpainter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    # Outlines with a pen of the given width, or fills when the width is 0
    def use(self, colour, width):
        colour = QColor(*colour)
        if width:
            pen = QPen(colour, width)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            self.qpainter.setPen(pen)
            self.qpainter.setBrush(Qt.BrushStyle.NoBrush)
        else:
            self.qpainter.setPen(Qt.PenStyle.NoPen)
            self.qpainter.setBrush(QBrush(colour))

    def line(self, colour, start, end, width=1):
        self.use(colour, max(1, width))
        self.qpainter.drawLine(QPointF(*start), QPointF(*end))

    def lines(self, colour, points, width=1):
        self.use(colour, max(1, width))
        self.qpainter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in points]))

    def rect(self, colour, rect, width=0, radius=0):
        left, top, wide, high = rect
        self.use(colour, width)
        # An outline is drawn inside the rectangle, as a filled one is
        shape = QRectF(left + width / 2, top + width / 2, wide - width, high - width)
        if radius:
            self.qpainter.drawRoundedRect(shape, radius, radius)
        else:
            self.qpainter.drawRect(shape)

    def circle(self, colour, centre, radius, width=0):
        self.use(colour, width)
        self.qpainter.drawEllipse(QPointF(*centre), radius - width / 2, radius - width / 2)

    def polygon(self, colour, points, width=0):
        self.use(colour, width)
        self.qpainter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))

    def text(self, font, text, colour, rotated=False, **anchor):
        text = str(text)
        made = qt_font(font)
        metrics = QFontMetricsF(made)
        wide, high = metrics.horizontalAdvance(text), metrics.height()
        self.qpainter.setFont(made)
        self.qpainter.setPen(QColor(*colour))

        if not rotated:
            left, top = draw.anchored(wide, high, anchor)
            self.qpainter.drawText(QPointF(left, top + metrics.ascent()), text)
            return

        # Turned to read upwards, the text's box is as wide as the text is tall
        left, top = draw.anchored(high, wide, anchor)
        self.qpainter.save()
        self.qpainter.translate(left, top + wide)
        self.qpainter.rotate(-90)
        self.qpainter.drawText(QPointF(0, metrics.ascent()), text)
        self.qpainter.restore()

    def set_clip(self, rect):
        if rect is None:
            self.qpainter.setClipping(False)
        else:
            self.qpainter.setClipRect(QRectF(*rect))


# Saves a chart as a PNG picture, drawn in its style at the style's export size.
# Raises OSError if the file can't be written
def export_png(chart_function, settings, path):
    width, height = chart.export_size(settings)
    picture = QImage(width, height, QImage.Format.Format_ARGB32)
    picture.fill(QColor(*chart.colour_in(settings, "background")))
    qpainter = QPainter(picture)
    try:
        chart.render(Painter(qpainter), Rect(0, 0, width, height), chart_function, settings, fill=True, margin=chart.EXPORT_MARGIN)
    finally:
        qpainter.end()
    if not picture.save(path, "PNG"):
        raise OSError(f"Could not write {path}")


class ChartWidget(QWidget):
    # Shows a chart: a function(painter, area) drawn in the style kept in the node's settings.
    # A chart that fails to draw says why in its place rather than stopping the app
    def __init__(self):
        super().__init__()
        self.chart = None
        self.settings = None
        self.setMinimumHeight(chart.MIN_SIZE[1])

    def show_chart(self, chart_function, settings):
        self.chart = chart_function
        self.settings = settings
        self.update()

    def paintEvent(self, event):
        if not self.chart:
            return
        qpainter = QPainter(self)
        area = Rect(0, 0, self.width(), self.height())
        try:
            if chart.has_room(area):
                chart.render(Painter(qpainter), area, self.chart, self.settings)
            else:
                self.say(qpainter, "No room to draw the chart here. Export PNG saves it in full")
        except Exception as error:
            qpainter.setClipping(False)
            self.say(qpainter, f"Could not draw the chart: {error}")
        finally:
            qpainter.end()

    def say(self, qpainter, text):
        qpainter.setPen(theme.colour("TEXT"))
        qpainter.setFont(theme.font(13))
        qpainter.drawText(self.rect().adjusted(4, 4, -4, -4), Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, text)
