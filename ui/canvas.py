import copy
import json

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (
    QApplication, QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsView, QMenu, QToolTip,
)

from engine import colours, ports, workspace
from engine.workflow import Connection
from ui import theme


NODE_WIDTH = 120
NODE_HEIGHT = 60
PORT_RADIUS = 7
# The least room each port gets down the side of a node, so a node with many ports grows taller
PORT_GAP = 15
BADGE_RADIUS = 8
BADGE_MARGIN = 4
GRID_SIZE = 40
MIN_ZOOM = 0.25
MAX_ZOOM = 4.0
ZOOM_STEP = 1.1
# A press and release closer together than this, in pixels, is a click rather than a drag
CLICK_DISTANCE = 4
# How far a pasted copy lands from the original, so it doesn't hide it
PASTE_OFFSET = 40
# What a module dragged from the toolbar is carried as
MODULE_MIME = "application/x-coefflow-module"
# The colour each type of port is filled with, by theme colour name (see engine/ports.py)
PORT_COLOURS = {ports.TABLE: "PORT", ports.MODEL: "CHART_LINE"}
STATUS_COLOURS = {"done": "NODE_DONE", "error": "NODE_ERROR", "running": "CHART_LINE", "queued": "SUBTITLE"}
SCENE_SIZE = 200000


def node_height(node):
    return max(NODE_HEIGHT, PORT_GAP * (max(node.inputs, node.outputs) + 1))


# The canvas positions of a node's ports down one side: its inputs on the left, outputs on the right
def port_positions(node, side):
    count = node.inputs if side == "input" else node.outputs
    height = node_height(node)
    x = node.x - NODE_WIDTH / 2 if side == "input" else node.x + NODE_WIDTH / 2
    top = node.y - height / 2
    return [QPointF(x, top + height * (i + 1) / (count + 1)) for i in range(count)]


# A curve from an output port across to an input port, leaving and arriving level
def connection_path(start, end):
    path = QPainterPath(start)
    reach = max(40.0, abs(end.x() - start.x()) / 2)
    path.cubicTo(QPointF(start.x() + reach, start.y()), QPointF(end.x() - reach, end.y()), end)
    return path


class NodeItem(QGraphicsItem):
    # A node on the canvas, drawn about its centre. Dragging it moves the node, and with several
    # selected they all move together
    def __init__(self, node, view):
        super().__init__()
        self.node = node
        self.view = view
        self.setPos(node.x, node.y)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    def body(self):
        height = node_height(self.node)
        return QRectF(-NODE_WIDTH / 2, -height / 2, NODE_WIDTH, height)

    def boundingRect(self):
        margin = PORT_RADIUS + 6
        return self.body().adjusted(-margin, -margin, margin, margin)

    def shape(self):
        path = QPainterPath()
        path.addRect(self.body())
        return path

    # The circle the ? sits in, at the top right of the node
    def badge(self):
        body = self.body()
        return QRectF(body.right() - BADGE_MARGIN - BADGE_RADIUS * 2, body.top() + BADGE_MARGIN, BADGE_RADIUS * 2, BADGE_RADIUS * 2)

    # Keeps the node's own position in step as the item is dragged, and its lines with it
    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node.x, self.node.y = value.x(), value.y()
            self.view.update_connections()
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        node = self.node
        body = self.body()
        radius = colours.NODE_RADIUS
        border = theme.colour(STATUS_COLOURS.get(node.status, "NODE_BORDER"))

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(border, 3)
        if node.status == "running":
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(theme.colour("NODE"))
        painter.drawRoundedRect(body, radius, radius)

        if self.isSelected():
            outline = QPen(theme.colour("TEXT"), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(outline)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(body.adjusted(-5, -5, 5, 5), radius, radius)

        painter.setPen(theme.colour("TEXT"))
        painter.setFont(theme.font(13))
        text_area = body.adjusted(6, 4, -6, -4)
        painter.drawText(text_area, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, node.name)

        badge = self.badge()
        painter.setPen(QPen(theme.colour("TEXT"), 1))
        painter.setBrush(theme.colour("TOOLBAR"))
        painter.drawEllipse(badge)
        painter.setFont(theme.font(11))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, "?")

        painter.setPen(QPen(theme.colour("NODE_BORDER"), 2))
        for side in ("input", "output"):
            for index, position in enumerate(port_positions(node, side)):
                if side == "input":
                    kind = ports.input_types(node, index)[0]
                else:
                    kind = ports.output_type(node, index)
                painter.setBrush(theme.colour(PORT_COLOURS.get(kind, "PORT")))
                painter.drawEllipse(position - self.pos(), PORT_RADIUS, PORT_RADIUS)


class ConnectionItem(QGraphicsPathItem):
    # A line between two ports. It can be selected by clicking it, and deleted like a node
    def __init__(self, connection):
        super().__init__()
        self.connection = connection
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.update_path()

    def update_path(self):
        connection = self.connection
        starts = port_positions(connection.source, "output")
        ends = port_positions(connection.target, "input")
        if connection.output_index < len(starts) and connection.input_index < len(ends):
            self.setPath(connection_path(starts[connection.output_index], ends[connection.input_index]))

    # A wider strip than the line itself, so it is easy to click
    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(self.path())

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = theme.colour("TEXT" if self.isSelected() else "CONNECTION")
        painter.setPen(QPen(colour, 4 if self.isSelected() else 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())


class Canvas(QGraphicsView):
    # The workflow area: shows a workflow's nodes and connections and lets them be edited.
    # The scroll wheel zooms towards the pointer, dragging the background moves the view and
    # Shift + dragging it selects every node in a box. Dragging a node moves it, a click opens
    # its settings and Ctrl + click adds it to the selection. A line is drawn by dragging from an
    # output port to an input port; grabbing a connected input unplugs its line to move it.
    # Every change to the workflow ends with changed, so the app can keep its undo history
    node_clicked = Signal(object)
    help_asked = Signal(str, QPoint)
    run_to_here = Signal(object)
    view_asked = Signal(object)
    changed = Signal()
    message = Signal(str)
    zoomed = Signal()

    def __init__(self, workflow):
        super().__init__()
        self.workflow = workflow
        self.zoom = 1.0
        self.locked = False
        self.node_items = {}
        self.connection_items = []
        self.connecting = None
        self.connecting_item = None
        self.panning = None
        self.pressed = None
        self.moved_from = {}

        scene = QGraphicsScene(self)
        scene.setSceneRect(-SCENE_SIZE / 2, -SCENE_SIZE / 2, SCENE_SIZE, SCENE_SIZE)
        self.setScene(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.centerOn(0, 0)
        self.rebuild()

    # ----- Showing the workflow -----

    # Remakes every item from the workflow, keeping the selection. Called after any change to
    # which nodes and connections there are
    def rebuild(self):
        selected = {node.id for node in self.selected_nodes()}
        scene = self.scene()
        scene.clear()
        self.connecting_item = None
        self.node_items = {}
        self.connection_items = []

        for order, node in enumerate(self.workflow.nodes):
            item = NodeItem(node, self)
            item.setZValue(order)
            item.setSelected(node.id in selected)
            scene.addItem(item)
            self.node_items[node.id] = item
        for connection in self.workflow.connections:
            item = ConnectionItem(connection)
            scene.addItem(item)
            self.connection_items.append(item)
        self.viewport().update()

    def update_connections(self):
        for item in self.connection_items:
            item.update_path()

    # Draws the nodes again, as after a run changes their colours
    def refresh(self):
        self.scene().update()
        self.viewport().update()

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, theme.colour("BACKGROUND"))
        spacing = GRID_SIZE * (2 if self.zoom < 0.5 else 1)
        painter.setPen(QPen(theme.colour("GRID"), 0))
        x = int(rect.left() // spacing) * spacing
        while x < rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += spacing
        y = int(rect.top() // spacing) * spacing
        while y < rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += spacing

    # ----- The view -----

    # Where the canvas is looking, as kept in a workflow file
    def view(self):
        centre = self.mapToScene(self.viewport().rect().center())
        return {"x": centre.x(), "y": centre.y(), "zoom": self.zoom}

    def set_view(self, view):
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, view.get("zoom", 1.0)))
        self.resetTransform()
        self.scale(self.zoom, self.zoom)
        self.centerOn(view.get("x", 0), view.get("y", 0))
        self.zoomed.emit()

    def zoom_by(self, factor):
        target = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        self.scale(target / self.zoom, target / self.zoom)
        self.zoom = target
        self.zoomed.emit()

    # The zoom buttons zoom about the middle of the canvas rather than the pointer
    def zoom_about_centre(self, factor):
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.zoom_by(factor)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def zoom_in(self):
        self.zoom_about_centre(ZOOM_STEP)

    def zoom_out(self):
        self.zoom_about_centre(1 / ZOOM_STEP)

    # Fits every node in the window, or goes back to the middle if there are none
    def zoom_to_fit(self):
        if not self.workflow.nodes:
            self.set_view(workspace.DEFAULT_VIEW)
            return
        bounds = QRectF()
        for item in self.node_items.values():
            bounds = bounds.united(item.sceneBoundingRect())
        bounds.adjust(-60, -60, 60, 60)
        size = self.viewport().size()
        zoom = min(size.width() / bounds.width(), size.height() / bounds.height(), 1.5)
        self.set_view({"x": bounds.center().x(), "y": bounds.center().y(), "zoom": zoom})

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_by(ZOOM_STEP)
        elif event.angleDelta().y() < 0:
            self.zoom_by(1 / ZOOM_STEP)

    # ----- Finding what is under the pointer -----

    def node_at(self, scene_pos):
        for item in self.scene().items(scene_pos):
            if isinstance(item, NodeItem):
                return item.node
        return None

    # Returns (node, "input" or "output", port number) for the port at a canvas position.
    # Ports are easier to hit than they look, and never smaller than a fingertip however far out
    def port_at(self, scene_pos):
        reach = max(PORT_RADIUS + 4, 10 / self.zoom)
        for node in reversed(self.workflow.nodes):
            for side in ("output", "input"):
                for index, position in enumerate(port_positions(node, side)):
                    offset = position - scene_pos
                    if offset.x() ** 2 + offset.y() ** 2 <= reach ** 2:
                        return node, side, index
        return None

    def on_badge(self, node, scene_pos):
        item = self.node_items.get(node.id)
        return item is not None and item.badge().contains(scene_pos - item.pos())

    def selected_nodes(self):
        return [item.node for item in self.node_items.values() if item.isSelected()]

    def selected_connections(self):
        return [item.connection for item in self.connection_items if item.isSelected()]

    def select_all(self):
        for item in self.node_items.values():
            item.setSelected(True)

    def select_only(self, nodes):
        self.scene().clearSelection()
        for node in nodes:
            if node.id in self.node_items:
                self.node_items[node.id].setSelected(True)

    # ----- The mouse -----

    def mousePressEvent(self, event):
        self.setFocus()
        position = event.position().toPoint()
        scene_pos = self.mapToScene(position)
        left = event.button() == Qt.MouseButton.LeftButton
        self.pressed = None

        if event.button() == Qt.MouseButton.MiddleButton:
            self.start_pan(position)
            return
        if not left:
            super().mousePressEvent(event)
            return

        port = None if self.locked else self.port_at(scene_pos)
        if port:
            self.start_connecting(port, scene_pos)
            return

        node = self.node_at(scene_pos)
        if node and self.on_badge(node, scene_pos):
            self.help_asked.emit(node.name, event.globalPosition().toPoint())
            return

        on_item = node or any(isinstance(item, ConnectionItem) for item in self.items(position))
        if not on_item:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
                super().mousePressEvent(event)
            else:
                self.scene().clearSelection()
                self.start_pan(position)
            return

        if node:
            self.pressed = (node, position)
            self.workflow.bring_to_front(node)
            for order, other in enumerate(self.workflow.nodes):
                self.node_items[other.id].setZValue(order)
        if self.locked:
            return
        self.moved_from = {n.id: (n.x, n.y) for n in self.workflow.nodes}
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        position = event.position().toPoint()
        if self.panning is not None:
            delta = position - self.panning
            self.panning = position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self.connecting:
            node, index = self.connecting
            start = port_positions(node, "output")[index]
            self.connecting_item.setPath(connection_path(start, self.mapToScene(position)))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        position = event.position().toPoint()
        if self.panning is not None:
            self.panning = None
            self.viewport().unsetCursor()
            return
        if self.connecting:
            self.finish_connecting(self.mapToScene(position))
            return

        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pressed, self.pressed = self.pressed, None
        if pressed:
            node, start = pressed
            moved = (position - start).manhattanLength()
            plain = not event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
            if moved <= CLICK_DISTANCE and plain:
                self.node_clicked.emit(node)
        if any(self.moved_from.get(n.id) != (n.x, n.y) for n in self.workflow.nodes):
            self.changed.emit()
        self.moved_from = {}

    def start_pan(self, position):
        self.panning = position
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    # Starts a line from an output port. Grabbing an input that is already connected unplugs
    # that line so it can be moved to another input, or dropped to remove it
    def start_connecting(self, port, scene_pos):
        node, side, index = port
        if side == "input":
            existing = self.workflow.connection_into(node, index)
            if not existing:
                return
            self.workflow.disconnect(existing)
            node, index = existing.source, existing.output_index
            self.rebuild()
            self.changed.emit()

        self.connecting = (node, index)
        self.connecting_item = QGraphicsPathItem()
        self.connecting_item.setPen(QPen(theme.colour("CONNECTION"), 3, Qt.PenStyle.DashLine))
        self.connecting_item.setZValue(len(self.workflow.nodes) + 1)
        start = port_positions(node, "output")[index]
        self.connecting_item.setPath(connection_path(start, scene_pos))
        self.scene().addItem(self.connecting_item)

    def finish_connecting(self, scene_pos):
        source, output_index = self.connecting
        self.connecting = None
        port = self.port_at(scene_pos)
        if port and port[1] == "input":
            refusal = self.workflow.refusal(source, output_index, port[0], port[2])
            if refusal:
                self.message.emit(refusal)
            else:
                self.workflow.connect(source, output_index, port[0], port[2])
                self.changed.emit()
        self.rebuild()

    # ----- Dropping modules from the toolbar -----

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(MODULE_MIME) and not self.locked:
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(MODULE_MIME) and not self.locked:
            event.acceptProposedAction()

    def dropEvent(self, event):
        name = bytes(event.mimeData().data(MODULE_MIME)).decode("utf-8")
        self.add_node(name, self.mapToScene(event.position().toPoint()))
        event.acceptProposedAction()

    # Adds a module at a canvas position, or in the middle of the view if none is given
    def add_node(self, name, scene_pos=None):
        if self.locked:
            return None
        if scene_pos is None:
            scene_pos = self.mapToScene(self.viewport().rect().center())
        node = self.workflow.add_node(name, scene_pos.x(), scene_pos.y())
        self.rebuild()
        self.select_only([node])
        self.setFocus()
        self.changed.emit()
        return node

    # ----- Editing -----

    # Removes the selected nodes, with their lines, and the selected lines
    def delete_selected(self):
        if self.locked:
            return
        nodes, connections = self.selected_nodes(), self.selected_connections()
        if not nodes and not connections:
            return
        for connection in connections:
            if connection in self.workflow.connections:
                self.workflow.disconnect(connection)
        for node in nodes:
            self.workflow.remove_node(node)
        self.rebuild()
        self.changed.emit()

    # The selected nodes and the lines between them, as text for the clipboard
    def copied(self):
        nodes = self.selected_nodes()
        if not nodes:
            return None
        chosen = {node.id for node in nodes}
        connections = [
            workspace.connection_record(c) for c in self.workflow.connections
            if c.source.id in chosen and c.target.id in chosen
        ]
        records = [workspace.memory_record(node) for node in nodes]
        return json.dumps({"coefflow_nodes": records, "connections": connections}, default=str)

    def copy(self):
        text = self.copied()
        if text:
            QApplication.clipboard().setText(text)
            self.message.emit(f"Copied {len(self.selected_nodes())} module(s)")

    def cut(self):
        if self.selected_nodes() and not self.locked:
            self.copy()
            self.delete_selected()

    # Adds a copy of the nodes on the clipboard, with new IDs, their settings and the lines
    # between them. They land where the pointer is if it is over the canvas, or else a little
    # down and right of the originals, and are left selected so they can be dragged into place
    def paste(self, text=None):
        if self.locked:
            return
        try:
            copied = json.loads(text if text is not None else QApplication.clipboard().text())
            records = copied["coefflow_nodes"]
        except (ValueError, KeyError, TypeError):
            return
        if not records:
            return

        offset_x = offset_y = PASTE_OFFSET
        pointer = self.viewport().mapFromGlobal(QCursor.pos())
        if text is None and self.viewport().rect().contains(pointer):
            target = self.mapToScene(pointer)
            offset_x = target.x() - min(record["x"] for record in records)
            offset_y = target.y() - min(record["y"] for record in records)

        made = {}
        for record in records:
            node = self.workflow.add_node(record["name"], record["x"] + offset_x, record["y"] + offset_y)
            node.settings = copy.deepcopy(record.get("settings") or {})
            made[record["id"]] = node
        for source_id, output_index, target_id, input_index in copied.get("connections", []):
            source, target = made.get(source_id), made.get(target_id)
            if source and target:
                self.workflow.connections.append(Connection(source, output_index, target, input_index))
        for node in made.values():
            self.workflow.tidy_inputs(node)

        self.rebuild()
        self.select_only(made.values())
        self.changed.emit()

    def duplicate(self):
        text = self.copied()
        if text:
            self.paste(text)

    def keyPressEvent(self, event):
        key, control = event.key(), event.modifiers() & Qt.KeyboardModifier.ControlModifier
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        elif control and key == Qt.Key.Key_A:
            self.select_all()
        elif control and key == Qt.Key.Key_C:
            self.copy()
        elif control and key == Qt.Key.Key_X:
            self.cut()
        elif control and key == Qt.Key.Key_V:
            self.paste()
        elif control and key == Qt.Key.Key_D:
            self.duplicate()
        elif key == Qt.Key.Key_Escape:
            self.scene().clearSelection()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        node = self.node_at(scene_pos)
        menu = QMenu(self)
        if node:
            if node not in self.selected_nodes():
                self.select_only([node])
            menu.addAction("Open settings", lambda: self.node_clicked.emit(node))
            menu.addAction("Run to here", lambda: self.run_to_here.emit(node)).setEnabled(not self.locked)
            menu.addAction("View data", lambda: self.view_asked.emit(node)).setEnabled(bool(node.last_outputs or node.last_inputs))
            menu.addAction("Help", lambda: self.help_asked.emit(node.name, event.globalPos()))
            menu.addSeparator()
            menu.addAction("Copy\tCtrl+C", self.copy)
            menu.addAction("Cut\tCtrl+X", self.cut).setEnabled(not self.locked)
            menu.addAction("Duplicate\tCtrl+D", self.duplicate).setEnabled(not self.locked)
            menu.addAction("Delete\tDel", self.delete_selected).setEnabled(not self.locked)
        else:
            menu.addAction("Paste\tCtrl+V", self.paste).setEnabled(not self.locked)
            menu.addAction("Select all\tCtrl+A", self.select_all)
            if self.selected_connections():
                menu.addAction("Delete line\tDel", self.delete_selected).setEnabled(not self.locked)
            menu.addSeparator()
            menu.addAction("Zoom to fit", self.zoom_to_fit)
        menu.exec(event.globalPos())

    # ----- Tooltips -----

    # Hovering over a port says what it carries, and over a node that failed, why
    def viewportEvent(self, event):
        if event.type() == QEvent.Type.ToolTip:
            scene_pos = self.mapToScene(event.pos())
            text = self.tooltip_at(scene_pos)
            if text:
                QToolTip.showText(event.globalPos(), text, self)
            else:
                QToolTip.hideText()
            return True
        return super().viewportEvent(event)

    def tooltip_at(self, scene_pos):
        port = self.port_at(scene_pos)
        if port:
            node, side, index = port
            specs = getattr(node.module, "INPUT_SPECS" if side == "input" else "OUTPUT_SPECS", [])
            which = 0 if node.growing and side == "input" else index
            if side == "input":
                kind = " or ".join(ports.input_types(node, index))
            else:
                kind = ports.output_type(node, index)
            label = f"{'In' if side == 'input' else 'Out'} {index + 1} ({kind})"
            if which < len(specs):
                return f"<b>{label}: {specs[which][0]}</b><br>{specs[which][1]}"
            return f"<b>{label}</b>"

        node = self.node_at(scene_pos)
        if node and node.error:
            return f"<b>{node.name} #{node.id}</b><br>{node.error}"
        if node and node.status is None and node.ran_as:
            return f"{node.name} #{node.id}: changed since it last ran"
        return ""
