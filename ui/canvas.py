import pygame

from engine.workflow import Connection, reaches
from ui import colours, display
from ui.node import Node, PORT_RADIUS


GRID_SIZE = 40
FONT_SIZE = 24
MIN_ZOOM = 0.25
MAX_ZOOM = 4.0
ZOOM_STEP = 1.1


# Changes the mouse cursor, if the system has that cursor
def set_cursor(cursor):
    try:
        pygame.mouse.set_cursor(cursor)
    except pygame.error:
        pass


class Canvas:
    # The workflow area, which keeps track of zoom, position, nodes and connections
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.zoom = 1.0
        self.offset_x = self.rect.left
        self.offset_y = self.rect.top
        self.nodes = []
        self.connections = []
        self.next_id = 1
        self.font_size = FONT_SIZE
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.panning = False

    # Converts between screen positions and canvas positions
    def to_screen(self, x, y):
        return x * self.zoom + self.offset_x, y * self.zoom + self.offset_y

    def to_canvas(self, x, y):
        return (x - self.offset_x) / self.zoom, (y - self.offset_y) / self.zoom

    # Places a new node where it was dropped on screen, with the next free ID
    def add_node(self, name, screen_x, screen_y):
        x, y = self.to_canvas(screen_x, screen_y)
        self.nodes.append(Node(name, x, y, self.next_id))
        self.next_id += 1

    # Returns the top node under a screen position, if any
    def node_at(self, screen_x, screen_y):
        for node in reversed(self.nodes):
            if node.screen_rect(self).collidepoint(screen_x, screen_y):
                return node
        return None

    # Returns the node whose ? help badge is under a screen position, if it isn't covered by another node
    def help_at(self, screen_x, screen_y):
        node = self.node_at(screen_x, screen_y)
        if node and node.help_clicked(self, screen_x, screen_y):
            return node
        return None

    # Returns (node, "input" or "output", port number) for the port under a screen position
    def port_at(self, screen_x, screen_y):
        radius = max(PORT_RADIUS * self.zoom, 6) + 3

        for node in reversed(self.nodes):
            for index, (x, y) in enumerate(node.output_positions(self)):
                if (x - screen_x) ** 2 + (y - screen_y) ** 2 <= radius ** 2:
                    return node, "output", index
            for index, (x, y) in enumerate(node.input_positions(self)):
                if (x - screen_x) ** 2 + (y - screen_y) ** 2 <= radius ** 2:
                    return node, "input", index

        return None

    # Moves a node to the end of the list so it draws on top
    def bring_to_front(self, node):
        self.nodes.remove(node)
        self.nodes.append(node)

    # Removes a node along with every connection to or from it
    def remove_node(self, node):
        self.nodes.remove(node)
        targets = [c.target for c in self.connections if c.source is node]
        self.connections = [
            c for c in self.connections if c.source is not node and c.target is not node
        ]
        for target in targets:
            self.tidy_inputs(target)

    def disconnect(self, connection):
        self.connections.remove(connection)
        self.tidy_inputs(connection.target)

    # Numbers a growing node's connected inputs 1, 2, 3... in the order they were, closing any gap,
    # and leaves one empty input after them to connect the next one to. Other nodes are left alone
    def tidy_inputs(self, node):
        if not node.growing:
            return
        into = sorted((c for c in self.connections if c.target is node), key=lambda c: c.input_index)
        for index, connection in enumerate(into):
            connection.input_index = index
        node.inputs = len(into) + 1

    # Returns the connection going into one input port, if there is one
    def connection_into(self, node, input_index):
        for connection in self.connections:
            if connection.target is node and connection.input_index == input_index:
                return connection
        return None

    # Connects an output to an input, replacing whatever was already plugged into that input.
    # On a growing node it goes in at that input instead, moving the ones from there down one,
    # so dragging a line to another input reorders them.
    # Refuses connections from a node to itself, or ones that would create a loop
    def connect(self, source, output_index, target, input_index):
        if source is target or reaches(self.connections, target, source):
            return False

        existing = self.connection_into(target, input_index)
        if existing and target.growing:
            for connection in self.connections:
                if connection.target is target and connection.input_index >= input_index:
                    connection.input_index += 1
        elif existing:
            self.connections.remove(existing)

        self.connections.append(Connection(source, output_index, target, input_index))
        self.tidy_inputs(target)
        return True

    # Zooms by a factor while keeping the point (x, y) fixed on screen
    def zoom_at(self, factor, x, y):
        canvas_x, canvas_y = self.to_canvas(x, y)

        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        self.offset_x = x - canvas_x * self.zoom
        self.offset_y = y - canvas_y * self.zoom

    # Button zooms use the centre of the canvas
    def zoom_in(self):
        self.zoom_at(ZOOM_STEP, self.rect.centerx, self.rect.centery)

    def zoom_out(self):
        self.zoom_at(1 / ZOOM_STEP, self.rect.centerx, self.rect.centery)

    # Scroll wheel zooms towards the mouse pointer, and holding the left button
    # on empty background drags the whole canvas
    def handle_event(self, event):
        self.handle_pan(event)

        if event.type == pygame.MOUSEWHEEL:
            mouse_x, mouse_y = display.mouse_pos()
            if self.rect.collidepoint(mouse_x, mouse_y):
                if event.y > 0:
                    self.zoom_at(ZOOM_STEP, mouse_x, mouse_y)
                elif event.y < 0:
                    self.zoom_at(1 / ZOOM_STEP, mouse_x, mouse_y)

    # Starts a pan when the background is pressed, moves the view with the mouse, and stops on release
    def handle_pan(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            on_background = not self.node_at(x, y) and not self.port_at(x, y)
            if self.rect.collidepoint(event.pos) and on_background:
                self.panning = True
                set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)

        if event.type == pygame.MOUSEMOTION and self.panning:
            self.offset_x += event.rel[0]
            self.offset_y += event.rel[1]

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.panning:
            self.panning = False
            set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    # Remakes the font only when the zoom changes its size
    def update_font(self):
        size = max(8, round(FONT_SIZE * self.zoom))
        if size != self.font_size:
            self.font_size = size
            self.font = pygame.font.Font(None, size)

    # Width of connection lines at the current zoom
    def line_width(self):
        return max(2, round(3 * self.zoom))

    # Draws the grid, then connections, then nodes, kept inside the canvas area
    def draw(self, screen):
        self.update_font()
        screen.set_clip(self.rect)

        spacing = GRID_SIZE * self.zoom

        x = self.offset_x % spacing
        while x < self.rect.right:
            pygame.draw.line(screen, colours.GRID, (x, self.rect.top), (x, self.rect.bottom))
            x += spacing

        y = self.offset_y % spacing
        while y < self.rect.bottom:
            pygame.draw.line(screen, colours.GRID, (self.rect.left, y), (self.rect.right, y))
            y += spacing

        for connection in self.connections:
            start = connection.source.output_positions(self)[connection.output_index]
            end = connection.target.input_positions(self)[connection.input_index]
            pygame.draw.line(screen, colours.CONNECTION, start, end, self.line_width())

        for node in self.nodes:
            node.draw(screen, self)

        screen.set_clip(None)

    # Shows a node following the mouse while it is being dragged
    def draw_preview(self, screen, name, screen_x, screen_y):
        x, y = self.to_canvas(screen_x, screen_y)
        Node(name, x, y).draw(screen, self)
