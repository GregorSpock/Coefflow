import pygame

from engine import significance
from ui import colours, dialogs, display
from ui.button import Button


PANEL_WIDTH = 720
PANEL_HEIGHT = 560
BUTTON_WIDTH = 120
ROW_HEIGHT = 40
PADDING = 10
CELL_HEIGHT = 26
OPTION_COLUMNS = 2
OPTION_HEIGHT = 32
OPTION_BUTTON = 30
MAX_OPTION_ROWS = 8
SCROLLBAR_WIDTH = 6
MIN_COLUMN_WIDTH = 100
TABLE_SCROLL_ROWS = 3
MIN_THUMB = 20
TOGGLE_SIZE = 20
# How far one notch of Shift + wheel slides option labels sideways, in pixels
LABEL_SCROLL_STEP = 40
NOT_ALLOWED = '\\/:*?"<>|'


# Shortens text with "..." until it fits inside a width
def fit_text(font, text, width):
    if font.size(text)[0] <= width:
        return text
    while text and font.size(text + "...")[0] > width:
        text = text[:-1]
    return text + "..."


# Splits text into lines that each fit inside a width, breaking between words
def wrap_text(font, text, width):
    lines = []
    line = ""
    for word in text.split():
        attempt = f"{line} {word}".strip()
        if font.size(attempt)[0] <= width:
            line = attempt
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


# Turns a table value into short text, keeping decimals tidy
def cell_text(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


# Draws a scrollbar track with a thumb showing which part of a list is in view.
# first is the first item shown, shown is how many fit and total is how many there are.
# The thumb never gets shorter than MIN_THUMB so it stays visible in long lists
def draw_scrollbar(screen, track, first, shown, total):
    thumb = track.copy()
    along = max(0, total - shown)
    if track.height > track.width:
        thumb.height = max(MIN_THUMB, track.height * shown / total)
        thumb.top = track.top + (track.height - thumb.height) * first / max(1, along)
    else:
        thumb.width = max(MIN_THUMB, track.width * shown / total)
        thumb.left = track.left + (track.width - thumb.width) * first / max(1, along)

    pygame.draw.rect(screen, colours.TAB, track, border_radius=SCROLLBAR_WIDTH // 2)
    pygame.draw.rect(screen, colours.TAB_SELECTED, thumb, border_radius=SCROLLBAR_WIDTH // 2)


class TextField:
    # A one-line text box. on_change(panel, text) runs after every edit.
    # blocked lists the characters that can't be typed, by default those not allowed in file names.
    # cursor is where typing goes in the text, moved by clicking or the arrow keys. Text too long for
    # the box slides along so the cursor stays in view, shift being how far it has slid in pixels.
    # fill makes the box reach the right of the panel
    def __init__(self, label, text, on_change, blocked=NOT_ALLOWED, fill=False):
        self.label = label
        self.text = text
        self.on_change = on_change
        self.blocked = blocked
        self.fill = fill
        self.rect = pygame.Rect(0, 0, 300, 36)
        self.focused = False
        self.cursor = len(text)
        self.shift = 0

    # Puts text in at the cursor and moves the cursor to its end
    def insert(self, text):
        text = "".join(c for c in text if c not in self.blocked)
        self.text = self.text[:self.cursor] + text + self.text[self.cursor:]
        self.cursor += len(text)

    # Replaces the whole text, with the cursor at the given place or the end
    def set_text(self, text, cursor=None):
        self.text = text
        self.cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))


class Panel:
    # A settings window for one node. Modules fill it with buttons, info lines,
    # an optional text field, rows of options, a data preview or chart, and a note at the bottom.
    # chart is a function(screen, area) that draws into the space it is given.
    # The inputs & outputs section is read from the node's module, so modules don't add it.
    # canvas is set when the panel opens, so modules can reach the other nodes and connections.
    # A button can set next_panel to another panel, which then replaces this one.
    # The inputs & outputs section starts hidden, and showing or hiding it applies to every panel.
    # A module can give the options a heading with options_title, which makes them a section like the
    # preview: a fixed height of max_option_rows, shown or hidden with a box by the heading, and
    # scrolled down with the wheel and sideways with Shift + wheel to read labels too long to fit
    ports_expanded = False
    options_expanded = True

    def __init__(self, node, title):
        self.node = node
        self.canvas = None
        self.title = title
        self.rect = pygame.Rect(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        self.font = pygame.font.Font(None, 26)
        self.table_font = pygame.font.Font(None, 22)
        self.heading_font = pygame.font.Font(None, 32)

        self.buttons = []
        self.actions = {}
        self.close_button = Button(0, 0, ROW_HEIGHT, ROW_HEIGHT, "X", self.font, "EXIT", "EXIT_HOVER")
        self.text_field = None
        self.lines = []
        self.options = []
        self.preview_title = "Data Preview"
        self.preview = None
        self.chart = None
        self.note = ""
        self.message = ""
        self.refresh = None
        self.closed = False
        self.next_panel = None
        self.option_scroll = 0
        self.max_option_rows = MAX_OPTION_ROWS
        self.option_columns = OPTION_COLUMNS
        self.options_rect = pygame.Rect(0, 0, 0, 0)
        self.options_title = None
        self.options_toggle = pygame.Rect(0, 0, 0, 0)
        self.label_shift = 0
        self.label_overflow = 0
        self.visible_buttons = []
        self.table_row = 0
        self.table_column = 0
        self.table_rows_shown = 1
        self.table_columns_shown = 1
        self.table_rect = pygame.Rect(0, 0, 0, 0)
        self.ports_toggle = pygame.Rect(0, 0, 0, 0)

    # Adds a button to the top row. action(panel, button) runs when it is clicked
    def add_button(self, label, action):
        button = Button(0, 0, BUTTON_WIDTH, ROW_HEIGHT, label, self.font, "TAB", "TAB_HOVER")
        self.buttons.append(button)
        self.actions[button] = action
        self.layout()
        return button

    # Adds a labelled row of small toggle buttons, e.g. a variable with "Y" and "X".
    # choices is a list of (text, action, selected). Each button's value is the row's label
    def add_option(self, label, choices):
        buttons = []
        for text, action, selected in choices:
            width = max(OPTION_BUTTON, self.table_font.size(text)[0] + 12)
            button = Button(0, 0, width, OPTION_BUTTON - 4, text, self.table_font, "TAB", "TAB_HOVER", "TAB_SELECTED")
            button.selected = selected
            button.value = label
            self.actions[button] = action
            buttons.append(button)
        self.options.append((label, buttons))

    def clear_options(self):
        for label, buttons in self.options:
            for button in buttons:
                del self.actions[button]
        self.options = []
        self.visible_buttons = []
        self.options_rect = pygame.Rect(0, 0, 0, 0)

    def option_rows(self):
        return (len(self.options) + self.option_columns - 1) // self.option_columns

    # Scrolls the option list by a number of rows, without going past either end
    def scroll_options(self, rows):
        hidden_rows = max(0, self.option_rows() - self.max_option_rows)
        self.option_scroll = max(0, min(self.option_scroll + rows, hidden_rows))

    # Slides the option labels sideways by a number of pixels, without going past either end
    def scroll_labels(self, pixels):
        self.label_shift = max(0, min(self.label_shift + pixels, self.label_overflow))

    # Scrolls the preview table by a number of rows and columns, without going past either end
    def scroll_table(self, rows, columns):
        if self.preview is None:
            return
        hidden_rows = max(0, len(self.preview) - self.table_rows_shown)
        hidden_columns = max(0, len(self.preview.columns) - self.table_columns_shown)
        self.table_row = max(0, min(self.table_row + rows, hidden_rows))
        self.table_column = max(0, min(self.table_column + columns, hidden_columns))

    def add_text_field(self, label, text, on_change, blocked=NOT_ALLOWED, fill=False):
        self.text_field = TextField(label, text, on_change, blocked, fill)
        self.layout()

    # Moves the panel so its top left corner is at (x, y), with the given height
    def place(self, x, y, height):
        self.rect.topleft = (x, y)
        self.rect.height = height
        self.layout()

    # Positions the buttons, close button and text field inside the panel
    def layout(self):
        x = self.rect.left + PADDING
        for button in self.buttons:
            button.rect.topleft = (x, self.rect.top + PADDING)
            x += BUTTON_WIDTH + PADDING

        self.close_button.rect.topright = (self.rect.right - PADDING, self.rect.top + PADDING)

        if self.text_field:
            label_width = self.font.size(self.text_field.label)[0]
            self.text_field.rect.topleft = (
                self.rect.left + PADDING * 2 + label_width,
                self.rect.top + ROW_HEIGHT + PADDING * 3,
            )
            if self.text_field.fill:
                self.text_field.rect.width = self.rect.right - PADDING - self.text_field.rect.left

    # Handles clicks and typing. Returns True if the panel used the event
    def handle_event(self, event):
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL):
            position = getattr(event, "pos", display.mouse_pos())
            if not self.rect.collidepoint(position):
                if event.type == pygame.MOUSEBUTTONDOWN and self.text_field:
                    self.text_field.focused = False
                return False

        if self.close_button.is_clicked(event):
            self.closed = True
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.ports_toggle.collidepoint(event.pos):
            Panel.ports_expanded = not Panel.ports_expanded
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.options_toggle.collidepoint(event.pos):
            Panel.options_expanded = not Panel.options_expanded
            return True

        # Under an options heading, Shift + wheel or a sideways swipe slides the labels along
        if event.type == pygame.MOUSEWHEEL and self.options_rect.collidepoint(display.mouse_pos()):
            if self.options_title and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.scroll_labels((event.x - event.y) * LABEL_SCROLL_STEP)
            else:
                self.scroll_options(-event.y)
                if self.options_title:
                    self.scroll_labels(event.x * LABEL_SCROLL_STEP)
            return True

        # Over the table the wheel scrolls rows, and Shift + wheel or a sideways swipe scrolls columns
        if event.type == pygame.MOUSEWHEEL and self.table_rect.collidepoint(display.mouse_pos()):
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.scroll_table(0, event.x - event.y)
            else:
                self.scroll_table(-event.y * TABLE_SCROLL_ROWS, event.x)
            return True

        for button in self.buttons + self.option_buttons():
            if button.is_clicked(event):
                self.actions[button](self, button)
                return True

        if self.text_field:
            if self.handle_typing(event):
                return True

        return event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL)

    # Only the option buttons currently scrolled into view can be clicked
    def option_buttons(self):
        return self.visible_buttons

    # Focuses the text field when clicked, putting the cursor where it was clicked, then types into
    # it until Enter or a click elsewhere. The arrow keys, Home and End move the cursor,
    # Backspace and Delete remove the character before or after it, and Ctrl+V pastes
    def handle_typing(self, event):
        field = self.text_field

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            field.focused = field.rect.collidepoint(event.pos)
            if field.focused:
                field.cursor = self.cursor_at(event.pos[0])
            return field.focused

        if not field.focused:
            return False

        if event.type == pygame.TEXTINPUT:
            field.insert(event.text)
            field.on_change(self, field.text)
            return True

        if event.type == pygame.KEYDOWN:
            text, cursor = field.text, field.cursor
            if event.key == pygame.K_BACKSPACE and cursor > 0:
                field.set_text(text[:cursor - 1] + text[cursor:], cursor - 1)
            elif event.key == pygame.K_DELETE:
                field.set_text(text[:cursor] + text[cursor + 1:], cursor)
            elif event.key == pygame.K_LEFT:
                field.cursor = max(0, cursor - 1)
            elif event.key == pygame.K_RIGHT:
                field.cursor = min(len(text), cursor + 1)
            elif event.key == pygame.K_HOME:
                field.cursor = 0
            elif event.key == pygame.K_END:
                field.cursor = len(text)
            elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL:
                field.insert(" ".join(dialogs.paste_text().split()))
            elif event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                field.focused = False
            if field.text != text:
                field.on_change(self, field.text)
            return True

        return False

    # The place in the text field's text nearest a click at x
    def cursor_at(self, x):
        field = self.text_field
        start = field.rect.left + 8 - field.shift
        widths = [self.font.size(field.text[:i])[0] for i in range(len(field.text) + 1)]
        return min(range(len(widths)), key=lambda i: abs(start + widths[i] - x))

    # Draws the panel, top row, text field, info lines and preview table
    def draw(self, screen):
        pygame.draw.rect(screen, colours.TOOLBAR, self.rect, border_radius=colours.BUTTON_RADIUS)
        pygame.draw.rect(screen, colours.TOOLBAR_LINE, self.rect, width=2, border_radius=colours.BUTTON_RADIUS)

        for button in self.buttons:
            button.draw(screen)
        self.close_button.draw(screen)

        number = f"  #{self.node.id}" if self.node.id is not None else ""
        title = self.font.render(f"{self.title}{number}", True, colours.TEXT)
        title_rect = title.get_rect(midright=(self.close_button.rect.left - PADDING, self.close_button.rect.centery))
        screen.blit(title, title_rect)

        y = self.rect.top + ROW_HEIGHT + PADDING * 2
        pygame.draw.line(screen, colours.TOOLBAR_LINE, (self.rect.left, y), (self.rect.right, y))
        y += PADDING

        if self.text_field:
            self.draw_text_field(screen)
            y = self.text_field.rect.bottom + PADDING

        for line in self.lines:
            text = fit_text(self.font, line, self.rect.width - PADDING * 2)
            screen.blit(self.font.render(text, True, colours.TEXT), (self.rect.left + PADDING, y))
            y += 30

        if self.node.error:
            text = fit_text(self.font, f"Last run error: {self.node.error}", self.rect.width - PADDING * 2)
            screen.blit(self.font.render(text, True, colours.NODE_ERROR), (self.rect.left + PADDING, y))
            y += 30

        if self.options_title:
            y = self.draw_options_section(screen, y)
        elif self.options:
            y = self.draw_options(screen, y + PADDING)

        y = self.draw_ports(screen, y)

        y = self.draw_heading(screen, self.preview_title, y + PADDING)

        note_top = self.draw_note(screen)

        # A results table with significance stars says what the stars mean above it
        if self.preview is not None and "stars" in self.preview.columns:
            legend = self.table_font.render(f"Stars: {significance.legend()}", True, colours.TEXT)
            screen.blit(legend, (self.rect.left + PADDING, y))
            y += legend.get_height() + PADDING

        if self.preview is not None:
            self.draw_table(screen, y, note_top)
        elif self.chart:
            area = pygame.Rect(self.rect.left + PADDING, y, self.rect.width - PADDING * 2, note_top - y - PADDING)
            self.chart(screen, area)
        elif self.message:
            screen.blit(self.font.render(self.message, True, colours.TEXT), (self.rect.left + PADDING, y))

    # Draws an underlined section heading at y. Returns the y where the section's content starts
    def draw_heading(self, screen, text, y):
        heading = self.heading_font.render(text, True, colours.TEXT)
        screen.blit(heading, (self.rect.left + PADDING, y))
        y += heading.get_height() + 2
        pygame.draw.line(screen, colours.TEXT, (self.rect.left + PADDING, y), (self.rect.left + PADDING + heading.get_width(), y))
        return y + PADDING

    # Lists what the module takes in on each input port and gives out on each output port,
    # from its INPUT_SPECS and OUTPUT_SPECS: one (name, description) per port, in port order.
    # Descriptions wrap under their own column so the port labels stay lined up.
    # A box beside the heading shows or hides the list, so it doesn't crowd the preview below.
    # Draws nothing if the module has neither. Returns the y below the section
    def draw_ports(self, screen, top):
        module = self.node.module
        ports = [(f"In {i + 1}", spec) for i, spec in enumerate(getattr(module, "INPUT_SPECS", []))]
        if getattr(self.node, "growing", False):
            ports = [("In 1+", spec) for label, spec in ports]
        ports += [(f"Out {i + 1}", spec) for i, spec in enumerate(getattr(module, "OUTPUT_SPECS", []))]
        if not ports:
            self.ports_toggle = pygame.Rect(0, 0, 0, 0)
            return top

        heading = "Inputs & Outputs"
        y = self.draw_heading(screen, heading, top + PADDING)
        self.ports_toggle = self.draw_toggle(screen, heading, top + PADDING, Panel.ports_expanded)
        if not Panel.ports_expanded:
            return y - PADDING

        left = self.rect.left + PADDING
        label_width = self.table_font.size("Out 9")[0] + PADDING
        text_width = self.rect.width - PADDING * 2 - label_width

        for label, (name, description) in ports:
            screen.blit(self.table_font.render(label, True, colours.TEXT), (left, y))
            for line in wrap_text(self.table_font, f"{name}: {description}", text_width):
                screen.blit(self.table_font.render(line, True, colours.TEXT), (left + label_width, y))
                y += 20
        return y

    # Draws the show/hide box just right of the heading drawn at heading_top, and returns it.
    # A downward arrow shows the section, an upward one hides it, like the workflow tabs toggle
    def draw_toggle(self, screen, heading, heading_top, expanded):
        heading_width, heading_height = self.heading_font.size(heading)
        toggle = pygame.Rect(0, 0, TOGGLE_SIZE, TOGGLE_SIZE)
        toggle.midleft = (self.rect.left + PADDING * 2 + heading_width, heading_top + heading_height // 2)

        hovered = toggle.collidepoint(display.mouse_pos())
        pygame.draw.rect(screen, colours.ZOOM_BUTTON_HOVER if hovered else colours.ZOOM_BUTTON, toggle)
        pygame.draw.rect(screen, colours.TOOLBAR_LINE, toggle, 1)
        x, y = toggle.center
        if expanded:
            points = [(x - 5, y + 3), (x + 5, y + 3), (x, y - 3)]
        else:
            points = [(x - 5, y - 3), (x + 5, y - 3), (x, y + 3)]
        pygame.draw.polygon(screen, colours.TEXT, points)
        return toggle

    # Draws the options under their own heading with a show/hide box. Shown, they take a fixed
    # height of max_option_rows, with a line under them saying which part is in view when they
    # don't all fit. Returns the y below the section
    def draw_options_section(self, screen, top):
        y = self.draw_heading(screen, self.options_title, top + PADDING)
        self.options_toggle = self.draw_toggle(screen, self.options_title, top + PADDING, Panel.options_expanded)
        if not Panel.options_expanded:
            self.options_rect = pygame.Rect(0, 0, 0, 0)
            self.visible_buttons = []
            return y - PADDING

        bottom = self.draw_options(screen, y)
        hidden_rows = self.option_rows() > self.max_option_rows
        if not hidden_rows and not self.label_overflow:
            return bottom

        first = self.option_scroll + 1
        last = min(self.option_rows(), self.option_scroll + self.max_option_rows)
        text = f"Rows {first}-{last} of {self.option_rows()}    (scroll for rows, Shift + scroll to read long rows)"
        note = self.table_font.render(fit_text(self.table_font, text, self.rect.width - PADDING * 2), True, colours.TEXT)
        screen.blit(note, (self.rect.left + PADDING, bottom + 2))
        return bottom + note.get_height() + 2

    # Draws the option rows in a grid, placing their buttons as it goes. Only max_option_rows rows
    # fit at once; the rest are reached with the mouse wheel. Under an options heading the grid keeps
    # that height however many rows there are, and labels too long to fit are slid along by
    # label_shift rather than cut short. Returns the y below the grid
    def draw_options(self, screen, top):
        column_width = (self.rect.width - PADDING * 3 - SCROLLBAR_WIDTH) // self.option_columns
        shown_rows = min(self.option_rows(), self.max_option_rows)
        self.scroll_options(0)

        first = self.option_scroll * self.option_columns
        shown = self.options[first:first + shown_rows * self.option_columns]
        self.visible_buttons = []
        room = 0

        for i, (label, buttons) in enumerate(shown):
            cell_left = self.rect.left + PADDING + (i % self.option_columns) * column_width
            y = top + (i // self.option_columns) * OPTION_HEIGHT

            x = cell_left
            for button in buttons:
                button.rect.topleft = (x, y)
                button.draw(screen)
                x += button.rect.width + 4
            self.visible_buttons.extend(buttons)

            room = cell_left + column_width - x - PADDING
            middle = y + (OPTION_BUTTON - 4) // 2
            if self.options_title:
                rendered = self.font.render(label, True, colours.TEXT)
                area = pygame.Rect(x + 4, y, room, OPTION_HEIGHT)
                clip = screen.get_clip()
                screen.set_clip(area)
                screen.blit(rendered, rendered.get_rect(midleft=(x + 4 - self.label_shift, middle)))
                screen.set_clip(clip)
            else:
                text = fit_text(self.font, label, room)
                rendered = self.font.render(text, True, colours.TEXT)
                screen.blit(rendered, rendered.get_rect(midleft=(x + 4, middle)))

        # How far the longest label in the whole list runs past its room, so the labels slide
        # the same way whichever rows are in view
        if self.options_title:
            shown_rows = self.max_option_rows
            longest = max((self.font.size(label)[0] for label, buttons in self.options), default=0)
            self.label_overflow = max(0, longest - room) if shown else 0
            self.scroll_labels(0)

        height = shown_rows * OPTION_HEIGHT
        self.options_rect = pygame.Rect(self.rect.left + PADDING, top, self.rect.width - PADDING * 2, height)
        if self.option_rows() > self.max_option_rows:
            track = pygame.Rect(0, 0, SCROLLBAR_WIDTH, self.options_rect.height - 4)
            track.topright = (self.options_rect.right, self.options_rect.top)
            draw_scrollbar(screen, track, self.option_scroll, self.max_option_rows, self.option_rows())
        if self.label_overflow and self.options_title:
            track = pygame.Rect(self.options_rect.left, top + height, self.options_rect.width - SCROLLBAR_WIDTH - 4, SCROLLBAR_WIDTH)
            width = track.width
            draw_scrollbar(screen, track, self.label_shift, width, width + self.label_overflow)
            height += SCROLLBAR_WIDTH + 4
        return top + height

    # Draws the note at the bottom of the panel, wrapped over as many lines as needed.
    # Returns the y where the note starts, so the content above can stop there
    def draw_note(self, screen):
        if not self.note:
            return self.rect.bottom - PADDING

        lines = wrap_text(self.font, self.note, self.rect.width - PADDING * 2)
        top = self.rect.bottom - PADDING - len(lines) * 28
        pygame.draw.line(screen, colours.TOOLBAR_LINE, (self.rect.left, top - PADDING), (self.rect.right, top - PADDING))

        for i, line in enumerate(lines):
            screen.blit(self.font.render(line, True, colours.TEXT), (self.rect.left + PADDING, top + i * 28))
        return top - PADDING

    # Draws the text field's label, box and current text, with a cursor when focused
    def draw_text_field(self, screen):
        field = self.text_field
        label = self.font.render(field.label, True, colours.TEXT)
        screen.blit(label, label.get_rect(midleft=(self.rect.left + PADDING, field.rect.centery)))

        border = colours.TAB_SELECTED if field.focused else colours.TOOLBAR_LINE
        pygame.draw.rect(screen, colours.BACKGROUND, field.rect, border_radius=colours.BUTTON_RADIUS)
        pygame.draw.rect(screen, border, field.rect, width=2, border_radius=colours.BUTTON_RADIUS)

        # Slides long text along just enough to keep the cursor inside the box
        inner = field.rect.inflate(-16, -4)
        cursor_x = self.font.size(field.text[:field.cursor])[0]
        field.shift = max(0, min(field.shift, cursor_x), cursor_x - inner.width + 2)

        rendered = self.font.render(field.text, True, colours.TEXT)
        clip = screen.get_clip()
        screen.set_clip(inner)
        screen.blit(rendered, rendered.get_rect(midleft=(inner.left - field.shift, field.rect.centery)))
        if field.focused:
            x = inner.left - field.shift + cursor_x
            pygame.draw.line(screen, colours.TEXT, (x, inner.top + 4), (x, inner.bottom - 4), 2)
        screen.set_clip(clip)

    # Draws the column names and as many rows and columns as fit above bottom. If there are more,
    # the mouse wheel scrolls through them and scrollbars show which part of the table is in view
    def draw_table(self, screen, top, bottom):
        left = self.rect.left + PADDING
        width = self.rect.width - PADDING * 2
        cells_width = width - SCROLLBAR_WIDTH - 4
        total_rows, total_columns = self.preview.shape

        # Room below the header for the rows, leaving space for the bottom scrollbar and position line
        self.table_columns_shown = max(1, min(total_columns, cells_width // MIN_COLUMN_WIDTH))
        self.table_rows_shown = max(1, min(total_rows, (bottom - top - SCROLLBAR_WIDTH - 4) // CELL_HEIGHT - 2))
        self.scroll_table(0, 0)

        data = self.preview.iloc[
            self.table_row:self.table_row + self.table_rows_shown,
            self.table_column:self.table_column + self.table_columns_shown,
        ]
        column_width = cells_width // self.table_columns_shown

        for i, column in enumerate(data.columns):
            x = left + i * column_width
            text = fit_text(self.table_font, str(column), column_width - 8)
            screen.blit(self.table_font.render(text, True, colours.TEXT), (x + 4, top))

        pygame.draw.line(screen, colours.TOOLBAR_LINE, (left, top + CELL_HEIGHT - 4), (left + cells_width, top + CELL_HEIGHT - 4))

        for row in range(len(data)):
            y = top + (row + 1) * CELL_HEIGHT
            for i in range(len(data.columns)):
                x = left + i * column_width
                text = fit_text(self.table_font, cell_text(data.iloc[row, i]), column_width - 8)
                screen.blit(self.table_font.render(text, True, colours.TEXT), (x + 4, y))

        rows_bottom = top + (self.table_rows_shown + 1) * CELL_HEIGHT
        self.table_rect = pygame.Rect(left, top, width, rows_bottom + SCROLLBAR_WIDTH + 4 - top)

        hidden_rows = total_rows > self.table_rows_shown
        hidden_columns = total_columns > self.table_columns_shown
        if hidden_rows:
            track = pygame.Rect(left + width - SCROLLBAR_WIDTH, top + CELL_HEIGHT, SCROLLBAR_WIDTH, rows_bottom - top - CELL_HEIGHT)
            draw_scrollbar(screen, track, self.table_row, self.table_rows_shown, total_rows)
        if hidden_columns:
            track = pygame.Rect(left, rows_bottom, cells_width, SCROLLBAR_WIDTH)
            draw_scrollbar(screen, track, self.table_column, self.table_columns_shown, total_columns)

        # Says which part of the table is in view, and how to see the rest
        if hidden_rows or hidden_columns:
            last_row = self.table_row + len(data)
            last_column = self.table_column + len(data.columns)
            text = (
                f"Rows {self.table_row + 1}-{last_row} of {total_rows}    "
                f"Columns {self.table_column + 1}-{last_column} of {total_columns}    "
                f"(scroll for rows, Shift + scroll for columns)"
            )
            note = self.table_font.render(fit_text(self.table_font, text, width), True, colours.TEXT)
            screen.blit(note, (left, rows_bottom + SCROLLBAR_WIDTH + 6))
