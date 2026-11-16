from engine import draw


# What a module's settings panel holds, kept apart from how it is shown. A module fills a Panel
# with buttons, info lines, an optional text box, rows of options, a data preview or chart, and a
# note, and changes them as its buttons are clicked. The window shows whatever the panel holds
# (see ui/panel_view.py), and looks again after every click and keystroke. Nothing here needs a
# window, so a module's panel can be opened and its buttons pressed in a test.

NOT_ALLOWED = '\\/:*?"<>|'
MAX_OPTION_ROWS = 8
OPTION_COLUMNS = 2


class Button:
    # A button on a panel. text is its label, which a module can change, selected highlights it,
    # and value is whatever the module wants to find on it when it is clicked: for an option's
    # button it starts as the row's label
    def __init__(self, text, value=None, selected=False):
        self.text = text
        self.value = value
        self.selected = selected


class TextField:
    # A one-line text box. on_change(panel, text) runs after every edit.
    # blocked lists the characters that can't be typed, by default those not allowed in file names.
    # cursor is where typing goes in the text. focused asks for the box to be the one typed in.
    # fill makes the box as wide as the panel
    def __init__(self, label, text, on_change, blocked=NOT_ALLOWED, fill=False):
        self.label = label
        self.text = text
        self.on_change = on_change
        self.blocked = blocked
        self.fill = fill
        self.focused = False
        self.cursor = len(text)

    # The text with the characters that can't be typed taken out
    def allowed(self, text):
        return "".join(c for c in text if c not in self.blocked)

    # Puts text in at the cursor and moves the cursor to its end
    def insert(self, text):
        text = self.allowed(text)
        self.text = self.text[:self.cursor] + text + self.text[self.cursor:]
        self.cursor += len(text)

    # Replaces the whole text, with the cursor at the given place or the end
    def set_text(self, text, cursor=None):
        self.text = text
        self.cursor = len(text) if cursor is None else max(0, min(cursor, len(text)))


class Panel:
    # The settings for one node:
    #   buttons        the top row, added with add_button
    #   text_field     the text box, or None, set with add_text_field
    #   lines          lines of information under the top row
    #   options        rows of (label, [buttons]) added with add_option, e.g. a variable with Y and X
    #   options_title  gives the options a heading of their own, max_option_rows tall
    #   preview        a table shown under the preview_title heading, or None
    #   chart          a function(painter, area) drawn in place of the table, or None. A panel
    #                  with a chart gets Settings and Export PNG buttons from the window.
    #                  chart_axes names the axes whose scale can be set there: "xy" unless the
    #                  module says otherwise, "y" for a chart of categories, "" for one with no axes
    #   message        shown when there is neither a table nor a chart
    #   note           shown at the bottom of the panel
    #   refresh        a function(panel) the app calls after the workflow runs
    #   next_panel     set to another panel to show it in place of this one
    #   closed         set to True to close the panel
    # canvas is the workflow the node is in, set when the panel opens, so a module can reach
    # the other nodes and the connections. The inputs & outputs help is read from the node's
    # module, so modules don't add it
    def __init__(self, node, title):
        self.node = node
        self.canvas = None
        self.title = title
        self.font = draw.font(26)
        self.table_font = draw.font(22)
        self.heading_font = draw.font(32)

        self.buttons = []
        self.actions = {}
        self.text_field = None
        self.lines = []
        self.options = []
        self.options_title = None
        self.max_option_rows = MAX_OPTION_ROWS
        self.option_columns = OPTION_COLUMNS
        self.preview_title = "Data Preview"
        self.preview = None
        self.chart = None
        self.chart_axes = "xy"
        self.note = ""
        self.message = ""
        self.refresh = None
        self.closed = False
        self.next_panel = None

    # Adds a button to the top row. action(panel, button) runs when it is clicked
    def add_button(self, label, action):
        button = Button(label)
        self.buttons.append(button)
        self.actions[button] = action
        return button

    # Adds a labelled row of small toggle buttons, e.g. a variable with "Y" and "X".
    # choices is a list of (text, action, selected). Each button's value is the row's label
    def add_option(self, label, choices):
        buttons = []
        for text, action, selected in choices:
            button = Button(text, label, selected)
            self.actions[button] = action
            buttons.append(button)
        self.options.append((label, buttons))

    def clear_options(self):
        for label, buttons in self.options:
            for button in buttons:
                self.actions.pop(button, None)
        self.options = []

    def add_text_field(self, label, text, on_change, blocked=NOT_ALLOWED, fill=False):
        self.text_field = TextField(label, text, on_change, blocked, fill)

    # Clicks a button the way the window does, running its action
    def click(self, button):
        self.actions[button](self, button)

    # The button with the given label, looked for in the top row and then the options
    def button(self, text, label=None):
        for button in self.buttons:
            if button.text == text and label is None:
                return button
        for row_label, buttons in self.options:
            for button in buttons:
                if button.text == text and label in (None, row_label):
                    return button
        return None
