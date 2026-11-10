# Helpers for fitting text into a width, shared by the panels and the charts


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
