from engine import workspace


# The undo history of one workflow: a list of its states, oldest first, and which one it is in now.
# Nothing has to say what changed. record() is called whenever something may have, and it adds a
# state only if the workflow differs from the last one kept, so every kind of change is covered
# the same way: moving a node, drawing a line, or a setting changed in a panel
LIMIT = 200


class History:
    def __init__(self, workflow):
        self.workflow = workflow
        self.states = [workspace.state(workflow)]
        self.texts = [workspace.snapshot(workflow)]
        self.position = 0

    # Keeps the workflow's state if it has changed since the last one kept, dropping any states
    # that had been undone. Returns True if it had changed
    def record(self):
        text = workspace.snapshot(self.workflow)
        if text == self.texts[self.position]:
            return False
        del self.states[self.position + 1:]
        del self.texts[self.position + 1:]
        self.states.append(workspace.state(self.workflow))
        self.texts.append(text)
        if len(self.states) > LIMIT:
            del self.states[0]
            del self.texts[0]
        self.position = len(self.states) - 1
        return True

    def can_undo(self):
        return self.position > 0

    def can_redo(self):
        return self.position < len(self.states) - 1

    # Puts the workflow back to how it was before the last change. Returns False if there is
    # nothing to undo
    def undo(self):
        self.record()
        if not self.can_undo():
            return False
        self.position -= 1
        workspace.restore(self.workflow, self.states[self.position])
        return True

    # Makes the last change that was undone again. Returns False if there is none
    def redo(self):
        if self.record() or not self.can_redo():
            return False
        self.position += 1
        workspace.restore(self.workflow, self.states[self.position])
        return True
