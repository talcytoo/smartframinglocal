class MouseState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.buttons = 0

    def callback(self, event, x, y, flags, userdata=None):
        self.x, self.y = x, y
        self.buttons = flags