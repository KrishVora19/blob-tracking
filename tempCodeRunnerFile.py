
    def draw(self, frame, text, pos):
        cv2.putText(
            frame,
            text,
            pos,
            self.font,
            self.font_scale,
            self.color,
            self.font_thickness,
            cv2.LINE_AA
        )