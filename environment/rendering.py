import pygame


class HospitalRenderer:
    def __init__(self, width=1280, height=760):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Hospital Triage RL Simulation")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)
        self.small_font = pygame.font.SysFont("arial", 18)
        self.title_font = pygame.font.SysFont("arial", 30, bold=True)

        self.colors = {
            "bg": (230, 236, 242),
            "room": (244, 248, 252),
            "panel": (250, 252, 255),
            "border": (70, 90, 110),
            "bed": (210, 220, 230),
            "sheet": (235, 240, 245),
            "station": (170, 200, 255),
            "nurse": (40, 110, 255),
            "mild": (70, 180, 90),
            "moderate": (245, 190, 60),
            "critical": (225, 70, 70),
            "text": (25, 25, 25),
            "muted": (150, 150, 150),
            "shadow": (190, 198, 208),
            "white": (255, 255, 255),
            "floor_line": (210, 217, 224),
            "monitor": (30, 40, 50),
            "monitor_glow": (90, 220, 120),
        }

        self.bed_positions = [
            (180, 110),
            (180, 270),
            (180, 430),
            (180, 590),
        ]

        self.station_position = (720, 350)
        self.agent_draw_x = self.station_position[0]
        self.agent_draw_y = self.station_position[1]

    def _severity_color(self, severity):
        if severity == 0:
            return self.colors["mild"]
        if severity == 1:
            return self.colors["moderate"]
        return self.colors["critical"]

    def _severity_text(self, severity):
        if severity == 0:
            return "Mild"
        if severity == 1:
            return "Moderate"
        return "Critical"

    def _draw_background(self):
        self.screen.fill(self.colors["bg"])

        pygame.draw.rect(self.screen, self.colors["room"], (20, 20, 840, 720), border_radius=20)
        pygame.draw.rect(self.screen, self.colors["border"], (20, 20, 840, 720), 3, border_radius=20)

        pygame.draw.rect(self.screen, self.colors["panel"], (890, 20, 370, 720), border_radius=20)
        pygame.draw.rect(self.screen, self.colors["border"], (890, 20, 370, 720), 3, border_radius=20)

        for y in [170, 330, 490, 650]:
            pygame.draw.line(self.screen, self.colors["floor_line"], (60, y), (820, y), 2)

        title = self.title_font.render("Emergency Room", True, self.colors["text"])
        self.screen.blit(title, (40, 35))

        info_title = self.title_font.render("Simulation Dashboard", True, self.colors["text"])
        self.screen.blit(info_title, (920, 35))

    def _draw_station(self):
        x, y = self.station_position

        pygame.draw.rect(self.screen, self.colors["shadow"], (x - 70, y - 35, 150, 120), border_radius=16)
        pygame.draw.rect(self.screen, self.colors["station"], (x - 80, y - 45, 150, 120), border_radius=16)
        pygame.draw.rect(self.screen, self.colors["border"], (x - 80, y - 45, 150, 120), 2, border_radius=16)

        label = self.font.render("Nurse Station", True, self.colors["text"])
        self.screen.blit(label, (x - 58, y - 6))

    def _draw_patient(self, bx, by, bed):
        color = self._severity_color(bed["severity"])

        pygame.draw.ellipse(self.screen, self.colors["white"], (bx + 72, by + 24, 70, 34))
        pygame.draw.circle(self.screen, (245, 218, 190), (bx + 82, by + 41), 13)
        pygame.draw.rect(self.screen, color, (bx + 96, by + 30, 78, 24), border_radius=8)

    def _draw_bed(self, bx, by, bed_index, bed):
        pygame.draw.rect(self.screen, self.colors["shadow"], (bx + 6, by + 6, 290, 105), border_radius=16)
        pygame.draw.rect(self.screen, self.colors["bed"], (bx, by, 290, 105), border_radius=16)
        pygame.draw.rect(self.screen, self.colors["border"], (bx, by, 290, 105), 2, border_radius=16)

        bed_label = self.font.render(f"Bed {bed_index + 1}", True, self.colors["text"])
        self.screen.blit(bed_label, (bx + 16, by + 10))

        if bed is None or bed.get("active", 0) == 0:
            empty_text = self.small_font.render("Empty Bed", True, self.colors["muted"])
            self.screen.blit(empty_text, (bx + 16, by + 54))
            return

        self._draw_patient(bx, by, bed)

        sev_text = self.small_font.render(self._severity_text(bed["severity"]), True, self.colors["text"])
        wait_text = self.small_font.render(f"Wait: {bed['waiting_time']}", True, self.colors["text"])

        self.screen.blit(sev_text, (bx + 16, by + 40))
        self.screen.blit(wait_text, (bx + 16, by + 62))

    def _draw_beds(self, beds):
        for i, (bx, by) in enumerate(self.bed_positions):
            self._draw_bed(bx, by, i, beds[i])

    def _get_target_position(self, nurse_position):
        if nurse_position == 0:
            return self.station_position
        bed_x, bed_y = self.bed_positions[nurse_position - 1]
        return (bed_x + 330, bed_y + 50)

    def _draw_nurse(self, nurse_position):
        target_x, target_y = self._get_target_position(nurse_position)

        speed = 10
        if abs(self.agent_draw_x - target_x) > speed:
            self.agent_draw_x += speed if self.agent_draw_x < target_x else -speed
        else:
            self.agent_draw_x = target_x

        if abs(self.agent_draw_y - target_y) > speed:
            self.agent_draw_y += speed if self.agent_draw_y < target_y else -speed
        else:
            self.agent_draw_y = target_y

        x = int(self.agent_draw_x)
        y = int(self.agent_draw_y)

        pygame.draw.circle(self.screen, (245, 218, 190), (x, y - 18), 12)
        pygame.draw.rect(self.screen, self.colors["nurse"], (x - 14, y - 8, 28, 36), border_radius=10)

    def _draw_dashboard(self, info, last_action):
        x = 920
        y = 95
        gap = 48

        items = [
            f"Step: {info.get('current_step', 0)}",
            f"Treated: {info.get('patients_treated', 0)}",
            f"Lost: {info.get('patients_lost', 0)}",
            f"Total Reward: {round(info.get('total_reward', 0.0), 2)}",
            f"Last Action: {last_action}",
        ]

        for item in items:
            text = self.font.render(item, True, self.colors["text"])
            self.screen.blit(text, (x, y))
            y += gap

    def render(self, beds, nurse_position, info, last_action="None"):
        # IMPORTANT: No event handling here anymore

        self._draw_background()
        self._draw_station()
        self._draw_beds(beds)
        self._draw_nurse(nurse_position)
        self._draw_dashboard(info, last_action)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()