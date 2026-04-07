import pygame
import math

screen_width = 1500
screen_height = 800
check_point = ((1370, 675), (1370, 215), (935, 465), (630, 180), (320, 160), (130, 675), (550, 702)) # race_track_ie.png
"""
Area for text boxes:
750,0 - 1500,100
1055,290 - 1300,605
"""

class Car:
    def __init__(self, car_file, map, pos): # map_file
        # self.map = pygame.image.load(map_file)
        self.map = map                                                    # the track image (used for collision detection)
        self.surface = pygame.image.load(car_file)
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = pos                                                     # [x, y] position on screen
        self.angle = 0                                                     # facing direction in degrees
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []                                                   # sensor readings (cleared and recalculated each step)
        self.radars_for_draw = []
        self.is_alive = True                                               # False = crashed
        self.goal = False                                                  # True = completed a lap
        self.distance = 0                                                  # total distance traveled
        self.time_spent = 0
        self.current_check = 0                                             # which checkpoint we're heading toward
        # Distance-to-next-checkpoint tracking (used for shaped progress reward in v3/v4).
        # Initialize these to the *actual* distance to checkpoint 0 so the first step of each
        # episode doesn't get a huge negative progress spike.
        self.cur_distance = get_distance(check_point[self.current_check], self.center)
        self.prev_distance = self.cur_distance
        self.check_flag = False
        """
        for d in range(-90, 120, 45): self.check_radar(d)
        for d in range(-90, 105, 15): self.check_radar_for_draw(d)
        """

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars: # or self.radars_for_draw
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def pixel_at(self,x,y):
        try:
            return self.map.get_at((x,y))
        except:
            return (255, 255, 255, 255)

    def check_collision(self, map=None):  # track image has white pixels for walls/grass and colored pixels for the road
        self.is_alive = True
        for p in self.four_points:
            if self.pixel_at(int(p[0]), int(p[1])) == (255, 255, 255, 255): # computes its 4 corner points and checks if any of them land on a white pixel
                self.is_alive = False
                break

    def check_radar(self, degree, map=None): # casts a ray from the car's center at a specific angle and counts how far it goes before hitting a white pixel
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len) # 5 radars are cast at angles -90°, -45°, 0°, +45°, +90° relative to the car's heading — so the car "sees" left, front-left, front, front-right, and right.
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.pixel_at(x, y) == (255, 255, 255, 255) and len < 200:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])
    """
    #------------------------------------------------------------------------------
    def draw_collision(self, screen):
        for i in range(4):
            x = int(self.four_points[i][0])
            y = int(self.four_points[i][1])
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 5)

    def check_radar_for_draw(self, degree, map=None):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 2000:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars_for_draw.append([(x, y), dist])
    """
    def check_checkpoint(self): # 7 checkpoints defined as coordinates. The car must pass near each one in order. 
        p = check_point[self.current_check]
        dist = get_distance(p, self.center)


        self.prev_distance = self.cur_distance

        if dist < 70:  # within 70 pixels = checkpoint reached
            self.current_check += 1
            self.check_flag = True
            if self.current_check >= len(check_point):
                self.current_check = 0
                self.goal = True  # full lap completed!
            else:
                self.goal = False

            # Reset distance-to-target tracking to the next checkpoint.
            next_p = check_point[self.current_check]
            next_dist = get_distance(next_p, self.center)
            self.cur_distance = next_dist
            self.prev_distance = next_dist
            return

        self.cur_distance = dist
    #------------------------------------------------------------------------------


    def update(self,map=None, min_speed=1): # pysics
        #check speed
        self.speed -= 0.5                   # friction: constant deceleration
        if self.speed > 10: self.speed = 10 # speed cap
        if self.speed < min_speed:  self.speed = min_speed  # minimum speed depends on the environment version
        
        # required for NEAT
        if map is not None:
            self.speed = 7 # NEAT

        #check position
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # caculate 4 collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        # required for NEAT
        if map is not None:
            self.check_collision(self.map)
            self.radars.clear()
            for d in range(-90, 120, 45):
                self.check_radar(d, self.map)

        """
        self.car.radars_for_draw.clear()
        for d in range(-90, 105, 15):
            self.car.check_radar_for_draw(d)
        pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        
        self.car.draw_collision(self.screen)
        # self.car.draw_radar(self.screen) # moved to car.draw()
        self.car.draw(self.screen)
        """

    #-------------------------------------------------------------------
    # required for NEAT
    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)
        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self):
        return self.distance / 50.0
    #-------------------------------------------------------------------

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


class PyRace2D:
    def __init__(self, is_render = True, car = True, mode = 0, version="v1"):
        # print('PyRace2D - INIT ENVIRONMENT')
        pygame.init()
        # Headless mode: when is_render=False, avoid creating a visible display window.
        # This makes SB3 training more stable and allows running without UI.
        if is_render or mode == 2:
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        else:
            self.screen = pygame.Surface((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.map = pygame.image.load('race_track_ie.png')
        self.cars = []
        self.version = version
        self.min_speed = 0 if self.version in ("v3", "v4") else 1
        if car:
            self.car = Car('car.png', self.map, [500, 650])
            self.cars.append(self.car)
        self.game_speed = 60*0 # as fast as possible...
        self.is_render = is_render
        self.mode = mode # 0: normal, 1:dark, 2: normal (force display)
        # Tracks consecutive "stall" steps (no progress + very low speed).
        # Helps prevent policies that just rotate in place to avoid crashing.
        self.stall_steps = 0

    def action(self, action): # translates integer actions to car physics:
        if self.version == "v4":
            steer = float(action[0])
            throttle = float(action[1])
            if steer > 1.0:
                steer = 1.0
            elif steer < -1.0:
                steer = -1.0
            if throttle > 1.0:
                throttle = 1.0
            elif throttle < -1.0:
                throttle = -1.0

            self.car.angle += 5.0 * steer
            self.car.speed += 2.0 * throttle
        else:
            if action == 0:
                self.car.speed += 2
            elif action == 1:
                self.car.angle += 5
            elif action == 2:
                self.car.angle -= 5
            elif action == 3 and self.version == "v3":
                self.car.speed -= 2

        self.car.update(min_speed=self.min_speed)
        self.car.check_collision()
        self.car.check_checkpoint()

        self.car.radars.clear()
        for d in range(-90, 120, 45):   # recalculate all 5 radar sensors
            self.car.check_radar(d)

    def evaluate(self): # reward function
        if self.version not in ("v3", "v4"):
            reward = 0
            """
            if self.car.check_flag:
                self.car.check_flag = False
                reward = 2000 - self.car.time_spent
                self.car.time_spent = 0
            """
            if not self.car.is_alive: # crash
                reward = -10000 + self.car.distance

            elif self.car.goal: # full lap
                # reward = 10000*(1+self.car.current_check)/len(check_point)
                reward = 10000
                # print('goal',self.car.current_check,len(check_point))
            return reward # everything else: 0

        reward = -0.1 # small per-step penalty to discourage stalling
        reward += 0.02 * self.car.speed # moving is better than standing still

        # Lane-centering shaping: penalize imbalance between left and right radar distances.
        # Indices: 0=left, 1=front-left, 2=front, 3=front-right, 4=right.
        # if len(self.car.radars) >= 5:
            # left_dist = float(self.car.radars[0][1])
            # right_dist = float(self.car.radars[4][1])
            # front_left = float(self.car.radars[1][1])
            # front = float(self.car.radars[2][1])
            # front_right = float(self.car.radars[3][1])

            # reward -= 0.02 * abs((left_dist - right_dist)/200)
            # reward -= 0.01 * abs((front_left - front_right)/200)

            # # Shape for forward corridor vs side corridor:
            # # Encourage (left + right) to be smaller than (front-left + front-right).
            # # Intuition: prefer having more space in front-diagonals than on the sides.
            # side_sum = left_dist + right_dist          # 0..400
            # diag_sum = front_left + front_right        # 0..400
            # balance = (diag_sum - side_sum) / 400.0    # approx [-1, 1]
            # reward += 0.10 * max(-1.0, min(1.0, balance))

            # Slight penalty for being too close to the wall for front-left/front/front-right radars.
            
            # close_thresh = 40.0  # pixels (out of max 200)
            # too_close_thresh = 10.0 
            # if front_left < close_thresh:
            #     reward -= 0.05 * (close_thresh - front_left) / close_thresh
            #     if front_left < too_close_thresh: 
            #         reward -= 0.15 * (too_close_thresh - front_left) / too_close_thresh
            # if left_dist < too_close_thresh:
            #     reward -= 0.15 * (too_close_thresh - left_dist) / too_close_thresh
            # if front < close_thresh:
            #     reward -= 0.20 * (close_thresh - front) / close_thresh
            #     if front < too_close_thresh: 
            #         reward -= 0.30 * (too_close_thresh - front) / too_close_thresh
            # if front_right < close_thresh:
            #     reward -= 0.05 * (close_thresh - front_right) / close_thresh
            #     if front_right < too_close_thresh: 
            #         reward -= 0.15 * (too_close_thresh - front_right) / too_close_thresh
            # if right_dist < too_close_thresh:
            #     reward -= 0.15 * (too_close_thresh - right_dist) / too_close_thresh

            # Speed modulation:
            # If distance between forward raders and walls are short, penalize going fast.
            # If distance between forward raders and walls are long, mildly encourage speed.
            # min_ahead = min(front_left, front, front_right)
            # danger = max(0.0, close_thresh - min_ahead) / close_thresh  # 0..1
            # speed01 = max(0.0, min(1.0, float(self.car.speed) / 10.0))
            # reward -= 0.15 * danger * speed01

            # clear01 = max(0.0, min(1.0, min_ahead / 200.0))
            # reward += 0.02 * clear01 * speed01

        if self.car.check_flag:
            reward += 15.0
            self.car.check_flag = False
            self.stall_steps = 0
        else:
            # Progress shaping: reward decreasing distance-to-next-checkpoint.
            # Clip the delta to avoid rare geometry jumps dominating the return.
            progress = float(self.car.prev_distance - self.car.cur_distance)
            if progress > 20.0:
                progress = 20.0
            elif progress < -20.0:
                progress = -20.0
            reward += 0.05 * progress

        if not self.car.is_alive: # crash
            # Keep crash strongly negative (to avoid "suiciding" to end episodes)
            # but allow small partial credit for reaching later checkpoints.
            # current_check is 0..6, so this yields [-300, -240].
            reward = -300.0 + 10.0 * float(self.car.current_check)

        elif self.car.goal: # full lap
            reward = 1000.0

        return reward

    def is_done(self):
        if not self.car.is_alive or self.car.goal:
            self.car.current_check = 0
            self.car.distance = 0
            return True
        return False

    def observe(self):
        # return state
        radars = self.car.radars
        # print('RADARS',radars)
        ret = [0, 0, 0, 0, 0]
        i = 0
        for r in radars:
            if self.version in ("v3", "v4"):
                ret[i] = float(r[1]) # raw radar distance for continuous observations
            else:
                ret[i] = int(r[1] / 20) # raw distance (0-200) → integer (0-10), discretization bottleneck
            i += 1

        return ret

    def view_(self, msgs=[]): # RENDERING...
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3
                if event.key == pygame.K_p:
                    self.mode += 1
                    self.mode = self.mode % 3
                elif event.key == pygame.K_q:
                    # Don't hard-exit the whole Python process from inside the env.
                    # Just close the pygame window.
                    pygame.quit()
                    return

        self.screen.blit(self.map, (0, 0))

        if self.mode == 1:
            self.screen.fill((0, 0, 0))
        """
        self.car.radars_for_draw.clear()
        for d in range(-90, 105, 15):
            self.car.check_radar_for_draw(d)
        """
        if len(self.cars) == 1:
            pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        """
        self.car.draw_collision(self.screen)
        """
        # self.car.draw_radar(self.screen) # moved to car.draw()
        
        # self.car.draw(self.screen)
        for car in self.cars:
            if car.get_alive():
                car.draw(self.screen)

        # Display messages...
        for k,msg in enumerate(msgs):
            myfont = pygame.font.SysFont("impact", 20)
            label = myfont.render(msg, 1, (0,0,0))
            self.screen.blit(label,(1055,290+k*25))
            pass

        text = self.font.render("Press 'm' to change view mode", True, (255, 255, 0))
        text_rect = text.get_rect()
        # text_rect.center = (screen_width/2, 100)
        text_rect.topleft = (750,0)
        self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(self.game_speed)


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))
