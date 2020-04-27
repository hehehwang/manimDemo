from manimlib.imports import *
import numpy as np


class MCKsystem_n:
    def __init__(self, delta_t, MCK, INIT, offset=0.01):
        # system variable
        self.MCK = MCK
        self.M, self.C, self.K = MCK
        self.x_0, self.x_dot_0 = INIT

        # data variable
        self.delta_t = delta_t
        self.offset = offset
        self.data = self.initiate_data_by_offset()

    def get_x_ddot(self, x, x_dot):
        return -self.K / self.M * x - self.C / self.M * x_dot

    def initiate_data_by_offset(self):
        # x_dat, x_dot_dat, x_ddot_dat = [], [], []
        dat = []
        offset = self.offset
        x = self.x_0
        x_dot = self.x_dot_0

        if abs(x) <= offset:
            raise NotImplementedError
        # calculation phase
        while True:
            x_ddot = self.get_x_ddot(x, x_dot)
            x_dot += x_ddot * self.delta_t
            x += x_dot * self.delta_t

            dat.append([x, x_dot, x_ddot])

            if abs(x) < offset and abs(x_dot) < offset: break

        return dat

    def get_data_by_offset(self, offset=None):
        return self.data

    def get_datum(self, t):
        # ITF: Return value in [x, xdot, xddot] format, in same way as get_x function
        # This function tries to behave like contiunuous function
        # Just like func(t)
        # now this data_raw will have PLAYTIME/DELTA_T sets of datum
        # i.e. [[1,2,3], [4,5,6], ... [9,9,9]]
        # each index of data should multiplied with DELTA_T to get proper value
        # or, input time value should be divided with DELTA_T to access with index
        multiplier = round(self.delta_t ** -1)  # round for preventing float error
        return self.data[int(multiplier * t)]

    def get_x(self, t):
        return self.get_datum(t)[0]

    def get_xdot(self, t):
        return self.get_datum(t)[1]

    def get_xddot(self, t):
        return self.get_datum(t)[2]

    def get_x_data(self):
        return tuple(zip(*self.data))[0]

    def get_xdot_data(self):
        return tuple(zip(*self.data))[1]

    def get_xddot_data(self):
        return tuple(zip(*self.data))[2]


class XVsTAxes(Axes):
    CONFIG = {
        "x_min": -10,
        "x_max": 10,
        "y_min": -10,
        "y_max": 10,
        "y_axis_config": {
            "tick_frequency": 0.5,
            "unit_size": 1.5,
        },
        "axis_config": {
            "color": "#EEEEEE",
            "stroke_width": 2,
            "include_tip": False,
        },
        "graph_style": {
            "stroke_color": WHITE,
            "stroke_width": 3,
            "fill_opacity": 0,
        },
    }

    def __init__(self, playspeed, **kwargs):
        super().__init__(**kwargs)
        self.add_labels()
        self.playspeed = playspeed

    def add_axes(self):
        self.axes = Axes(**self.axes_config)
        self.add(self.axes)

    def add_labels(self):
        x_axis = self.get_x_axis()
        y_axis = self.get_y_axis()

        t_label = self.t_label = TexMobject("t")
        t_label.next_to(x_axis.get_right(), UP, MED_SMALL_BUFF)
        x_axis.label = t_label
        x_axis.add(t_label)
        y_label = self.y_label = TexMobject("x", ",\\,", "\\dot{x}", ",\\,", "\\ddot{x}").scale(0.8)
        y_label[2].set_color(BLUE_C)
        y_label[4].set_color(RED_C)
        y_label.next_to(y_axis.get_top(), UP, SMALL_BUFF)
        y_axis.label = y_label
        y_axis.add(y_label)

        self.y_axis_label = y_label
        self.x_axis_label = t_label

        # x_axis.add_numbers()
        x_axis.add(self.get_x_axis_coordinates(x_axis))
        # y_axis.add_numbers()
        y_axis.add(self.get_y_axis_coordinates(y_axis))

    def get_x_axis_coordinates(self, x_axis):
        xmax, xmin = round(self.x_max) + 1, 0
        texs = [str(_) for _ in range(xmin, xmax, self.x_label_frequency)]
        values = np.arange(xmin, xmax, self.x_label_frequency)
        labels = VGroup()
        for tex, value in zip(texs, values):
            symbol = TexMobject(tex)
            symbol.scale(0.5)
            point = x_axis.number_to_point(value)
            symbol.next_to(point, DOWN, MED_SMALL_BUFF)
            labels.add(symbol)
        return labels

    def get_y_axis_coordinates(self, y_axis):
        ymax, ymin = round(self.y_max + 0.9), round(self.y_min - 0.9)
        texs = [str(_) for _ in range(ymin, ymax, self.y_label_frequency)]
        values = np.arange(ymin, ymax, self.y_label_frequency)
        labels = VGroup()
        for tex, value in zip(texs, values):
            symbol = TexMobject(tex)
            symbol.scale(0.5)
            point = y_axis.number_to_point(value)
            symbol.next_to(point, LEFT, MED_SMALL_BUFF)
            labels.add(symbol)
        return labels

    # def get_y_axis_coordinates(self, y_axis):
    #     # texs = [
    #     #     # "\\pi \\over 4",
    #     #     # "\\pi \\over 2",
    #     #     # "3 \\pi \\over 4",
    #     #     # "\\pi",
    #     #     "\\pi / 4",
    #     #     "\\pi / 2",
    #     #     "3 \\pi / 4",
    #     #     "\\pi",
    #     # ]
    #     texs = [str(_) for _ in range(0, 10, 1)]
    #     values = np.arange(0, 10, 1)
    #     labels = VGroup()
    #     for pos_tex, pos_value in zip(texs, values):
    #         neg_tex = "-" + pos_tex
    #         neg_value = -1 * pos_value
    #         for tex, value in (pos_tex, pos_value), (neg_tex, neg_value):
    #             if value > self.y_max or value < self.y_min:
    #                 continue
    #             symbol = TexMobject(tex)
    #             symbol.scale(0.5)
    #             point = y_axis.number_to_point(value)
    #             symbol.next_to(point, LEFT, MED_SMALL_BUFF)
    #             labels.add(symbol)
    #     return labels

    def get_live_drawn_graph_x(self, system,
                               t_max=None,
                               t_step=1.0 / 60,
                               **style):
        style = merge_dicts_recursively(self.graph_style, style)
        if t_max is None:
            t_max = self.x_max

        graph = VMobject()
        style['stroke_color'] = WHITE  # XCOLOR
        graph.set_style(**style)

        graph.all_coords = [(0, system.get_x(0))]
        graph.time = 0
        graph.time_of_last_addition = 0

        def update_graph(graph, dt):
            dt = dt * self.playspeed  # play speed correction
            graph.time += dt
            if graph.time > t_max:
                graph.remove_updater(update_graph)
                return
            new_coords = (graph.time, system.get_x(graph.time))
            if graph.time - graph.time_of_last_addition >= t_step:
                graph.all_coords.append(new_coords)
                graph.time_of_last_addition = graph.time
            points = [
                self.coords_to_point(*coords)
                for coords in [*graph.all_coords, new_coords]
            ]
            graph.set_points_smoothly(points)

        graph.add_updater(update_graph)
        return graph

    def get_live_drawn_graph_xdot(self, system,
                                  t_max=None,
                                  t_step=1.0 / 60,
                                  **style):
        style = merge_dicts_recursively(self.graph_style, style)
        if t_max is None:
            t_max = self.x_max

        graph = VMobject()
        style['stroke_color'] = BLUE_E  # VCOLOR
        graph.set_style(**style)

        graph.all_coords = [(0, system.get_xdot(0))]
        graph.time = 0
        graph.time_of_last_addition = 0

        def update_graph(graph, dt):
            dt = dt * self.playspeed  # play speed correction
            graph.time += dt
            if graph.time > t_max:
                graph.remove_updater(update_graph)
                return
            new_coords = (graph.time, system.get_xdot(graph.time))  # ##
            if graph.time - graph.time_of_last_addition >= t_step:
                graph.all_coords.append(new_coords)
                graph.time_of_last_addition = graph.time
            points = [
                self.coords_to_point(*coords)
                for coords in [*graph.all_coords, new_coords]
            ]
            graph.set_points_smoothly(points)

        graph.add_updater(update_graph)
        return graph

    def get_live_drawn_graph_xddot(self, system,
                                   t_max=None,
                                   t_step=1.0 / 60,
                                   **style):
        style = merge_dicts_recursively(self.graph_style, style)
        if t_max is None:
            t_max = self.x_max

        graph = VMobject()
        style['stroke_color'] = RED_E  # ACOLOR
        graph.set_style(**style)

        graph.all_coords = [(0, system.get_xddot(0))]
        graph.time = 0
        graph.time_of_last_addition = 0

        def update_graph(graph, dt):
            dt = dt * self.playspeed  # play speed correction
            graph.time += dt
            if graph.time > t_max:
                graph.remove_updater(update_graph)
                return
            new_coords = (graph.time, system.get_xddot(graph.time))  # ##
            if graph.time - graph.time_of_last_addition >= t_step:
                graph.all_coords.append(new_coords)
                graph.time_of_last_addition = graph.time
            points = [
                self.coords_to_point(*coords)
                for coords in [*graph.all_coords, new_coords]
            ]
            graph.set_points_smoothly(points)

        graph.add_updater(update_graph)
        return graph


class MCK_Simulation_v2(Scene):
    CONFIG = {
        # data
        "MCK": [10, 6, 10],
        "INIT": [5, -1],
        "DELTA_T": 0.0001,
        "DATA_RAW": [],  # for cpu saving
        # animation
        "REST_POS_OFFSET": 0.1,
        "PLAY_SPEED": 2,
        "PLAY_TIME": 0,
        # regulation
        "X_REG_SCALE": 4,
        "X_REG_OFFSET": 0,
        "VA_REG_SCALE": 3,
        "VA_REG_OFFSET": 0,
        # colors
        "COLOR_INDICATOR": LIGHT_GRAY,
        "COLOR_MASS": GREEN_E,
        "COLOR_SPRING": TEAL_E,
        "COLOR_DAMPER": YELLOW_E,
        # graph
        "axes_config": {
            "x_min": 0,
            "x_max": 10,
            "x_axis_config": {
                "tick_frequency": 1,
                "unit_size": 0.8,
            },
            "y_min": -5,
            "y_max": 5,
            "y_axis_config": {
                "tick_frequency": 1,
                "unit_size": 0.15,
            },
            "axis_config": {
                "color": "#EEEEEE",
                "stroke_width": 2,
                "include_tip": False,
            },
            "graph_style": {
                "stroke_color": WHITE,
                "stroke_width": 5,
                "fill_opacity": 0,
            },
            "x_label_frequency": 2,
            "y_label_frequency": 2,
        },
        "axes_corner": UL,
        "axes_edge": UP,
    }

    def construct(self):
        # value = ValueTracker(self.MCKfunc(0))
        mck_system = MCKsystem_n(self.DELTA_T, self.MCK, self.INIT, self.REST_POS_OFFSET)

        # initiate config value
        self.PLAY_TIME = int(len(mck_system.get_data_by_offset()) * self.DELTA_T / self.PLAY_SPEED)
        self.axes_config['x_max'] = self.PLAY_TIME * self.PLAY_SPEED + LARGE_BUFF
        self.axes_config['y_max'] = max(mck_system.get_x_data() +
                                        mck_system.get_xdot_data() +
                                        mck_system.get_xddot_data()) + LARGE_BUFF
        self.axes_config['y_min'] = min(mck_system.get_x_data() +
                                        mck_system.get_xdot_data() +
                                        mck_system.get_xddot_data()) - LARGE_BUFF

        # graph
        axes = XVsTAxes(self.PLAY_SPEED, **self.axes_config)
        axes.center()
        axes.to_edge(self.axes_edge, buff=1.5)
        graph_x = axes.get_live_drawn_graph_x(mck_system)
        graph_xdot = axes.get_live_drawn_graph_xdot(mck_system)
        graph_xddot = axes.get_live_drawn_graph_xddot(mck_system)

        # Generate Objects
        indicator = Line(1.5 * UP, 1.5 * DOWN,
                         stroke_color=self.COLOR_INDICATOR)  # INDICATORCOLOR
        mass = Square(color=GRAY,
                      sheen_factor=0.7,
                      stroke_color=GREEN_E,
                      stroke_width=20)  # MASSCOLOR
        wall = Rectangle(height=3,
                         width=1,
                         fill_color=DARKER_GRAY,
                         fill_opacity=1,
                         stroke_opacity=0.5,
                         stroke_color=DARK_GRAY,
                         sheen_factor=0.3, )
        spring = DashedLine(positive_space_ratio=0.7,
                            dash_length=0.1,
                            stroke_color=TEAL_E)  # SPRINGCOLOR
        damper = DashedLine(positive_space_ratio=0.7,
                            dash_length=0.1,
                            stroke_color=YELLOW_E)  # DAMPERCOLOR
        labels = {'mass': TexMobject('\\text{M}', '\\text{ass} \\, (', f'm={self.MCK[0]}', ')'),
                  'damper': TexMobject('\\text{D}', '\\text{apmer}\\,(', f'c={self.MCK[1]}', ')'),
                  'spring': TexMobject('\\text{S}', '\\text{pring}\\,(', f'k={self.MCK[2]}', ')'),
                  # 'wall': TexMobject('\\text{Wall}'),
                  'zero': TexMobject('x=0')}
        title = TextMobject('Mass', '-', 'Damper', '-', 'Spring', ' System',
                            color=WHITE,
                            sheen_factor=0.5,
                            height=0.9)

        vector_velocity = Arrow(stroke_color=BLUE_E)  # VCOLOR
        vector_acceleration = Arrow(stroke_color=RED_E)  # ACOLOR

        # some constants and groups
        obj_group = VGroup(indicator, mass, wall, spring, damper, vector_velocity, vector_acceleration)
        obj_position = np.array([0, -2, 0])
        line_gap = np.array([0, 0.5, 0])

        # set initial position
        mass.move_to(obj_position)
        wall.move_to(np.array([-5, 0, 0]) + obj_position)
        indicator.move_to(obj_position)

        vector_velocity.move_to(obj_position + 0.5 * UP)
        vector_velocity.scale(0.5)
        vector_acceleration.move_to(obj_position + 0.5 * DOWN)
        vector_acceleration.scale(0.5)

        labels['zero'].next_to(indicator, UP, buff=SMALL_BUFF)
        labels['zero'].scale(0.9)
        # labels['wall'].move_to(wall.get_center())
        # labels['wall'].scale(0.7)
        labels['mass'].to_edge(RIGHT)
        labels['mass'].scale(0.8)
        labels['spring'].move_to(np.array([-3, +0.75, 0]) + obj_position)
        labels['spring'].scale(0.7)
        labels['damper'].move_to(np.array([-3, -0.75, 0]) + obj_position)
        labels['damper'].scale(0.7)

        # set colors
        title[0].set_color(self.COLOR_MASS)
        title[2].set_color(self.COLOR_DAMPER)
        title[4].set_color(self.COLOR_SPRING)
        labels['zero'].set_color(self.COLOR_INDICATOR)  # INDICATORCOLOR
        labels['mass'][0].set_color(self.COLOR_MASS)
        labels['mass'][2].set_color(self.COLOR_MASS)  # MASSCOLOR
        labels['spring'][0].set_color(self.COLOR_SPRING)
        labels['spring'][2].set_color(self.COLOR_SPRING)  # SPRINGCOLOR
        labels['damper'][0].set_color(self.COLOR_DAMPER)  # DAMPERCOLOR
        labels['damper'][2].set_color(self.COLOR_DAMPER)

        spring.put_start_and_end_on(wall.get_right() + line_gap, mass.get_left() + line_gap)
        damper.put_start_and_end_on(wall.get_right() - line_gap, mass.get_left() - line_gap)

        # update things
        def frame_idx():
            i = 0
            while 1:
                yield i
                i += 1
        idx = frame_idx()

        def get_point_from_value(value):
            return np.array([value, 0, 0])

        def update_mass(obj, dt):
            ind, m, w, s, d, vv, va = obj
            # indicator, mass, wall, spring, damper, labels, vector_velocity, vector_acceleration
            dt_mult = dt * self.PLAY_SPEED
            time_now = dt_mult * next(idx)

            x, v, a = mck_system.get_datum(time_now)
            x_point, v_point, a_point = [get_point_from_value(_) for _ in [x, v, a]]
            m.move_to(x_point + obj_position)
            # labels['mass'].move_to(move_point + obj_position)

            s.put_start_and_end_on(w.get_right() + line_gap, m.get_left() + line_gap)
            d.put_start_and_end_on(w.get_right() - line_gap, m.get_left() - line_gap)

            vv.put_start_and_end_on(x_point + obj_position + line_gap, x_point + v_point + obj_position + line_gap)
            va.put_start_and_end_on(x_point + obj_position - line_gap, x_point + a_point + obj_position - line_gap)


        # upcomming objects
        self.play(Write(title))
        self.play(title.to_edge, dict(edge=UP, buff=MED_SMALL_BUFF,),
                  title.set_height, 0.4)
        self.play(Write(axes))
        self.play(DrawBorderThenFill(obj_group))
        # magic starts
        self.play(mass.shift, get_point_from_value(mck_system.get_x(0)),
                  vector_velocity.put_start_and_end_on, *(get_point_from_value(mck_system.get_x(0)) + obj_position + line_gap,
                                                          get_point_from_value(mck_system.get_x(0) + mck_system.get_xdot(0)) + obj_position + line_gap),
                  vector_acceleration.put_start_and_end_on, *(get_point_from_value(mck_system.get_x(0)) + obj_position - line_gap,
                                                          get_point_from_value(mck_system.get_x(0) + mck_system.get_xddot(0)) + obj_position - line_gap),
                  spring.put_start_and_end_on, *(wall.get_right() + line_gap,
                                                 get_point_from_value(mck_system.get_x(0)) + obj_position + line_gap),
                  damper.put_start_and_end_on, *(wall.get_right() - line_gap,
                                                 get_point_from_value(mck_system.get_x(0)) + obj_position - line_gap), )

        # some captions
        caption_arrow = Arrow(labels['mass'].get_edge_center(DOWN), mass.get_edge_center(UP),buff=SMALL_BUFF ,
                              stroke_color=self.COLOR_MASS)
        caption_vv = TextMobject('Velocity Vector', color=BLUE_C, height=0.3)
        # caption_vv.scale(0.6)
        caption_vv.next_to(vector_velocity, DOWN, buff=SMALL_BUFF)
        caption_va = TextMobject('Acceleration Vector', color=RED_C, height=0.3)
        # caption_va.scale(0.6)
        caption_va.next_to(vector_acceleration, DOWN, buff=SMALL_BUFF)
        captions = (caption_arrow, caption_vv, caption_va)

        self.play(*[ShowCreation(_) for _ in labels.values()],
                  *[ShowCreation(_) for _ in captions], run_time=2)
        self.wait(2)
        self.play(*[FadeOut(_) for _ in labels.values()],
                  *[FadeOut(_) for _ in captions])
        obj_group.add_updater(update_mass)
        # [self.bring_to_front(_) for _ in labels.values()]
        self.add(axes, graph_x, graph_xdot, graph_xddot)
        self.wait(self.PLAY_TIME*0.95)


def turn_value_to_vector(value):
    return np.array([value, 0, 0])


def get_vector_from_value(value):
    return np.array([value, 0, 0])


class MCK_Simulation(Scene):
    CONFIG = {
        "n_steps_per_frame": 100,
        # mck, init, tt, deltt = [10, 10, 6], [5, 0], 12, 0.0001
        "MCK": [10, 2, 10],
        "INIT": [10, 10],
        "TIME": 12,
        "DELTA_T": 0.0001,
        "PLAY_SPEED": 2,
        # regulation
        "X_REG_SCALE": 4,
        "X_REG_OFFSET": 0,
        "VA_REG_SCALE": 3,
        "VA_REG_OFFSET": 0,
    }

    def _regulate_x_by_scale(self, x_list):
        x_max, x_min = max(x_list), min(x_list)
        return [x * self.X_REG_SCALE / (x_max - x_min) + self.X_REG_OFFSET for x in x_list]

    def _regulate_va_by_scale(self, va_list):
        va_max, va_min = max(va_list), min(va_list)
        return [va * self.VA_REG_SCALE / (va_max - va_min) + self.VA_REG_OFFSET for va in va_list]

    def regulate_data_by_scale(self, data):
        data = list(zip(*data))
        data_return = [self._regulate_x_by_scale(data[0]),
                       self._regulate_va_by_scale(data[1]),
                       self._regulate_va_by_scale(data[2])]
        data_return = list(zip(*data_return))
        return data_return

    def get_values(self):
        system = MCKsystem_n(self.DELTA_T, self.MCK, self.INIT)
        return system.get_data_by_offset(0.1)

    def skip_data_by_frame(self, data):
        framerate = self.camera.frame_rate
        data_raw = data
        frame_ratio = int((1 / self.DELTA_T) / framerate * self.PLAY_SPEED)
        data_return = []
        i = 0
        while 1:
            try:
                data_return.append(data_raw[i * frame_ratio])
                i += 1
            except:
                break
        return data_return

    def process_data(self):
        data_rare = self.get_values()
        data_medium = self.skip_data_by_frame(data_rare)
        data_medium_welldone = self.regulate_data_by_scale(data_medium)
        data_welldone = [[get_vector_from_value(v) for v in datum] for datum in data_medium_welldone]
        return data_welldone

    def setup(self):
        indicator = Line(2 * UP, 2 * DOWN)
        indicator.move_to(np.array([self.X_REG_OFFSET, 0, 0]))
        indicator.set_stroke(color=GRAY)
        label0 = TexMobject('x=0')
        label0.move_to(indicator.get_edge_center(UP) + 0.5 * UP)
        self.add(indicator)
        self.play(Write(label0))

    def construct(self):
        data = self.process_data()
        data_iter = iter(data)
        play_duration = len(data) // self.camera.frame_rate

        # mobs
        mass = Rectangle(height=2, width=2, color=GRAY, fill_opacity=1)

        wall = Rectangle(height=4,
                         width=1,
                         fill_color=GRAY,
                         fill_opacity=1,
                         stroke_opacity=1,
                         stroke_color=DARK_GRAY,
                         background_stroke_opacity=0.5,
                         sheen_factor=0.5, )
        wall.to_edge(LEFT, buff=2)

        spring = Line(wall.get_right() + 0.5 * UP, mass.get_left() + 0.5 * UP)
        damper = DashedLine(wall.get_right() + 0.5 * DOWN, mass.get_left() + 0.5 * DOWN)

        arrow_velocity = Arrow().shift(0.5 * UP)
        arrow_acceleration = Arrow().shift(0.5 * DOWN)
        arrow_group = VGroup(arrow_velocity, arrow_acceleration)

        def get_values():
            return next(data_iter)

        def update_mass(mob, dt):
            datum = get_values()
            displacement, velocity, acceleration = datum
            mass.move_to(displacement)
            spring.put_start_and_end_on(wall.get_right() + 0.5 * UP, mass.get_left() + 0.5 * UP)
            damper.put_start_and_end_on(wall.get_right() + 0.5 * DOWN, mass.get_left() + 0.5 * DOWN)

            point_vel_origin = displacement + 0.5 * UP
            arrow_velocity.put_start_and_end_on(point_vel_origin, point_vel_origin + velocity)
            point_acc_origin = displacement + 0.5 * DOWN
            arrow_acceleration.put_start_and_end_on(point_acc_origin, point_acc_origin + acceleration)

        def update_initiate_position(mob, dt):
            datum = data[0]
            x, xdot, xddot = datum
            mass.move_to(x)
            arrow_velocity.put_start_and_end_on(x + 0.5 * UP, x + 0.5 * UP + xdot)
            arrow_acceleration.put_start_and_end_on(x + 0.5 * DOWN, x + 0.5 * DOWN + xddot)

        self.play(DrawBorderThenFill(wall),
                  DrawBorderThenFill(mass))
        self.play(*map(Write, [spring, damper, arrow_group]))
        self.wait()
        mass.add_updater(update_mass)
        self.wait(play_duration)
