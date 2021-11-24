import os
import gym
import numpy as np
import pandas as pd
from gym.spaces import Box, Discrete

class DiscreteVaccumRobotV0(gym.Env):
    
    def __init__(self, env_config):
        np.set_printoptions(linewidth=200)
        self.room_file = env_config['room_file']
        self.max_moves = env_config['max_moves']
        self.window_size = env_config['window_size']
        self.action_space = Discrete(n=3)
        self.observation_space = Box(low=0, high=1, shape=(self.window_size ** 2 + 1,))

    def reset(self):
        self.room_data = pd.read_csv(self.room_file, header=None).astype(float)
        self.room_data = self.room_data.values
        self.room_data_shape = self.room_data.shape
        not_visited_slots = np.where(self.room_data == 1)
        initial_pos_index = np.random.randint(0, len(not_visited_slots[0]))
        self.position = [not_visited_slots[0][initial_pos_index],
                         not_visited_slots[1][initial_pos_index]]
        self.rotation = np.random.choice([0, 90, 180, 270])
        self.room_data[self.position[0]][self.position[1]] = -0.1
        self.current_move = 0
        self.room_memory = np.zeros(self.room_data_shape)
        self.room_memory[self.position[0]][self.position[1]] = 1
        surroundings = self.crop_room_matrix().flatten() / 2
        return list(surroundings) + [self.rotation / 270]

    def render(self):
        rendered_room_data = self.room_data.copy()
        rendered_room_data[self.position[0]][self.position[1]] = 3
        os.system('cls' if os.name == 'nt' else 'clear')
        str_to_be_printed = ''
        for row in rendered_room_data:
            for value in row:
                if value == 1:
                    str_to_be_printed += 'o'
                elif value == 2:
                    str_to_be_printed += '#'
                elif value == 3:
                    str_to_be_printed += 'O'
                else:
                    str_to_be_printed += '.'
            str_to_be_printed += '\n'
        print(str_to_be_printed)

    def render_room_memory(self):
        rendered_room_memory = self.room_memory.copy()
        rendered_room_memory[self.position[0]][self.position[1]] = 3
        os.system('cls' if os.name == 'nt' else 'clear')
        str_to_be_printed = ''
        for row in rendered_room_memory:
            for value in row:
                if value == 0:
                    str_to_be_printed += '?'
                elif value == 1:
                    str_to_be_printed += '.'
                elif value == 2:
                    str_to_be_printed += '#'
                else:
                    str_to_be_printed += 'O'
            str_to_be_printed += '\n'
        print(str_to_be_printed)

    def crop_room_matrix(self):
        surroundings = np.full((self.window_size, self.window_size), 2)
        row_min = self.position[0] - int(self.window_size / 2)
        row_max = self.position[0] + int(self.window_size / 2) + 1
        column_min = self.position[1] - int(self.window_size / 2)
        column_max = self.position[1] + int(self.window_size / 2) + 1
        selection = self.room_memory[row_min if row_min >= 0 else 0: row_max if row_max < self.room_memory.shape[0] else self.room_memory.shape[0],
                                     column_min if column_min >= 0 else 0: column_max if column_max < self.room_memory.shape[1] else self.room_memory.shape[1]]
        row_offset = 0
        column_offset = 0
        if row_min < 0:
            row_offset = -row_min
        if column_min < 0:
            column_offset = -column_min

        surroundings[row_offset: selection.shape[0] + row_offset, column_offset: selection.shape[1] + column_offset] = selection
        if self.rotation == 90:
            surroundings = np.rot90(surroundings)
        elif self.rotation == 180:
            surroundings = np.rot90(surroundings)
            surroundings = np.rot90(surroundings)
        elif self.rotation == 270:
            surroundings = np.rot90(surroundings)
            surroundings = np.rot90(surroundings)
            surroundings = np.rot90(surroundings)
        return surroundings.astype(int)
        
    def step(self, action):
        # Move forward
        if action == 0:
            if self.rotation == 0:
                next_position = [self.position[0] - 1, self.position[1]]
            elif self.rotation == 90:
                next_position = [self.position[0], self.position[1] + 1]
            elif self.rotation == 180:
                next_position = [self.position[0] + 1, self.position[1]]
            elif self.rotation == 270:
                next_position = [self.position[0], self.position[1] - 1]
            if self.room_data[next_position[0]][next_position[1]] != 2:
                reward = self.room_data[next_position[0]][next_position[1]]
                self.position = next_position
                self.room_data[next_position[0]][next_position[1]] = -0.1
                self.room_memory[self.position[0]][self.position[1]] = 1
            else:
                reward = -0.1
                self.room_memory[next_position[0]][next_position[1]] = 2
        else:
            # Rotate
            reward = -0.1
            self.rotation += 90 if action == 1 else -90
            if self.rotation == -90:
                self.rotation = 270
            elif self.rotation == 360:
                self.rotation = 0
        self.current_move += 1
        surroundings = self.crop_room_matrix().flatten() / 2
        return [list(surroundings) + [self.rotation / 270],
                reward,
                (self.room_data == 1).any() == False or self.current_move == self.max_moves,
                {}]
    

if __name__ == '__main__':
    env_config = {'room_file': 'room1.csv', 'max_moves': 100, 'window_size': 5}
    robot_env = DiscreteVaccumRobotV0(env_config)
    state = robot_env.reset()
    print(state)
    print(robot_env.room_data)