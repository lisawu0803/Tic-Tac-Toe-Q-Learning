#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pygame
import numpy as np
import random
import pickle

# Game settings
WIDTH, HEIGHT = 300, 300
LINE_WIDTH = 10
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = WIDTH // BOARD_COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Q-learning settings
epsilon = 0.5  # Exploration rate
alpha = 0.5    # Learning rate
gamma = 0.9    # Discount factor

# Game state
EMPTY = 0
X = 1
O = -1

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.current_player = X

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.current_player = X

    def is_winner(self, player):
        for row in range(BOARD_ROWS):
            if np.all(self.board[row, :] == player):
                return True
        for col in range(BOARD_COLS):
            if np.all(self.board[:, col] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_full(self):
        return np.all(self.board != EMPTY)

    def available_actions(self):
        return [(r, c) for r in range(BOARD_ROWS) for c in range(BOARD_COLS) if self.board[r, c] == EMPTY]

    def make_move(self, row, col):
        if self.board[row, col] == EMPTY:
            self.board[row, col] = self.current_player
            self.current_player = O if self.current_player == X else X
            return True
        return False

class QLearningAgent:
    def __init__(self):
        self.q_table = {}

    def save_q_table(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)  # Save Q-table to file

    def load_q_table(self, filename='q_table.pkl'):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)  # Load Q-table from file
                print("Q-table loaded successfully.")
        except FileNotFoundError:
            print("Q-table file not found, starting fresh.")
            self.q_table = {}  # Initialize Q-table if file not found

    def get_state_key(self, board):
        return str(board.reshape(9))

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < epsilon:
            return random.choice(available_actions)
        else:
            q_values = [self.q_table.get((state, (r, c)), 0) for r, c in available_actions]
            max_q = max(q_values)
            return available_actions[q_values.index(max_q)]

    def learn(self, state, action, reward, next_state, available_actions):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        current_q = self.q_table.get((state_key, action), 0)
        
        # Get maximum Q-value of the next state
        future_q = max([self.q_table.get((next_state_key, a), 0) for a in available_actions], default=0)
        
        # Update the Q-value for the state-action pair
        self.q_table[(state_key, action)] = current_q + alpha * (reward + gamma * future_q - current_q)

def draw_board(board):
    # Draw grid lines
    for r in range(1, BOARD_ROWS):
        pygame.draw.line(screen, BLACK, (0, r * SQUARE_SIZE), (WIDTH, r * SQUARE_SIZE), LINE_WIDTH)
    for c in range(1, BOARD_COLS):
        pygame.draw.line(screen, BLACK, (c * SQUARE_SIZE, 0), (c * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

    # Draw X and O
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r, c] == X:
                pygame.draw.line(screen, GREEN, (c * SQUARE_SIZE + 10, r * SQUARE_SIZE + 10), (c * SQUARE_SIZE + SQUARE_SIZE - 10, r * SQUARE_SIZE + SQUARE_SIZE - 10), LINE_WIDTH)
                pygame.draw.line(screen, GREEN, (c * SQUARE_SIZE + SQUARE_SIZE - 10, r * SQUARE_SIZE + 10), (c * SQUARE_SIZE + 10, r * SQUARE_SIZE + SQUARE_SIZE - 10), LINE_WIDTH)
            elif board[r, c] == O:
                pygame.draw.circle(screen, RED, (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 3)

def draw_text(text, y_offset, size, color, background_color=None):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color, background_color)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, y_offset))
    screen.blit(text_surface, text_rect)

def draw_buttons():
    pygame.draw.rect(screen, BLUE, (100, 200, 100, 30))   # Reset button
    pygame.draw.rect(screen, BLUE, (100, 230, 100, 30))  # Quit button
    draw_text("Reset", 215, 30, WHITE)  # Set text position for Reset, y-coordinate just below the button
    draw_text("Quit", 245, 30, WHITE)     # Set text position for Quit, y-coordinate just below the button

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe with Q-Learning')

def main():
    game = TicTacToe()
    agent = QLearningAgent()
    agent.load_q_table()
    clock = pygame.time.Clock()
    running = True
    game_over = False
    result_message = ""

    while running:
        screen.fill(WHITE)
        draw_board(game.board)

        if game_over:
            draw_text(result_message, 100, 30, WHITE, BLUE)
            draw_buttons()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouseX, mouseY = event.pos

                    if 100 <= mouseX <= 200 and 200 <= mouseY <= 230:
                        # Reset button
                        game.reset()
                        game_over = False
                    elif 100 <= mouseX <= 200 and 230 <= mouseY <= 260:
                        # Quit button
                        running = False
            pygame.display.flip()
            clock.tick(30)  # Control the frame rate of the game
            continue  # Skip the rest of the code and return to the loop

        # Check if the game is over
        if game.is_winner(X):
            result_message = "X wins!"
            game_over = True
            reward = -1  # Penalize AI if X wins
        elif game.is_winner(O):
            result_message = "O wins!"
            game_over = True
            reward = 1  # Reward AI if O wins
        elif game.is_full():
            result_message = "It's a draw!"
            game_over = True
            reward = 0  # Zero reward for a draw
        else:
            reward = -0.1  # Small penalty if the game is ongoing

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouseX, mouseY = event.pos
                    row = mouseY // SQUARE_SIZE
                    col = mouseX // SQUARE_SIZE

                    # Player's move
                    if game.make_move(row, col):
                        state = game.board.copy()
                        available_actions = game.available_actions()

                        # Let AI make a move after checking if the game is over
                        if not game.is_winner(X) and not game.is_winner(O) and available_actions:
                            action = agent.choose_action(agent.get_state_key(state), available_actions)
                            game.make_move(action[0], action[1])

                            # Update Q-table at the end of each round
                            next_state = game.board.copy()
                            agent.learn(state, action, reward, next_state, available_actions)

        pygame.display.flip()
        clock.tick(30)  # Control the frame rate of the game
        
    # Save Q-table after the game ends
    agent.save_q_table()

    pygame.quit()
    
if __name__ == "__main__":
    main()

