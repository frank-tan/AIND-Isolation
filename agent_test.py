"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload

from sample_players import RandomPlayer, GreedyPlayer


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.AlphaBetaPlayer()
        self.player2 = GreedyPlayer()
        self.game = isolation.Board(self.player1, self.player2)

    def test_play_game(self):
        winner, history, outcome = self.game.play(150)
        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        print(self.game.to_string())
        print(1-float(len(self.game.get_blank_spaces()))/self.game.width/self.game.height)
        print("Move history:\n{!s}".format(history))

if __name__ == '__main__':
    unittest.main()
