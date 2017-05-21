"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def best_advantage_in_next_move(game, player, possible_moves):
    if possible_moves is None or len(possible_moves) == 0:
        return float("-inf")

    paths = []
    opponent = game.get_opponent(player)
    for move in possible_moves:
        game_copy = game.forecast_move(move)
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("inf")
        play_legal_moves = len(game_copy.get_legal_moves(player))
        opponent_legal_moves = len(game_copy.get_legal_moves(opponent))
        paths.append(play_legal_moves - 0.5 * opponent_legal_moves)
    return max(paths)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_possible_moves = len(game.get_legal_moves(player))
    opponent_possible_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(my_possible_moves - 0.5 * opponent_possible_moves)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    center_distance = float((h - y) ** 2 + (w - x) ** 2)

    my_possible_moves = len(game.get_legal_moves(player))
    opponent_possible_moves = len(game.get_legal_moves(game.get_opponent(player)))
    occupied_rate = 1 - float(len(game.get_blank_spaces())) / (game.width * game.height)

    """ 
    Heuristics 2:
    
    At the beginning of the game (less than 10% occupied rate),
    1. maximise my possible moves,
    2. minimise my opponent moves and
    3. on top of #1 and #2, try to get closer to center as we can
    
    Afterwards (occupied rate >= 10% and occupied rate <= 30%), 
    1. maximise my possible moves and
    2. minimise my opponent moves with same weight
    
    Towards the end of a game (30%+ occupied rate):
    1. maximise my possible moves and
    2. minimise my opponent moves but with less weight
    
    """
    if occupied_rate < 0.1:
        return float(my_possible_moves - 0.2 * center_distance - opponent_possible_moves)
    if occupied_rate < 0.3:
        return float(my_possible_moves - opponent_possible_moves)
    return float(my_possible_moves - 0.5 * opponent_possible_moves)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_possible_moves = game.get_legal_moves(player)
    len_my_possible_moves = len(my_possible_moves)

    opponent = game.get_opponent(player)
    opponent_possible_moves = game.get_legal_moves(opponent)
    len_opponent_possible_moves = len(opponent_possible_moves)

    occupied_rate = 1 - float(len(game.get_blank_spaces())) / (game.width * game.height)

    if occupied_rate < 0.4:
        return float(len_my_possible_moves - 0.5 * len_opponent_possible_moves)
    return float(best_advantage_in_next_move(game, player, my_possible_moves))

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        max_value_move = self.__get_max_value_move(game, depth)
        return max_value_move

    def __get_max_value_move(self, game, max_depth):
        legal_moves = game.get_legal_moves(self)
        if legal_moves is None or len(legal_moves) == 0:
            return (-1, -1)

        move_value_dict = {}
        current_depth = 0
        for move in legal_moves:
            move_value_dict[move] = self.__min_value_for_move(game, move, current_depth, max_depth)

        return max(move_value_dict, key=move_value_dict.get)

    def __min_value_for_move(self, game, move, current_depth, max_depth):
        return self.__get_value_for_move(game, move, current_depth, max_depth, self.__max_value_for_move, min)

    def __max_value_for_move(self, game, move, current_depth, max_depth):
        return self.__get_value_for_move(game, move, current_depth, max_depth, self.__min_value_for_move, max)

    def __get_value_for_move(self, game, move, current_depth, max_depth, get_next_move_value_fn, policy_fn):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        game_copy = game.forecast_move(move)
        current_depth += 1
        next_legal_moves = game_copy.get_legal_moves(game_copy.active_player)
        if next_legal_moves is None or len(next_legal_moves) == 0 or current_depth >= max_depth:
            return self.score(game_copy, self)

        values_for_moves_array  = []
        for next_move in next_legal_moves:
            values_for_moves_array.append(get_next_move_value_fn(game_copy, next_move, current_depth, max_depth))

        return policy_fn(values_for_moves_array)

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = 0
            while 1:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        max_value_move = self.__get_max_value_move(game, depth, alpha, beta)
        return max_value_move

    def __get_max_value_move(self, game, max_depth, alpha, beta):
        legal_moves = game.get_legal_moves(self)
        if legal_moves is None or len(legal_moves) == 0:
            return (-1, -1)

        move_value_dict = {}
        current_depth = 0
        max_value = float("-inf")
        for move in legal_moves:
            value = self.__min_value_for_move(game, move, current_depth, max_depth, alpha, beta)
            max_value = max(max_value, value)
            move_value_dict[move] = value
            if max_value >= beta:
                return move
            alpha = max(alpha, max_value)

        return max(move_value_dict, key=move_value_dict.get)

    def __min_value_for_move(self, game, move, current_depth, max_depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        game_copy = game.forecast_move(move)
        current_depth += 1
        next_legal_moves = game_copy.get_legal_moves(game_copy.active_player)
        if next_legal_moves is None or len(next_legal_moves) == 0 or current_depth >= max_depth:
            return self.score(game_copy, self)

        min_value = float("inf")
        for next_move in next_legal_moves:
            min_value = min(min_value, self.__max_value_for_move(game_copy, next_move, current_depth, max_depth, alpha, beta))
            if min_value <= alpha:
                return min_value
            beta = min(min_value, beta)
            if alpha >= beta:
                return beta

        return min_value

    def __max_value_for_move(self, game, move, current_depth, max_depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        game_copy = game.forecast_move(move)
        current_depth += 1
        next_legal_moves = game_copy.get_legal_moves(game_copy.active_player)
        if next_legal_moves is None or len(next_legal_moves) == 0 or current_depth >= max_depth:
            return self.score(game_copy, self)

        max_value = float("-inf")
        for next_move in next_legal_moves:
            max_value = max(max_value, self.__min_value_for_move(game_copy, next_move, current_depth, max_depth, alpha, beta))
            if max_value >= beta:
                return max_value
            alpha = max(alpha, max_value)

        return max_value
