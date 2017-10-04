"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


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

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if opp_moves == 0:
        score = 10 # easy win if opponent has no moves left
    else:
        score = own_moves - opp_moves

    return float(score)


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

    x,y = game.get_player_location(player)

    a = [ [ 1, 2, 3, 3, 3, 2, 1], \
          [ 2, 3, 5, 5, 5, 3, 2], \
          [ 3, 5, 8, 7, 8, 5, 3], \
          [ 3, 5, 7, 7, 7, 5, 3], \
          [ 3, 5, 8, 7, 8, 5, 3], \
          [ 2, 3, 5, 5, 5, 3, 2], \
          [ 1, 2, 3, 3, 3, 3, 1] ]

    blanks = len(game.get_blank_spaces())

    if blanks < 35:
        score = a[x][y]
    else:
        score = -1 * a[x][y] # favor outer cell in the beginning

    return float(score)


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

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if own_moves > 0 or (own_moves == 0 and opp_moves == 0):
        score = 10 - opp_moves # try to minimize opponent's moves ...
    else:
        score = 0 # ... keeping also in mind that we need to have moves left

    #blanks = len(game.get_blank_spaces())
    #x1, y1 = game.get_player_location(player)
    #x2, y2 = game.get_player_location(game.get_opponent(player))
    #reward = -2 if (abs(x1-x2) <= 1 and abs(y1-y2) <= 1) else 0
    #moves = len(game.get_legal_moves(player))
    #score = moves + reward 
    #score = len(game.get_legal_moves(player))
    #if blanks >= 20:
    #    score *= -1
    #    if score == 0:
    #        score = -9

    return float(score)


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
            pass  # TODO Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration

        return best_move

    def minimax(self, game, depth):
        """Depth-limited minimax search algorithm as described in
        the lectures.

        A modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

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
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf") 
        best_move = (-1,-1)
    
        for m in game.get_legal_moves():
            score = self.min_value(game.forecast_move(m), depth-1)
            if score > best_score:
                best_score = score
                best_move = m
            
        return best_move

    def terminal_test(self, game, depth):
        """ TODO Return True if the game is over for the active player or 
        the search reached maximum depth, and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return (depth == 0) or (len(game.get_legal_moves()) == 0)

    def min_value(self, game, depth):
        """ TODO Return the score for a win if max depth has been reached,
        otherwise return the minimum value over all legal child nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game, depth):
            score = self.score(game, self)
        else:
            score = float("inf") 
            for m in game.get_legal_moves():
                newGame = game.forecast_move(m)
                score = min(score, self.max_value(newGame, depth-1))

        return score

    def max_value(self, game, depth):
        """ TODO Return the score for a loss if max depth has been reached,
        otherwise return the maximum value over all legal child§ nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if self.terminal_test(game, depth):
            score = self.score(game, self)
        else:
            score = float("-inf") 
            for m in game.get_legal_moves():
                newGame = game.forecast_move(m)
                score = max(score, self.min_value(newGame, depth-1))

        return score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        get_move() method implements iterative deepening (ID) search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. Method must return _before_ the
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

        id_limit = 25 # set this to 25 for full iterative deepening experience
      
        #assert self.TIMER_THRESHOLD > 5

        try:
            # Iterative deepening strategy:
            # Increment depth of the search until timer expirese 
            n = 1
            while n < id_limit:
                best_move = self.alphabeta(game, n)
                n += 1

        except SearchTimeout:
            pass  # TODO Handle any actions required after timeout as needed

        #assert self.time_left() > 0

        # if no move selected pick the first from the list of legal moves
        if best_move == (-1,-1):
            moves = game.get_legal_moves()
            best_move = moves[0] if len(moves)>0 else (-1,-1)

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Depth-limited minimax search with alpha-beta pruning 

        This be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

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

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf") 
        best_move = (-1,-1)
    
        for m in game.get_legal_moves():
            score = self.min_value(game.forecast_move(m), depth-1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = m
                # propagate the best score as lower bound to next branch 
                # of minimizing layer 1
                alpha = best_score
            
        return best_move

    def terminal_test(self, game, depth):
        """TODO Return True if the game is over for the active player or 
        the search reached maximum depth, and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return (depth == 0) or (len(game.get_legal_moves()) == 0)

    def min_value(self, game, depth, alpha, beta):
        """TODO
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game, depth):
            score = self.score(game, self)
        else:
            score = float("inf") 
            for m in game.get_legal_moves():
                newGame = game.forecast_move(m)
                score = min(score, self.max_value(newGame, depth-1, alpha, beta))
                if score <= alpha:
                    break
                beta = min(beta, score)

        return score

    def max_value(self, game, depth, alpha, beta):
        """TODO
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if self.terminal_test(game, depth):
            score = self.score(game, self)
        else:
            score = float("-inf") 
            for m in game.get_legal_moves():
                newGame = game.forecast_move(m)
                score = max(score, self.min_value(newGame, depth-1, alpha, beta))
                if score >= beta:
                    break
                alpha = max(alpha, score)

        return score
