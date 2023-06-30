from flask import Flask, render_template, request
import chess


app = Flask(__name__)
initial_board = chess.Board()
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King's value doesn't affect the evaluation score
}

@app.route('/')
def chess_game():
    svg = initial_board._repr_svg_()
    return render_template('chess.html', svg=svg)

@app.route('/', methods=['POST'])
def make_move():
    move_uci = request.form.get('move')

    if move_uci:
        move = chess.Move.from_uci(move_uci)
        initial_board.push(move)

        # Computer's move
        _, best_move = alpha_beta_minimax(initial_board, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
        initial_board.push(best_move)

    svg = initial_board._repr_svg_()
    return render_template('chess.html', svg=svg)

def count_attacks(board, color):
    """
    Counts the number of attacks for the given color.
    """
    count = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color == color:
            for attack_square in board.attacks(square):
                attack_piece = board.piece_at(attack_square)
                if attack_piece is not None and attack_piece.color != color:
                    count += piece_values[attack_piece.piece_type]
    return count


def evaluate_board(board):
    """
    Evaluates the current board position and returns a score.
    """
    score = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type] - len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    score += count_attacks(board, chess.WHITE) - count_attacks(board, chess.BLACK)
    return score


def alpha_beta_minimax(board, depth, alpha, beta, maximizing_player):
    """
    Uses the alpha-beta minimax algorithm to determine the best move.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    if maximizing_player:
        max_score = float('-inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score, _ = alpha_beta_minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if score > max_score:
                max_score = score
                best_move = move
            alpha = max(alpha, max_score)
            if beta <= alpha:
                break
        return max_score, best_move

    else:
        min_score = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            score, _ = alpha_beta_minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if score < min_score:
                min_score = score
                best_move = move
            beta = min(beta, min_score)
            if beta <= alpha:
                break
        return min_score, best_move

if __name__ == '__main__':
    app.run(debug=True)
