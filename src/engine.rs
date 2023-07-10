use crate::{Board, Move, Result};

pub trait Engine {
    /// creates a new engine in the given position
    fn new_from_board(board: Board) -> Self;

    /// advance the position by `mve`
    fn accept_move(&mut self, mve: Move);

    /// start searching for the best move. This done on a spearate thead.
    fn start_search(&mut self);

    /// stop searching for the best move.
    fn end_search(&mut self) -> Result<()>;

    /// returns the best move the engine found so far or `None`.
    /// This must be set to `Some` after [start_search] and [end_search]
    /// have been called.
    fn best_move(&self) -> Option<Move>;

    /// returns the calculated score for the current position
    fn current_score(&self) -> f32;
}

pub mod ella;
