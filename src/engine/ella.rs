use anyhow::{bail, Context};

use super::{Engine, Result};
use crate::{utils::AtomicF32, Board, Color, Move, PieceType};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};

pub struct EllaChess {
    board: Board,
    control: Arc<SearchControl>,
    search_thead: Option<JoinHandle<Result<()>>>,
}

struct SearchControl {
    search: Mutex<Option<Search>>,
    searching: AtomicBool,
    abort_search: AtomicBool,
    score: AtomicF32,
    best_move: Mutex<Option<Move>>,
}

impl Engine for EllaChess {
    fn new_from_board(board: Board) -> Self {
        let control = Arc::new(SearchControl {
            searching: AtomicBool::new(false),
            abort_search: AtomicBool::new(false),
            score: AtomicF32::new(0.0),
            best_move: Mutex::new(None),
            search: Mutex::new(None),
        });

        let search = Search::new(control.clone());

        *control.search.lock().unwrap() = Some(search);

        EllaChess {
            board,
            control,
            search_thead: None,
        }
    }

    fn accept_move(&mut self, mve: Move) {
        let restart_search = self.control.searching.load(Ordering::SeqCst);
        if restart_search {
            if let Err(e) = self.end_search() {
                println!("previous search failed: {e}");
            }
        }
        self.board.play_move(mve);
        *self.control.best_move.lock().unwrap() = None;
        {
            let mut guard = self.control.search.lock().unwrap();
            let search = guard
                .as_mut()
                .expect("search must be some while not searching");

            search.transpositions.clear();
        }
        if restart_search {
            self.start_search();
        }
    }

    fn start_search(&mut self) {
        if self.search_thead.is_some() {
            return;
        }

        self.control.abort_search.store(false, Ordering::SeqCst);

        let search = self.control.search.lock().unwrap().take().unwrap();
        let board = self.board.clone();
        self.search_thead = Some(thread::spawn(move || search.search(board, 10)));
    }

    fn end_search(&mut self) -> Result<()> {
        if let Some(search_thread) = self.search_thead.take() {
            self.control.abort_search.store(true, Ordering::SeqCst);
            match search_thread.join() {
                Ok(result) => result.context("end search")?,
                Err(search_panic) => bail!("Search thread paniced: {search_panic:?}"),
            };
        }

        Ok(())
    }

    fn best_move(&self) -> Option<Move> {
        *self.control.best_move.lock().unwrap()
    }

    fn current_score(&self) -> f32 {
        self.control.score.load(Ordering::SeqCst)
    }
}

impl EllaChess {
    pub fn search_to_depth(&mut self, target_depth: u32) -> Result<(Option<Move>, f32)> {
        if self.search_thead.is_some() {
            bail!("Can't search to depth while search is already running");
        }

        self.control.abort_search.store(false, Ordering::SeqCst);

        let search = self.control.search.lock().unwrap().take().unwrap();
        let board = self.board.clone();

        search.search(board, target_depth)?;

        Ok((self.best_move(), self.current_score()))
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct TranspositionKey {
    zobrist: u64,
    depth: u32,
}

impl TranspositionKey {
    fn new(zobrist: u64, depth: u32) -> Self {
        TranspositionKey { zobrist, depth }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct SearchResult {
    best_move: Option<Move>,
    score: f32,
}

impl SearchResult {
    fn new(best_move: Option<Move>, score: f32) -> Self {
        SearchResult { best_move, score }
    }

    fn score(score: f32) -> Self {
        SearchResult {
            best_move: None,
            score,
        }
    }
}

struct Search {
    control: Arc<SearchControl>,
    transpositions: HashMap<TranspositionKey, SearchResult>,
    transposition_hits: u64,
    positons_checked: u64,
}

impl Search {
    fn new(control: Arc<SearchControl>) -> Self {
        Search {
            control,
            transpositions: HashMap::new(),
            transposition_hits: 0,
            positons_checked: 0,
        }
    }

    fn search(mut self, board: Board, target_depth: u32) -> Result<()> {
        let control = self.control.clone();
        {
            let result = self.search_recursive(&board, target_depth)?;

            *control.best_move.lock().unwrap() = result.best_move;
            control.score.store(result.score, Ordering::SeqCst);
        }
        *control.search.lock().unwrap() = Some(self);

        Ok(())
    }

    fn search_recursive(&mut self, board: &Board, target_depth: u32) -> Result<SearchResult> {
        let transposition_key = TranspositionKey::new(board.zobrist_hash, target_depth);

        self.positons_checked += 1;
        if let Some(transposition) = self.transpositions.get(&transposition_key) {
            self.transposition_hits += 1;
            return Ok(*transposition);
        }

        if target_depth == 0 || self.control.abort_search.load(Ordering::SeqCst) {
            let score = self.estimate_board(board)?;
            return Ok(SearchResult::score(score));
        }

        if board.draw_by_repetition_or_50_moves() {
            let result = SearchResult::score(0.0);
            self.transpositions.insert(transposition_key, result);
            return Ok(result);
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_move = None;

        let moves = board.generate_valid_moves(board.next_move);

        if moves.is_empty() {
            if board.is_in_check(board.next_move) {
                return Ok(SearchResult::score(f32::NEG_INFINITY));
            } else {
                return Ok(SearchResult::score(0.0));
            }
        }

        for mve in moves {
            let mut board = board.clone();
            board.play_move(mve);

            let score = self.search_recursive(&board, target_depth - 1)?.score;
            let score = -score;
            if score > best_score {
                best_move = Some(mve);
                best_score = score;
            }
        }

        let result = SearchResult::new(best_move, best_score);
        self.transpositions.insert(transposition_key, result);
        Ok(result)
    }

    fn estimate_board(&self, board: &Board) -> Result<f32> {
        let mut white_piece_score = 0.0;
        let mut black_piece_score = 0.0;

        for piece in board.fields.iter() {
            if let Some(piece) = piece {
                match piece.color() {
                    Color::White => white_piece_score += Self::piece_value(piece.typ()),
                    Color::Black => black_piece_score += Self::piece_value(piece.typ()),
                }
            }
        }
        match board.next_move {
            Color::White => Ok(white_piece_score - black_piece_score),
            Color::Black => Ok(black_piece_score - white_piece_score),
        }
    }

    #[inline]
    fn piece_value(piece_type: PieceType) -> f32 {
        match piece_type {
            PieceType::King => 0.0, // NOTE: King is at the same time infinite and zero value, because a captured king means you loose
            PieceType::Queen => 9.0,
            PieceType::Bishop => 3.2,
            PieceType::Knight => 3.0,
            PieceType::Rook => 5.0,
            PieceType::Pawn => 1.0,
        }
    }
}
