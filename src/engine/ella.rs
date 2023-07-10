use anyhow::{bail, Context};

use super::Engine;
use crate::{utils::AtomicF32, Board, Color, Move, Piece, PieceType, Result};
use std::{
    cmp,
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

    pub fn stats(&mut self) -> Option<SearchStats> {
        let search_guard = self.control.search.lock().unwrap();
        search_guard.as_ref().map(|s| s.stats.clone())
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
    stats: SearchStats,
}

#[derive(Debug, Clone)]
pub struct SearchStats {
    transposition_hits: u64,
    positions_checked: u64,
    positions_pruned: u64,

    #[cfg(feature = "slow-stats")]
    max_move_count: usize,
}

impl Search {
    fn new(control: Arc<SearchControl>) -> Self {
        Search {
            control,
            transpositions: HashMap::new(),
            stats: SearchStats {
                transposition_hits: 0,
                positions_checked: 0,
                positions_pruned: 0,
                #[cfg(feature = "slow-stats")]
                max_move_count: 0,
            },
        }
    }

    fn search(mut self, mut board: Board, target_depth: u32) -> Result<()> {
        let control = self.control.clone();
        {
            let result = self.search_recursive(
                &mut board,
                1,
                target_depth,
                f32::NEG_INFINITY,
                f32::INFINITY,
            )?;

            *control.best_move.lock().unwrap() = result.best_move;
            control.score.store(result.score, Ordering::SeqCst);
        }
        *control.search.lock().unwrap() = Some(self);

        Ok(())
    }

    fn search_recursive(
        &mut self,
        board: &mut Board,
        depth: u32,
        target_depth: u32,
        mut alpha: f32,
        beta: f32,
    ) -> Result<SearchResult> {
        let transposition_key = TranspositionKey::new(board.zobrist_hash, target_depth);

        self.stats.positions_checked += 1;
        if let Some(transposition) = self.transpositions.get(&transposition_key) {
            self.stats.transposition_hits += 1;
            return Ok(*transposition);
        }

        if target_depth == depth || self.control.abort_search.load(Ordering::SeqCst) {
            let score = self.estimate_board(board)?;
            return Ok(SearchResult::score(score));
        }

        if board.draw_by_repetition_or_50_moves() {
            let result = SearchResult::score(0.0);
            self.transpositions.insert(transposition_key, result);
            return Ok(result);
        }

        let mut moves = board.generate_valid_moves(board.next_move);
        for mve in moves.iter_mut() {
            mve.calculate_move_info_if_missing(board);
        }
        moves.sort_unstable_by(Self::compare_moves);

        #[cfg(feature = "slow-stats")]
        {
            if self.stats.max_move_count < moves.len() {
                self.stats.max_move_count = moves.len();
            }
        }

        if moves.is_empty() {
            if board.is_in_check(board.next_move) {
                return Ok(SearchResult::score(f32::NEG_INFINITY));
            } else {
                return Ok(SearchResult::score(0.0));
            }
        }

        let mut best_move = None;
        let mut best_score = f32::NEG_INFINITY;
        for mve in moves {
            if board.with_temp_move(mve, |board| -> Result<bool> {
                let score = self
                    .search_recursive(board, depth + 1, target_depth, -beta, -alpha)?
                    .score;
                let score = -score;
                alpha = alpha.max(score);
                if score > best_score {
                    best_move = Some(mve);
                    best_score = score;
                }

                if score > beta {
                    self.stats.positions_pruned += 1;
                    Ok(true)
                } else {
                    Ok(false)
                }
            })? {
                break;
            }
        }

        let result = SearchResult::new(best_move, best_score);
        self.transpositions.insert(transposition_key, result);
        Ok(result)
    }

    fn compare_moves(m1: &Move, m2: &Move) -> cmp::Ordering {
        let info1 = m1.additional_info.expect("m1.additional_info must be Some");
        let info2 = m2.additional_info.expect("m2.additional_info must be Some");

        let piece1 = info1.piece;
        let piece2 = info2.piece;
        debug_assert_eq!(piece1.color(), piece2.color());

        if info1.is_check && !info2.is_check {
            return cmp::Ordering::Less;
        }
        if !info1.is_check && info2.is_check {
            return cmp::Ordering::Greater;
        }

        match (info1.captured_piece, info2.captured_piece) {
            (None, None) => cmp::Ordering::Equal,
            (None, Some(captured)) => {
                if Self::caputure_score(piece2, captured) > 0.0 {
                    cmp::Ordering::Less
                } else {
                    cmp::Ordering::Greater
                }
            }
            (Some(captured), None) => {
                if Self::caputure_score(piece1, captured) > 0.0 {
                    cmp::Ordering::Greater
                } else {
                    cmp::Ordering::Less
                }
            }
            (Some(captured1), Some(captured2)) => {
                let score1 = Self::caputure_score(piece1, captured1);
                let score2 = Self::caputure_score(piece2, captured2);

                f32::partial_cmp(&score1, &score2).unwrap()
            }
        }
    }

    #[inline(always)]
    fn caputure_score(piece: Piece, captured: Piece) -> f32 {
        debug_assert_ne!(piece.color(), captured.color());

        if piece.typ() == PieceType::King {
            return Self::piece_value(captured.typ());
        }

        Self::piece_value(captured.typ()) - Self::piece_value(piece.typ())
    }

    #[inline]
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
