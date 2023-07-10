pub mod engine;
pub mod utils;
pub mod zobrist;

use core::fmt;
use std::{
    collections::HashMap,
    fmt::Display,
    hash::Hash,
    mem,
    num::NonZeroU8,
    ops::{Index, IndexMut, Not},
};

use anyhow::{bail, ensure, Context};
use zobrist::ZobristHasher;

pub type Result<T> = std::result::Result<T, anyhow::Error>;

pub const START_BOARD_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0";

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceType {
    King = 0b001,
    Queen = 0b010,
    Bishop = 0b011,
    Knight = 0b100,
    Rook = 0b101,
    Pawn = 0b110,
}

impl PieceType {
    pub const ALL_TYPES: [PieceType; 6] = {
        use PieceType::*;
        [King, Queen, Bishop, Knight, Rook, Pawn]
    };

    pub const ALL_PROMTION_TARGETS: [PieceType; 4] = {
        use PieceType::*;
        [Queen, Rook, Knight, Bishop]
    };
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 0b1000,
}

impl Color {
    pub const ALL_COLORS: [Color; 2] = [Color::White, Color::Black];
}

impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Color::White => f.write_str("White"),
            Color::Black => f.write_str("Black"),
        }
    }
}

impl Not for Color {
    type Output = Color;

    fn not(self) -> Self::Output {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Piece(NonZeroU8);

impl Piece {
    pub fn new(typ: PieceType, color: Color) -> Self {
        // Safety: typ is always > 0
        unsafe { Piece(NonZeroU8::new_unchecked(typ as u8 | color as u8)) }
    }

    #[inline(always)]
    pub fn typ(&self) -> PieceType {
        let typ = self.0.get() & 0b111;
        assert!(typ != 0 && typ != 0b111, "Invalid chess piece");
        unsafe { mem::transmute_copy(&typ) }
    }

    #[inline(always)]
    pub fn color(&self) -> Color {
        if self.0.get() & Color::Black as u8 == Color::Black as u8 {
            Color::Black
        } else {
            Color::White
        }
    }

    pub fn fen_char(&self) -> char {
        match (self.typ(), self.color()) {
            (PieceType::King, Color::White) => 'K',
            (PieceType::King, Color::Black) => 'k',
            (PieceType::Queen, Color::White) => 'Q',
            (PieceType::Queen, Color::Black) => 'q',
            (PieceType::Bishop, Color::White) => 'B',
            (PieceType::Bishop, Color::Black) => 'b',
            (PieceType::Knight, Color::White) => 'N',
            (PieceType::Knight, Color::Black) => 'n',
            (PieceType::Rook, Color::White) => 'R',
            (PieceType::Rook, Color::Black) => 'r',
            (PieceType::Pawn, Color::White) => 'P',
            (PieceType::Pawn, Color::Black) => 'p',
        }
    }

    fn zobrist_index(&self) -> usize {
        match (self.typ(), self.color()) {
            (PieceType::King, Color::White) => 0,
            (PieceType::King, Color::Black) => 1,
            (PieceType::Queen, Color::White) => 2,
            (PieceType::Queen, Color::Black) => 3,
            (PieceType::Bishop, Color::White) => 4,
            (PieceType::Bishop, Color::Black) => 5,
            (PieceType::Knight, Color::White) => 6,
            (PieceType::Knight, Color::Black) => 7,
            (PieceType::Rook, Color::White) => 8,
            (PieceType::Rook, Color::Black) => 9,
            (PieceType::Pawn, Color::White) => 10,
            (PieceType::Pawn, Color::Black) => 11,
        }
    }
}

impl fmt::Debug for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Piece")
            .field("type", &self.typ())
            .field("color", &self.color())
            .finish()
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoveType {
    #[default]
    Normal = 0,
    Castle,
    EnPassant,
    Promotion,
}

// TODO check if memory alignment helps performance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promote_to: Option<Piece>,
    pub typ: MoveType,
    pub additional_info: Option<MoveInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoveInfo {
    pub is_check: bool,
    pub captured_piece: Option<Piece>,
    pub piece: Piece,
}

impl Move {
    pub fn new(from: u8, to: u8) -> Self {
        Move {
            from,
            to,
            promote_to: None,
            typ: MoveType::Normal,
            additional_info: None,
        }
    }

    pub fn new_in_board(from: u8, to: u8, board: &Board) -> Self {
        let piece = board[from].unwrap();

        match piece.typ() {
            PieceType::King => {
                if from.abs_diff(to) == 2 {
                    Self::castle(from, to)
                } else {
                    Self::new(from, to)
                }
            }
            PieceType::Pawn => {
                if let Some(en_passant_square) = board.en_passant_square {
                    if en_passant_square == to {
                        Self::en_passant(from, to)
                    } else {
                        Self::new(from, to)
                    }
                } else {
                    Self::new(from, to)
                }
            }
            _ => Self::new(from, to),
        }
    }

    pub fn en_passant(from: u8, to: u8) -> Self {
        Move {
            from,
            to,
            promote_to: None,
            typ: MoveType::EnPassant,
            additional_info: None,
        }
    }

    pub fn castle(from: u8, to: u8) -> Self {
        Move {
            from,
            to,
            promote_to: None,
            typ: MoveType::Castle,
            additional_info: None,
        }
    }

    pub fn promotion(from: u8, to: u8, target: Piece) -> Self {
        Move {
            from,
            to,
            promote_to: Some(target),
            typ: MoveType::Promotion,
            additional_info: None,
        }
    }

    pub fn calculate_move_info_if_missing(&mut self, board: &Board) {
        if self.additional_info.is_none() {
            self.additional_info = Some(MoveInfo::new_from(&self, board));
        }
    }
}

impl MoveInfo {
    pub fn new_from(mve: &Move, board: &Board) -> Self {
        let captured_piece = board[mve.to];
        let is_check = {
            let piece = board[mve.from].expect("mve.from needs to point to existing piece");

            let moves = Vec::with_capacity(12);
            let moves = board.generate_moves_for_piece_int(mve.to, piece, false, moves);

            moves
                .iter()
                .map(|m| m.to)
                .any(|pos| pos == board.piece_positions(!piece.color()).king)
        };

        MoveInfo {
            is_check,
            captured_piece,
            piece: board[mve.from].unwrap(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiecePositions {
    pub king: u8,
    pub castle_king: bool,
    pub castle_queen: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    fields: [Option<Piece>; 64],

    pub white_pieces: PiecePositions,
    pub black_pieces: PiecePositions,

    pub next_move: Color,

    pub en_passant_square: Option<u8>,

    pub half_moves_since_capture_or_pawn_move: u8,
    pub full_move_count: u32,

    pub repetition_counter: HashMap<u64, u8>,

    pub zobrist_hash: u64,
    pub zobrist_hasher: &'static ZobristHasher<u64>,
}

struct BoardUndoSaveState {
    zobrist_hash: u64,
    en_passant_square: Option<u8>,
    half_moves_since_capture: u8,
    white_castle_king: bool,
    white_castle_queen: bool,
    black_castle_king: bool,
    black_castle_queen: bool,
    /// The piece captured by the move that is associated with this undo state
    /// If the move is an EnPassant, this should be `None` and is assumed to be
    /// a enemy pawn
    captured_piece: Option<Piece>,
}

impl Board {
    pub fn empty() -> Self {
        Board {
            fields: [None; 64],
            white_pieces: PiecePositions {
                king: 0,
                castle_king: true,
                castle_queen: true,
            },
            black_pieces: PiecePositions {
                king: 0,
                castle_king: true,
                castle_queen: true,
            },
            next_move: Color::White,
            en_passant_square: None,
            half_moves_since_capture_or_pawn_move: 0,
            full_move_count: 0,
            repetition_counter: HashMap::new(),
            zobrist_hash: 0,
            zobrist_hasher: &zobrist::ZOBRIST_HASHER,
        }
    }

    pub fn from_fen(fen: &str) -> Result<Self> {
        let mut fields = [None; 64];

        let mut idx = 0;

        let mut iter = fen.chars();

        let mut white_king = 0;
        let mut black_king = 0;

        // parse piece positions
        for char in &mut iter {
            if idx == 64 {
                if char != ' ' {
                    bail!("Expected ' ' after pieces".to_string());
                }
                break;
            }
            match char {
                'k' => {
                    let piece = Piece::new(PieceType::King, Color::Black);
                    fields[idx] = Some(piece);
                    black_king = idx;
                    idx += 1;
                }
                'q' => {
                    let piece = Piece::new(PieceType::Queen, Color::Black);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'b' => {
                    let piece = Piece::new(PieceType::Bishop, Color::Black);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'n' => {
                    let piece = Piece::new(PieceType::Knight, Color::Black);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'r' => {
                    let piece = Piece::new(PieceType::Rook, Color::Black);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'p' => {
                    let piece = Piece::new(PieceType::Pawn, Color::Black);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'K' => {
                    let piece = Piece::new(PieceType::King, Color::White);
                    fields[idx] = Some(piece);
                    white_king = idx;
                    idx += 1;
                }
                'Q' => {
                    let piece = Piece::new(PieceType::Queen, Color::White);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'B' => {
                    let piece = Piece::new(PieceType::Bishop, Color::White);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'N' => {
                    let piece = Piece::new(PieceType::Knight, Color::White);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'R' => {
                    let piece = Piece::new(PieceType::Rook, Color::White);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                'P' => {
                    let piece = Piece::new(PieceType::Pawn, Color::White);
                    fields[idx] = Some(piece);
                    idx += 1;
                }
                '1'..='8' => {
                    let dist = char as u32 - '0' as u32;
                    idx += dist as usize;
                }
                '/' => {
                    if idx % 8 != 0 {
                        bail!("'/' must come between 2 lines".to_string());
                    }
                }
                _ => bail!(format!("unexpected '{char}' instead of piece")),
            }
        }

        let next_move = match iter.next() {
            Some('b') => Color::Black,
            Some('w') => Color::White,
            _ => bail!("Expected either 'w' or 'b' to move".to_string()),
        };

        if iter.next() != Some(' ') {
            bail!("Expected ' ' after move color".to_string());
        }

        let mut white_castle_king: bool = false;
        let mut white_castle_queen: bool = false;
        let mut black_castle_king: bool = false;
        let mut black_castle_queen: bool = false;

        loop {
            match iter.next() {
                Some('-') => {
                    if iter.next() != Some(' ') {
                        bail!("Expected ' ' after '-' for castling".to_string());
                    }
                    break;
                }
                Some('k') => black_castle_king = true,
                Some('q') => black_castle_queen = true,
                Some('K') => white_castle_king = true,
                Some('Q') => white_castle_queen = true,
                Some(' ') => break,
                c => bail!(format!("Expected castling availability but got {c:?}")),
            }
        }

        let en_passant_square = {
            let col = iter.next();
            if col.is_none() {
                bail!("Expected en-passant".to_string());
            }
            let col = col.unwrap();
            if col != '-' {
                let row = iter.next();
                if row.is_none() {
                    bail!("Expected en-passant row".to_string());
                }
                let row = row.unwrap();

                let square = format!("{}{}", col, row);

                Some(Self::square_name_to_pos(&square)?)
            } else {
                None
            }
        };

        if iter.next() != Some(' ') {
            bail!("Expected ' ' after en-passant".to_string());
        }

        let clocks_str: String = iter.collect();
        let mut clocks = clocks_str.trim_end().split(' ');

        let half_moves_since_capture: u8 = if let Some(half_moves) = clocks.next() {
            half_moves
                .parse()
                .context("Could not parse half-move-count")?
        } else {
            bail!("expected half move count".to_string());
        };

        let full_move_count: u32 = if let Some(full_moves) = clocks.next() {
            full_moves.parse().context("Could not parse move-count")?
        } else {
            bail!("Expected move count".to_string());
        };

        if clocks.next().is_some() {
            bail!("Expected end of FEN".to_string());
        }

        let white_pieces = PiecePositions {
            king: white_king as u8,
            castle_king: white_castle_king,
            castle_queen: white_castle_queen,
        };
        let black_pieces = PiecePositions {
            king: black_king as u8,
            castle_king: black_castle_king,
            castle_queen: black_castle_queen,
        };

        let zobrist_hasher = &zobrist::ZOBRIST_HASHER;
        let zobrist_hash = zobrist_hasher.hash(
            &fields,
            &black_pieces,
            &white_pieces,
            en_passant_square,
            next_move,
        );

        let mut previous_positions = HashMap::new();
        previous_positions.insert(zobrist_hash, 1);

        let board = Board {
            fields,
            white_pieces,
            black_pieces,
            next_move,
            en_passant_square,
            half_moves_since_capture_or_pawn_move: half_moves_since_capture,
            full_move_count,
            repetition_counter: previous_positions,
            zobrist_hash,
            zobrist_hasher,
        };

        Ok(board)
    }

    pub fn square_name_to_pos(square: &str) -> Result<u8> {
        ensure!(square.chars().count() == 2);

        let mut chars = square.chars();

        let col = chars.next().unwrap();
        let row = chars.next().unwrap();

        ensure!(('a'..='h').contains(&col));
        let col = col as u32 - 'a' as u32;

        ensure!(('1'..='8').contains(&row));
        let row = row as u32 - '1' as u32;

        Ok(((7 - row) * 8 + col) as u8)
    }

    pub fn calculate_zobrist_hash(&self) -> u64 {
        self.zobrist_hasher.hash(
            &self.fields,
            &self.black_pieces,
            &self.white_pieces,
            self.en_passant_square,
            self.next_move,
        )
    }

    pub fn generate_fen(&self) -> String {
        let mut fen = String::new();

        for row in 0..8 {
            let mut empty_count = 0;
            for col in 0..8 {
                let bindex = (row * 8 + col) as u8;
                match self[bindex] {
                    Some(piece) => {
                        if empty_count > 0 {
                            fen.push(empty_count.to_string().chars().next().unwrap());
                            empty_count = 0;
                        }
                        fen.push(piece.fen_char())
                    }
                    None => empty_count += 1,
                }
            }
            if empty_count > 0 {
                fen.push(empty_count.to_string().chars().next().unwrap());
            }
            if row != 7 {
                fen.push('/');
            }
        }

        fen.push(' ');
        match self.next_move {
            Color::White => fen.push('w'),
            Color::Black => fen.push('b'),
        }
        fen.push(' ');

        if !self.white_pieces.castle_king
            && !self.white_pieces.castle_queen
            && !self.black_pieces.castle_king
            && !self.black_pieces.castle_queen
        {
            fen.push('-');
        } else {
            if self.white_pieces.castle_king {
                fen.push('K');
            }
            if self.white_pieces.castle_queen {
                fen.push('Q');
            }
            if self.black_pieces.castle_king {
                fen.push('k');
            }
            if self.black_pieces.castle_queen {
                fen.push('q');
            }
        }
        fen.push(' ');

        if let Some(en_passant_square) = self.en_passant_square {
            let row = 8 - (en_passant_square / 8);
            let col = (en_passant_square % 8) as u32;
            fen.push(char::from_u32(('a' as u32) + col).unwrap());
            fen.push_str(&row.to_string());
        } else {
            fen.push('-');
        }
        fen.push(' ');

        fen.push_str(&self.half_moves_since_capture_or_pawn_move.to_string());
        fen.push(' ');
        fen.push_str(&self.full_move_count.to_string());

        fen
    }

    pub fn draw_by_repetition_or_50_moves(&self) -> bool {
        if self.repetition_counter[&self.zobrist_hash] >= 3 {
            return true;
        }
        if self.half_moves_since_capture_or_pawn_move >= 100 {
            return true;
        }
        false
    }

    pub fn winner(&mut self) -> Option<Option<Color>> {
        let valid_moves = self.generate_valid_moves(self.next_move);
        if valid_moves.is_empty() {
            if self.is_in_check(self.next_move) {
                Some(Some(!self.next_move))
            } else {
                Some(None)
            }
        } else {
            None
        }
    }

    #[inline]
    pub fn piece_positions(&self, color: Color) -> &PiecePositions {
        match color {
            Color::White => &self.white_pieces,
            Color::Black => &self.black_pieces,
        }
    }

    #[inline]
    pub fn piece_positions_mut(&mut self, color: Color) -> &mut PiecePositions {
        match color {
            Color::White => &mut self.white_pieces,
            Color::Black => &mut self.black_pieces,
        }
    }

    pub fn is_in_check(&self, king_color: Color) -> bool {
        let king_pos = self.piece_positions(king_color).king;
        self.generate_attacking_moves(!king_color)
            .iter()
            .any(|m| m.to == king_pos)
    }

    /// Checks if a given move is valid. This does ignore [Board.next_move] and
    /// instead pretends that it is temporarialy set to the color of the piece
    /// in `mve`.
    pub fn is_valid_move(&mut self, mve: Move) -> bool {
        if let Some(piece) = self[mve.from] {
            let color = piece.color();
            let orig_color = self.next_move;

            // NOTE: play asserts the color is matching
            self.next_move = color;
            let in_check = self.with_temp_move(mve, |board| board.is_in_check(color));
            self.next_move = orig_color;

            !in_check
        } else {
            false
        }
    }

    fn generate_valid_moves_ignore_checks(&self, color: Color) -> Vec<Move> {
        let mut moves = Vec::with_capacity(218);
        for pos in 0..64u8 {
            if let Some(piece) = self[pos] {
                if piece.color() == color {
                    moves = self.generate_moves_for_piece_int(pos, piece, true, moves);
                }
            }
        }
        moves
    }

    pub fn generate_valid_moves(&mut self, color: Color) -> Vec<Move> {
        if self.draw_by_repetition_or_50_moves() {
            return vec![];
        }
        let mut moves = self.generate_valid_moves_ignore_checks(color);
        moves.retain(|m| self.is_valid_move(*m));
        moves
    }

    /// Generate attacking moves to check for possible checks.
    /// This skips castling and validity checks
    pub fn generate_attacking_moves(&self, color: Color) -> Vec<Move> {
        let mut moves = Vec::with_capacity(218);
        for pos in 0..64u8 {
            if let Some(piece) = self[pos] {
                if piece.color() == color {
                    moves = self.generate_moves_for_piece_int(pos, piece, false, moves);
                }
            }
        }
        moves
    }

    pub fn generate_valid_moves_for_piece(&mut self, piece_at: u8) -> Vec<Move> {
        if self.draw_by_repetition_or_50_moves() {
            return vec![];
        }
        let piece = self[piece_at];
        if piece.is_none() {
            return vec![];
        }
        let piece = piece.unwrap();
        if piece.color() != self.next_move {
            return vec![];
        }

        // max moves for queen is 27, rook 14, bishop/pawn 12, king/knight 8
        // however most of the time 14 is probably good enough
        let moves = Vec::with_capacity(12);
        let mut moves = self.generate_moves_for_piece_int(piece_at, piece, true, moves);
        moves.retain(|m| self.is_valid_move(*m));
        moves
    }

    fn generate_moves_for_piece_int(
        &self,
        piece_at: u8,
        piece: Piece,
        allow_castle: bool,
        moves: Vec<Move>,
    ) -> Vec<Move> {
        match piece.typ() {
            PieceType::King => self.generate_king_moves(piece_at, piece, allow_castle, moves),
            PieceType::Queen => self.generate_sliding_moves(piece_at, piece, true, true, moves),
            PieceType::Bishop => self.generate_sliding_moves(piece_at, piece, false, true, moves),
            PieceType::Knight => self.generate_knight_moves(piece_at, piece, moves),
            PieceType::Rook => self.generate_sliding_moves(piece_at, piece, true, false, moves),
            PieceType::Pawn => self.generate_pawn_moves(piece_at, piece, moves),
        }
    }

    fn generate_knight_moves(&self, piece_at: u8, piece: Piece, mut moves: Vec<Move>) -> Vec<Move> {
        let color = piece.color();
        let x = (piece_at % 8) as i8;
        let y = (piece_at / 8) as i8;

        let offsets: [(i8, i8); 8] = [
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
            (1, 2),
        ];

        for offset in offsets {
            let x = x + offset.0;
            let y = y + offset.1;

            if !(0..8).contains(&x) || !(0..8).contains(&y) {
                continue;
            }

            let target_index = (y * 8 + x) as u8;

            match self[target_index] {
                None => moves.push(Move::new(piece_at, target_index)),
                Some(other) => {
                    if other.color() != color {
                        moves.push(Move::new(piece_at, target_index));
                    }
                }
            }
        }

        moves
    }

    fn generate_pawn_moves(&self, piece_at: u8, piece: Piece, mut moves: Vec<Move>) -> Vec<Move> {
        let color = piece.color();
        let (dir, promotion_target, double_move_start) = match color {
            Color::White => (-1, 0, 6),
            Color::Black => (1, 7, 1),
        };

        let x = (piece_at % 8) as i8;
        let y = (piece_at / 8) as i8;

        fn bidx<X: TryInto<u8>, Y: TryInto<u8>>(x: X, y: Y) -> Option<u8> {
            let x = x.try_into().ok()?;
            let y = y.try_into().ok()?;
            // try into checks for < 0
            if x >= 8 || y >= 8 {
                return None;
            }
            Some(x + y * 8)
        }

        // add possible moves to `target`, including possible promotions
        let mut add_moves = |target, y, en_passant| {
            if en_passant {
                moves.push(Move::en_passant(piece_at, target));
                return false;
            }
            if y == promotion_target {
                for promo in PieceType::ALL_PROMTION_TARGETS {
                    moves.push(Move::promotion(piece_at, target, Piece::new(promo, color)));
                }
                false
            } else {
                moves.push(Move::new(piece_at, target));
                true
            }
        };

        // move 1 forward
        if let Some(target_idx) = bidx(x, y + dir) {
            if self[target_idx].is_none() && add_moves(target_idx, y + dir, false) {
                // check if we can move 2 forward
                let target_idx = bidx(x, y + 2 * dir).unwrap();
                if y == double_move_start && self[target_idx].is_none() {
                    add_moves(target_idx, y + 2 * dir, false);
                }
            }
        }

        // check capure and en-passant
        if let Some(target_idx) = bidx(x - 1, y + dir) {
            if self.en_passant_square == Some(target_idx) {
                add_moves(target_idx, y + dir, true);
            } else if let Some(other) = self[target_idx] {
                if other.color() != color {
                    add_moves(target_idx, y + dir, false);
                }
            }
        }
        if let Some(target_idx) = bidx(x + 1, y + dir) {
            if self.en_passant_square == Some(target_idx) {
                add_moves(target_idx, y + dir, true);
            } else if let Some(other) = self[target_idx] {
                if other.color() != color {
                    add_moves(target_idx, y + dir, false);
                }
            }
        }

        moves
    }

    fn generate_king_moves(
        &self,
        piece_at: u8,
        piece: Piece,
        allow_castle: bool,
        mut moves: Vec<Move>,
    ) -> Vec<Move> {
        let color = piece.color();

        let mut search = |x: i8, y: i8| {
            if !(0..8).contains(&x) || !(0..8).contains(&y) {
                return;
            }
            let bidx = (y * 8 + x) as u8;
            if let Some(other) = self[bidx] {
                if other.color() != color {
                    moves.push(Move::new(piece_at, bidx));
                }
            } else {
                moves.push(Move::new(piece_at, bidx));
            }
        };

        let x = (piece_at % 8) as i8;
        let y = (piece_at / 8) as i8;

        search(x - 1, y - 1);
        search(x, y - 1);
        search(x + 1, y - 1);
        search(x + 1, y);
        search(x + 1, y + 1);
        search(x, y + 1);
        search(x - 1, y + 1);
        search(x - 1, y);

        if allow_castle {
            let (castle_row, pawn_row) = match color {
                Color::White => (7, 6),
                Color::Black => (0, 1),
            };

            let bidx = |x, y| (y * 8 + x) as u8;

            let piece_positions = self.piece_positions(color);
            let castle_king = piece_positions.castle_king
                && self[bidx(5, castle_row)].is_none()
                && self[bidx(6, castle_row)].is_none();
            let castle_queen = piece_positions.castle_queen
                && self[bidx(1, castle_row)].is_none()
                && self[bidx(2, castle_row)].is_none()
                && self[bidx(3, castle_row)].is_none();

            if !castle_king && !castle_queen {
                return moves;
            }

            // HACK: generate_attacking moves does not generate attack moves
            // for pawns to empty squares. Therefor we have to manually detect,
            // if an enemy pawn blocks promotion
            if let Some(piece_in_front_of_king) = self[bidx(4, pawn_row)] {
                if piece_in_front_of_king.typ() == PieceType::Pawn
                    && piece_in_front_of_king.color() != color
                {
                    return moves;
                }
            }

            let enemy_moves = self.generate_attacking_moves(!color);

            if castle_king
                && !enemy_moves.iter().any(|m| {
                    m.to == bidx(4, castle_row)
                        || m.to == bidx(5, castle_row)
                        || m.to == bidx(6, castle_row)
                })
            {
                moves.push(Move::castle(piece_at, bidx(6, castle_row)));
            }
            if castle_queen
                && !enemy_moves.iter().any(|m| {
                    m.to == bidx(2, castle_row)
                        || m.to == bidx(3, castle_row)
                        || m.to == bidx(4, castle_row)
                })
            {
                moves.push(Move::castle(piece_at, bidx(2, castle_row)));
            }
        }

        moves
    }

    fn generate_sliding_moves(
        &self,
        piece_at: u8,
        piece: Piece,
        rook: bool,
        bishop: bool,
        mut moves: Vec<Move>,
    ) -> Vec<Move> {
        let x = piece_at % 8;
        let y = piece_at / 8;
        let color = piece.color();

        let mut calculate_moves = |off1: u8, off2: u8| {
            let mut search = |x_off, y_off| {
                let mut pos_x = x as i8;
                let mut pos_y = y as i8;
                loop {
                    pos_x += x_off;
                    pos_y += y_off;
                    if !(0..8).contains(&pos_x) || !(0..8).contains(&pos_y) {
                        break;
                    }
                    let board_idx = (pos_x + pos_y * 8) as u8;
                    let target_piece = self[board_idx];
                    match target_piece {
                        None => moves.push(Move::new(piece_at, board_idx)),
                        Some(p) => {
                            if p.color() != color {
                                moves.push(Move::new(piece_at, board_idx));
                            }
                            break;
                        }
                    }
                }
            };

            let up_x = off2 as i8;
            let up_y = -(off1 as i8);
            let right_x = off1 as i8;
            let right_y = off2 as i8;
            let down_x = -(off2 as i8);
            let down_y = off1 as i8;
            let left_x = -(off1 as i8);
            let left_y = -(off2 as i8);

            search(up_x, up_y);
            search(right_x, right_y);
            search(down_x, down_y);
            search(left_x, left_y);
        };

        if rook {
            calculate_moves(1, 0);
        }
        if bishop {
            calculate_moves(1, 1);
        }

        moves
    }

    /// plays a given move. This assumes that the move is valid
    pub fn play_move(&mut self, mve: Move) {
        debug_assert!(self[mve.from].is_some());
        debug_assert_eq!(self[mve.from].unwrap().color(), self.next_move);

        let zobrist_hasher = self.zobrist_hasher;

        self.next_move = !self.next_move;
        if self.next_move == Color::White {
            self.full_move_count += 1;
        }
        self.zobrist_hash ^= zobrist_hasher.black_move;

        self.half_moves_since_capture_or_pawn_move += 1;

        match mve.typ {
            MoveType::Normal => {
                let piece = self[mve.from].unwrap();
                let captured = self[mve.to];

                // clear out old position
                self[mve.from] = None;
                self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.from, piece);

                // remove captured from hash
                if let Some(captured) = captured {
                    debug_assert_ne!(captured.color(), piece.color());
                    self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.to, captured);

                    // disable castling if captured piece is rook
                    let mut hash_update = 0;
                    if mve.to == 0 && captured.color() == Color::Black {
                        if self.black_pieces.castle_queen {
                            hash_update ^= zobrist_hasher.black_king.castle_queen;
                            self.black_pieces.castle_queen = false;
                        }
                    } else if mve.to == 7 && captured.color() == Color::Black {
                        if self.black_pieces.castle_king {
                            hash_update ^= zobrist_hasher.black_king.castle_king;
                            self.black_pieces.castle_king = false;
                        }
                    } else if mve.to == 56 && captured.color() == Color::White {
                        if self.white_pieces.castle_queen {
                            hash_update ^= zobrist_hasher.white_king.castle_queen;
                            self.white_pieces.castle_queen = false;
                        }
                    } else if mve.to == 63 && captured.color() == Color::White {
                        if self.white_pieces.castle_king {
                            hash_update ^= zobrist_hasher.white_king.castle_king;
                            self.white_pieces.castle_king = false;
                        }
                    }

                    self.zobrist_hash ^= hash_update;

                    self.half_moves_since_capture_or_pawn_move = 0;
                }

                // move piece into new positon
                self[mve.to] = Some(piece);
                self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.to, piece);

                // disable en_passant
                if let Some(old_en_passant) = self.en_passant_square {
                    self.zobrist_hash ^= self.zobrist_hasher.en_passant_hash(old_en_passant);
                }
                self.en_passant_square = None;

                // update castling rights and piece positions and reenable en_passant
                match piece.typ() {
                    PieceType::King => {
                        let king_hasher = zobrist_hasher.king_move(piece.color());
                        let piece_positions = self.piece_positions_mut(piece.color());
                        let mut hash_update = 0;
                        piece_positions.king = mve.to;
                        if piece_positions.castle_king {
                            hash_update ^= king_hasher.castle_king;
                            piece_positions.castle_king = false;
                        }
                        if piece_positions.castle_queen {
                            hash_update ^= king_hasher.castle_queen;
                            piece_positions.castle_queen = false;
                        }

                        self.zobrist_hash ^= hash_update;
                    }
                    PieceType::Rook => {
                        let mut hash_update = 0;
                        if mve.from == 0 && piece.color() == Color::Black {
                            if self.black_pieces.castle_queen {
                                hash_update ^= zobrist_hasher.black_king.castle_queen;
                                self.black_pieces.castle_queen = false;
                            }
                        } else if mve.from == 7 && piece.color() == Color::Black {
                            if self.black_pieces.castle_king {
                                hash_update ^= zobrist_hasher.black_king.castle_king;
                                self.black_pieces.castle_king = false;
                            }
                        } else if mve.from == 56 && piece.color() == Color::White {
                            if self.white_pieces.castle_queen {
                                hash_update ^= zobrist_hasher.white_king.castle_queen;
                                self.white_pieces.castle_queen = false;
                            }
                        } else if mve.from == 63 && piece.color() == Color::White {
                            if self.white_pieces.castle_king {
                                hash_update ^= zobrist_hasher.white_king.castle_king;
                                self.white_pieces.castle_king = false;
                            }
                        }

                        self.zobrist_hash ^= hash_update;
                    }
                    PieceType::Pawn => {
                        self.half_moves_since_capture_or_pawn_move = 0;

                        // if pawn moved 2 squares, update en_passant
                        if mve.from.abs_diff(mve.to) == 16 {
                            self.en_passant_square = Some(
                                (mve.from as i16 + (mve.to as i16 - mve.from as i16) / 2) as u8,
                            );
                            self.zobrist_hash ^=
                                zobrist_hasher.en_passant_hash(self.en_passant_square.unwrap());
                        }
                    }
                    _ => (),
                }
            }
            MoveType::Castle => {
                let king = self[mve.from].unwrap();
                debug_assert_eq!(king.typ(), PieceType::King);

                let (rook_from, rook_to): (u8, u8) = match mve.to {
                    2 => {
                        debug_assert_eq!(king.color(), Color::Black);
                        (0, 3)
                    }
                    6 => {
                        debug_assert_eq!(king.color(), Color::Black);
                        (7, 5)
                    }
                    58 => {
                        debug_assert_eq!(king.color(), Color::White);
                        (56, 59)
                    }
                    62 => {
                        debug_assert_eq!(king.color(), Color::White);
                        (63, 61)
                    }
                    _ => panic!("castling only possible targeting 4 specific squares"),
                };

                let rook = self[rook_from].unwrap();
                debug_assert_eq!(rook.color(), king.color());

                // disabling castling, you can only castle once after all
                {
                    let piece_positions = self.piece_positions_mut(king.color());
                    let king_hasher = zobrist_hasher.king_move(king.color());

                    let mut hash_update = 0;
                    if piece_positions.castle_king {
                        piece_positions.castle_king = false;
                        hash_update ^= king_hasher.castle_king;
                    }
                    if piece_positions.castle_queen {
                        piece_positions.castle_queen = false;
                        hash_update ^= king_hasher.castle_queen;
                    }
                    self.zobrist_hash ^= hash_update;
                }

                // disable en_passant
                if let Some(old_en_passant) = self.en_passant_square {
                    self.zobrist_hash ^= self.zobrist_hasher.en_passant_hash(old_en_passant);
                }
                self.en_passant_square = None;

                // remove king from old square
                self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.from, king);
                self[mve.from] = None;

                // remvoe rook from old square
                self.zobrist_hash ^= zobrist_hasher.piece_hash(rook_from, rook);
                self[rook_from] = None;

                // move king to new position
                self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.to, king);
                self[mve.to] = Some(king);
                self.piece_positions_mut(king.color()).king = mve.to;

                // move the rook to new position
                self.zobrist_hash ^= zobrist_hasher.piece_hash(rook_to, rook);
                self[rook_to] = Some(rook);
            }
            MoveType::EnPassant => {
                let pawn = self[mve.from].unwrap();
                debug_assert_eq!(pawn.typ(), PieceType::Pawn);
                debug_assert!(self.en_passant_square.is_some());
                debug_assert_eq!(mve.to, self.en_passant_square.unwrap());

                let capture_pos = if pawn.color() == Color::White {
                    mve.to + 8
                } else {
                    mve.to - 8
                };
                let captured_pawn =
                    self[capture_pos].expect("EnPassant must always capture a piece");
                debug_assert_eq!(captured_pawn.typ(), PieceType::Pawn);
                debug_assert_ne!(captured_pawn.color(), pawn.color());

                // remove captured pawn
                self.zobrist_hash ^= zobrist_hasher.piece_hash(capture_pos, captured_pawn);
                self[capture_pos] = None;

                // remove pawn from old pos
                self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.from, pawn);
                self[mve.from] = None;

                // move pawn to new pos
                self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.to, pawn);
                self[mve.to] = Some(pawn);

                // disable en_passant
                if let Some(old_en_passant) = self.en_passant_square {
                    self.zobrist_hash ^= self.zobrist_hasher.en_passant_hash(old_en_passant);
                }
                self.en_passant_square = None;

                self.half_moves_since_capture_or_pawn_move = 0;
            }
            MoveType::Promotion => {
                let piece = self[mve.from].unwrap();
                let captured = self[mve.to];
                let promote_to = mve
                    .promote_to
                    .expect("Promotion move needs promotion target");
                debug_assert_eq!(piece.color(), promote_to.color());

                // clear out old position
                self[mve.from] = None;
                self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.from, piece);

                // remove captured from hash
                if let Some(captured) = captured {
                    debug_assert_ne!(captured.color(), piece.color());
                    self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.to, captured);

                    // disable castling if captured piece is rook
                    let mut hash_update = 0;
                    if mve.to == 0 && captured.color() == Color::Black {
                        if self.black_pieces.castle_queen {
                            hash_update ^= zobrist_hasher.black_king.castle_queen;
                            self.black_pieces.castle_queen = false;
                        }
                    } else if mve.to == 7 && captured.color() == Color::Black {
                        if self.black_pieces.castle_king {
                            hash_update ^= zobrist_hasher.black_king.castle_king;
                            self.black_pieces.castle_king = false;
                        }
                    } else if mve.to == 56 && captured.color() == Color::White {
                        if self.white_pieces.castle_queen {
                            hash_update ^= zobrist_hasher.white_king.castle_queen;
                            self.white_pieces.castle_queen = false;
                        }
                    } else if mve.to == 63 && captured.color() == Color::White {
                        if self.white_pieces.castle_king {
                            hash_update ^= zobrist_hasher.white_king.castle_king;
                            self.white_pieces.castle_king = false;
                        }
                    }

                    self.zobrist_hash ^= hash_update;
                }

                // move piece into new positon
                self[mve.to] = Some(promote_to);
                self.zobrist_hash ^= zobrist_hasher.piece_hash(mve.to, promote_to);

                // disable en_passant
                if let Some(old_en_passant) = self.en_passant_square {
                    self.zobrist_hash ^= self.zobrist_hasher.en_passant_hash(old_en_passant);
                }
                self.en_passant_square = None;

                self.half_moves_since_capture_or_pawn_move = 0;
            }
        }

        *self
            .repetition_counter
            .entry(self.zobrist_hash)
            .or_insert(0) += 1;
    }

    pub fn with_temp_move<F, R>(&mut self, mve: Move, func: F) -> R
    where
        F: FnOnce(&mut Board) -> R,
    {
        let captured = self[mve.to];
        let save_state = self.create_undo_save_state(captured);
        self.play_move(mve);

        let result = func(self);

        self.undo_move(mve, save_state);

        result
    }

    #[inline]
    fn undo_move(&mut self, mve: Move, save_state: BoardUndoSaveState) {
        // decrement repetition counter of the move we undo
        *self.repetition_counter.get_mut(&self.zobrist_hash).unwrap() -= 1;

        // undo next_move change
        self.next_move = !self.next_move;
        if self.next_move == Color::Black {
            self.full_move_count -= 1;
        }

        match mve.typ {
            MoveType::Normal => {
                let piece = self[mve.to].unwrap();

                debug_assert_ne!(
                    save_state
                        .captured_piece
                        .map(|p| p.color())
                        .unwrap_or_else(|| !piece.color()),
                    piece.color()
                );

                // move piece to original position
                self[mve.from] = Some(piece);

                // clear out piece or change back to captured
                self[mve.to] = save_state.captured_piece;

                // update king position
                if piece.typ() == PieceType::King {
                    self.piece_positions_mut(piece.color()).king = mve.from;
                }
            }
            MoveType::Castle => {
                let king = self[mve.to].unwrap();
                let (rook_from, rook_to): (u8, u8) = match mve.to {
                    2 => {
                        debug_assert_eq!(king.color(), Color::Black);
                        (0, 3)
                    }
                    6 => {
                        debug_assert_eq!(king.color(), Color::Black);
                        (7, 5)
                    }
                    58 => {
                        debug_assert_eq!(king.color(), Color::White);
                        (56, 59)
                    }
                    62 => {
                        debug_assert_eq!(king.color(), Color::White);
                        (63, 61)
                    }
                    _ => panic!("castling only possible targeting 4 specific squares"),
                };
                let rook = self[rook_to].unwrap();
                debug_assert_eq!(rook.color(), king.color());

                // remove king
                self[mve.to] = None;

                // remove rook
                self[rook_to] = None;

                // put back king to original position
                self[mve.from] = Some(king);
                self.piece_positions_mut(king.color()).king = mve.from;

                // put back rook to original position
                self[rook_from] = Some(rook);
            }
            MoveType::EnPassant => {
                let pawn = self[mve.to].unwrap();
                let captured_pawn = Piece::new(PieceType::Pawn, !pawn.color());

                debug_assert_eq!(save_state.captured_piece, None);

                // move pawn to original position
                self[mve.from] = Some(pawn);

                // clear out pawn
                self[mve.to] = None;

                // put back captured pawn
                let capture_pos = if pawn.color() == Color::White {
                    mve.to + 8
                } else {
                    mve.to - 8
                };
                self[capture_pos] = Some(captured_pawn);
            }
            MoveType::Promotion => {
                let promoted_piece = self[mve.to].unwrap();
                let pawn = Piece::new(PieceType::Pawn, promoted_piece.color());

                debug_assert_eq!(promoted_piece, mve.promote_to.unwrap());
                debug_assert_ne!(
                    save_state
                        .captured_piece
                        .map(|p| p.color())
                        .unwrap_or_else(|| !pawn.color()),
                    pawn.color()
                );

                // move pawn to original position
                self[mve.from] = Some(pawn);

                // clear out promoted piece or change back to captured
                self[mve.to] = save_state.captured_piece;
            }
        }

        self.zobrist_hash = save_state.zobrist_hash;
        self.en_passant_square = save_state.en_passant_square;
        self.half_moves_since_capture_or_pawn_move = save_state.half_moves_since_capture;
        self.white_pieces.castle_king = save_state.white_castle_king;
        self.white_pieces.castle_queen = save_state.white_castle_queen;
        self.black_pieces.castle_king = save_state.black_castle_king;
        self.black_pieces.castle_queen = save_state.black_castle_queen;
    }

    /// captured_piece should be none for EnPassant captures, as those are always pawns
    fn create_undo_save_state(&self, captured_piece: Option<Piece>) -> BoardUndoSaveState {
        BoardUndoSaveState {
            zobrist_hash: self.zobrist_hash,
            en_passant_square: self.en_passant_square,
            half_moves_since_capture: self.half_moves_since_capture_or_pawn_move,
            white_castle_king: self.white_pieces.castle_king,
            white_castle_queen: self.white_pieces.castle_queen,
            black_castle_king: self.black_pieces.castle_king,
            black_castle_queen: self.black_pieces.castle_queen,
            captured_piece,
        }
    }
}

impl Index<usize> for Board {
    type Output = Option<Piece>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.fields[index]
    }
}

impl IndexMut<usize> for Board {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.fields[index]
    }
}

impl Index<u8> for Board {
    type Output = Option<Piece>;

    fn index(&self, index: u8) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<u8> for Board {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

pub mod test_fens {
    use crate::START_BOARD_FEN;

    pub const MOST_POSS_MOVES_FEN: &str = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNNK1B1 w - - 0 0";
    pub const MOST_POSS_MOVES_FEN_B: &str = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNNK1B1 b - - 0 0";
    pub const MANY_POSS_MOVES_FEN_W: &str =
        "r6R/2pbpBk1/1p1B1N2/6q1/4Q3/2nn1p2/1PK1NbP1/R6r w - - 0 0";
    pub const MANY_POSS_MOVES_FEN_B: &str =
        "r6R/2pbpBk1/1p1B1N2/6q1/4Q3/2nn1p2/1PK1NbP1/R6r b - - 0 0";
    pub const EN_PASSANT_POS_W: &str =
        "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 0";
    pub const EN_PASSANT_POS_W2: &str =
        "rnbqkbnr/pppp2pp/4p3/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 0";
    pub const EN_PASSANT_POS_B: &str =
        "rnbqkbnr/ppppp1pp/8/4P3/5pP1/8/PPPP1P1P/RNBQKBNR b KQkq g3 0 0";
    pub const CASTLE_QUEEN_W: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 0";
    pub const CASTLE_KING_W: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 0";
    pub const CASTLE_KING_B: &str = "rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 0";
    pub const CASTLE_QUEEN_B: &str = "r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 0";
    pub const WHITE_PROMOTION: &str = "rnbqkb1r/pppp2Pp/2n5/8/8/8/PPP1PPPP/RNBQKBNR w KQkq - 1 4";
    pub const CAPTURE_BLACK_KING_ROOK: &str =
        "rnbqk2r/pppp1pPp/4p3/2bn4/8/8/PPPPP1PP/RNBQKBNR w KQkq - 1 4";
    pub const CAPTURE_BLACK_QUEEN_ROOK: &str =
        "r3kbnr/p1pp1ppp/bpB1p3/6q1/8/6PN/PPPPPP1P/RNBQ1RK1 w kq - 1 5";
    pub const CAPTURE_WHITE_KING_ROOK: &str =
        "rn1qkb1r/pbpppppp/1p3n2/6N1/8/6PB/PPPPPP1P/RNBQK2R b KQkq - 4 3";
    pub const CAPTURE_WHITE_QUEEN_ROOK: &str =
        "rnbqkbnr/pp1ppppp/8/1N4B1/8/3P4/PpPQPPPP/R3KBNR b KQkq - 1 4";
    pub const PAWN_IN_FRONT_OF_KING_NO_CASTLE: &str =
        "r3k2r/p1ppPpb1/bn2pnp1/4N3/4P3/1pN2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1";

    /// https://www.chessprogramming.org/Perft_Results
    pub const POSITION_2: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0";
    /// https://www.chessprogramming.org/Perft_Results
    pub const POSITION_3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 0";
    /// https://www.chessprogramming.org/Perft_Results
    pub const POSITION_4_W: &str =
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
    /// https://www.chessprogramming.org/Perft_Results
    pub const POSITION_4_B: &str =
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1";

    pub const ALL_TEST_FENS: &[&str] = &[
        START_BOARD_FEN,
        MOST_POSS_MOVES_FEN,
        MOST_POSS_MOVES_FEN_B,
        MANY_POSS_MOVES_FEN_W,
        MANY_POSS_MOVES_FEN_B,
        EN_PASSANT_POS_W,
        EN_PASSANT_POS_W2,
        EN_PASSANT_POS_B,
        CASTLE_KING_B,
        CASTLE_KING_W,
        CASTLE_QUEEN_B,
        CASTLE_QUEEN_W,
        WHITE_PROMOTION,
        CAPTURE_BLACK_KING_ROOK,
        CAPTURE_BLACK_QUEEN_ROOK,
        CAPTURE_WHITE_KING_ROOK,
        CAPTURE_BLACK_QUEEN_ROOK,
        PAWN_IN_FRONT_OF_KING_NO_CASTLE,
        POSITION_2,
        POSITION_3,
        POSITION_4_W,
        POSITION_4_B,
    ];
}

#[cfg(test)]
mod test {
    use crate::{
        test_fens::{ALL_TEST_FENS, EN_PASSANT_POS_B, EN_PASSANT_POS_W},
        Board, Color, Piece, PieceType,
    };
    use std::collections::HashSet;

    #[test]
    fn pices_are_unique_u8() {
        let mut pieces = HashSet::new();
        for color in Color::ALL_COLORS {
            for typ in PieceType::ALL_TYPES {
                let piece = Piece::new(typ, color);
                assert!(!pieces.contains(&piece));
                pieces.insert(piece);
            }
        }
        assert!(pieces.len() == 12);
    }

    #[test]
    fn parse_fen() {
        for fen in ALL_TEST_FENS {
            assert!(Board::from_fen(fen).is_ok());
        }
    }

    #[test]
    fn fen_round_trip() {
        for fen in ALL_TEST_FENS {
            let board = Board::from_fen(fen).unwrap();
            let generated_fen = board.generate_fen();
            let board_from_generated = Board::from_fen(&generated_fen).unwrap();
            assert_eq!(board, board_from_generated);
        }
    }

    #[test]
    fn parse_fn_en_passant() {
        let board = Board::from_fen(EN_PASSANT_POS_W).unwrap();
        assert_eq!(board.en_passant_square, Some(19));

        let board = Board::from_fen(EN_PASSANT_POS_B).unwrap();
        assert_eq!(board.en_passant_square, Some(46));
    }

    #[test]
    fn temp_move_doesnt_change_state() {
        for fen in ALL_TEST_FENS {
            let mut board = Board::from_fen(fen).unwrap();
            let original = board.clone();
            for mve in board.generate_valid_moves(board.next_move) {
                board.with_temp_move(mve, |_| {});

                // with_temp does not remove 0 counts
                board.repetition_counter.retain(|_, count| *count > 0);

                assert_eq!(board, original, "\n failed to undo move {mve:?} from {fen}");
            }
        }
    }

    mod moves {
        use crate::{
            test_fens::{
                CAPTURE_BLACK_KING_ROOK, CAPTURE_BLACK_QUEEN_ROOK, CAPTURE_WHITE_KING_ROOK,
                CAPTURE_WHITE_QUEEN_ROOK, CASTLE_KING_B, CASTLE_KING_W, CASTLE_QUEEN_B,
                CASTLE_QUEEN_W, EN_PASSANT_POS_B, EN_PASSANT_POS_W, EN_PASSANT_POS_W2,
            },
            Board, Color, Move, Piece, PieceType,
        };

        #[test]
        fn en_passant_white() {
            let mut board = Board::from_fen(EN_PASSANT_POS_W).unwrap();

            let mut expected =
                Board::from_fen("rnbqkbnr/ppp2ppp/3Pp3/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 0")
                    .unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::en_passant(28, 19));

            assert_eq!(board, expected);
        }

        #[test]
        fn en_passant_white_2() {
            let mut board = Board::from_fen(EN_PASSANT_POS_W2).unwrap();

            let mut expected =
                Board::from_fen("rnbqkbnr/pppp2pp/4pP2/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 0")
                    .unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::en_passant(28, 21));

            assert_eq!(board, expected);
        }

        #[test]
        fn en_passant_black() {
            let mut board = Board::from_fen(EN_PASSANT_POS_B).unwrap();

            let mut expected =
                Board::from_fen("rnbqkbnr/ppppp1pp/8/4P3/8/6p1/PPPP1P1P/RNBQKBNR w KQkq - 0 1")
                    .unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::en_passant(37, 46));

            assert_eq!(board, expected);
        }

        #[test]
        fn castle_queen_white() {
            let mut board = Board::from_fen(CASTLE_QUEEN_W).unwrap();

            let mut expected =
                Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/2KR1BNR b kq - 1 0").unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::castle(60, 58));

            assert_eq!(board, expected);
        }

        #[test]
        fn castle_king_white() {
            let mut board = Board::from_fen(CASTLE_KING_W).unwrap();

            let mut expected =
                Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 b kq - 1 0").unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::castle(60, 62));

            assert_eq!(board, expected);
        }

        #[test]
        fn castle_queen_black() {
            let mut board = Board::from_fen(CASTLE_QUEEN_B).unwrap();

            let mut expected =
                Board::from_fen("2kr1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 1 1").unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::castle(4, 2));

            assert_eq!(board, expected);
        }

        #[test]
        fn castle_king_black() {
            let mut board = Board::from_fen(CASTLE_KING_B).unwrap();

            let mut expected =
                Board::from_fen("rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 1 1").unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::castle(4, 6));

            assert_eq!(board, expected);
        }

        #[test]
        fn captured_rook_disables_castle_black_king() {
            let board = Board::from_fen(CAPTURE_BLACK_KING_ROOK).unwrap();
            for promotion in PieceType::ALL_PROMTION_TARGETS {
                let target_piece = Piece::new(promotion, Color::White);

                let mut expected = Board::from_fen(&format!(
                    "rnbqk2{}/pppp1p1p/4p3/2bn4/8/8/PPPPP1PP/RNBQKBNR b KQq - 0 4",
                    target_piece.fen_char()
                ))
                .unwrap();
                expected.repetition_counter.insert(board.zobrist_hash, 1);

                let mut board = board.clone();

                board.play_move(Move::promotion(14, 7, target_piece));

                assert!(!board.black_pieces.castle_king);
                assert_eq!(board, expected);
            }
        }

        #[test]
        fn captured_rook_disables_castle_black_queen() {
            let mut board = Board::from_fen(CAPTURE_BLACK_QUEEN_ROOK).unwrap();

            let mut expected =
                Board::from_fen("B3kbnr/p1pp1ppp/bp2p3/6q1/8/6PN/PPPPPP1P/RNBQ1RK1 b k - 0 5")
                    .unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::new(18, 0));

            assert!(!board.black_pieces.castle_queen);
            assert_eq!(board, expected);
        }

        #[test]
        fn captured_rook_disables_castle_white_king() {
            let mut board = Board::from_fen(CAPTURE_WHITE_KING_ROOK).unwrap();

            let mut expected =
                Board::from_fen("rn1qkb1r/p1pppppp/1p3n2/6N1/8/6PB/PPPPPP1P/RNBQK2b w Qkq - 0 4")
                    .unwrap();
            expected.repetition_counter.insert(board.zobrist_hash, 1);

            board.play_move(Move::new(9, 63));

            assert!(!board.white_pieces.castle_king);
            assert_eq!(board, expected);
        }

        #[test]
        fn captured_rook_disables_castle_white_queen() {
            let board = Board::from_fen(CAPTURE_WHITE_QUEEN_ROOK).unwrap();
            for promotion in PieceType::ALL_PROMTION_TARGETS {
                let target_piece = Piece::new(promotion, Color::Black);
                let mut board = board.clone();

                let mut expected = Board::from_fen(&format!(
                    "rnbqkbnr/pp1ppppp/8/1N4B1/8/3P4/P1PQPPPP/{}3KBNR w Kkq - 0 5",
                    target_piece.fen_char()
                ))
                .unwrap();
                expected.repetition_counter.insert(board.zobrist_hash, 1);

                board.play_move(Move::promotion(49, 56, target_piece));

                assert!(!board.white_pieces.castle_queen);
                assert_eq!(board, expected);
            }
        }
    }

    #[test]
    fn zobrist_hash_valid_after_move() {
        for fen in ALL_TEST_FENS {
            let mut board = Board::from_fen(fen).unwrap();
            for mve in board.generate_valid_moves(board.next_move) {
                let mut board = board.clone();
                board.play_move(mve);
                assert_eq!(
                    board.zobrist_hash,
                    board.calculate_zobrist_hash(),
                    "\nZobrist hash failed for move {mve:?} at fen \"{fen}\""
                );
            }
        }
    }

    mod move_gen {
        use crate::{
            test_fens::{
                CASTLE_KING_B, CASTLE_KING_W, CASTLE_QUEEN_B, CASTLE_QUEEN_W, EN_PASSANT_POS_B,
                EN_PASSANT_POS_W, MANY_POSS_MOVES_FEN_B, MANY_POSS_MOVES_FEN_W,
                MOST_POSS_MOVES_FEN, MOST_POSS_MOVES_FEN_B, PAWN_IN_FRONT_OF_KING_NO_CASTLE,
                WHITE_PROMOTION,
            },
            Board, START_BOARD_FEN,
        };

        #[test]
        fn correct_move_count_1() {
            let mut start = Board::from_fen(START_BOARD_FEN).unwrap();
            assert_eq!(
                start.generate_valid_moves(start.next_move).len(),
                20,
                "White to move"
            );
            assert_eq!(
                start.generate_valid_moves(!start.next_move).len(),
                20,
                "Black to move"
            );
        }

        #[test]
        fn correct_move_count_2() {
            let mut most_moves = Board::from_fen(MOST_POSS_MOVES_FEN).unwrap();
            assert_eq!(
                most_moves.generate_valid_moves(most_moves.next_move).len(),
                216
            );
            assert_eq!(
                most_moves.generate_valid_moves(!most_moves.next_move).len(),
                0
            );

            let mut most_moves = Board::from_fen(MOST_POSS_MOVES_FEN_B).unwrap();
            assert_eq!(
                most_moves.generate_valid_moves(most_moves.next_move).len(),
                0
            );
            assert_eq!(
                most_moves.generate_valid_moves(!most_moves.next_move).len(),
                216
            );
        }

        #[test]
        fn correct_move_count_3() {
            let mut board = Board::from_fen(MANY_POSS_MOVES_FEN_W).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 88);
            assert_eq!(board.generate_valid_moves(!board.next_move).len(), 90);
            let mut board = Board::from_fen(MANY_POSS_MOVES_FEN_B).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 90);
            assert_eq!(board.generate_valid_moves(!board.next_move).len(), 88);
        }

        #[test]
        fn correct_move_count_en_passant_w() {
            let mut board = Board::from_fen(EN_PASSANT_POS_W).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 30);
        }

        #[test]
        fn correct_move_count_en_passant_b() {
            let mut board = Board::from_fen(EN_PASSANT_POS_B).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 20);
        }

        #[test]
        fn correct_move_count_castle_queen_w() {
            let mut board = Board::from_fen(CASTLE_QUEEN_W).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 23)
        }

        #[test]
        fn correct_move_count_castle_king_w() {
            let mut board = Board::from_fen(CASTLE_KING_W).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 22)
        }

        #[test]
        fn correct_move_count_castle_queen_b() {
            let mut board = Board::from_fen(CASTLE_QUEEN_B).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 23)
        }

        #[test]
        fn correct_move_count_castle_king_b() {
            let mut board = Board::from_fen(CASTLE_KING_B).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 22)
        }

        #[test]
        fn correct_move_count_promotion() {
            let mut board = Board::from_fen(WHITE_PROMOTION).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 43);
        }

        #[test]
        fn enemy_pawn_in_front_of_king_prohibits_castling() {
            let mut board = Board::from_fen(PAWN_IN_FRONT_OF_KING_NO_CASTLE).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 36);
        }

        mod count {
            use std::ops::{Add, AddAssign};

            use crate::{
                test_fens::{POSITION_2, POSITION_3, POSITION_4_B, POSITION_4_W},
                Board, Move, START_BOARD_FEN,
            };

            #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
            struct CountedMoves {
                total: u64,
                captures: u32,
                en_passant: u32,
                castle: u32,
                promotions: u32,
            }

            impl CountedMoves {
                fn new(
                    total: u64,
                    captures: u32,
                    en_passant: u32,
                    castle: u32,
                    promotions: u32,
                ) -> Self {
                    CountedMoves {
                        total,
                        captures,
                        en_passant,
                        castle,
                        promotions,
                    }
                }
            }

            impl Add<CountedMoves> for CountedMoves {
                type Output = CountedMoves;

                fn add(self, rhs: CountedMoves) -> Self::Output {
                    CountedMoves {
                        total: self.total + rhs.total,
                        captures: self.captures + rhs.captures,
                        en_passant: self.en_passant + rhs.en_passant,
                        castle: self.castle + rhs.castle,
                        promotions: self.promotions + rhs.promotions,
                    }
                }
            }

            impl AddAssign for CountedMoves {
                fn add_assign(&mut self, rhs: Self) {
                    *self = *self + rhs;
                }
            }

            fn count_moves(board: &mut Board, depth: u32) -> CountedMoves {
                if depth == 0 {
                    assert!(false);
                }
                let mut count = CountedMoves::default();

                let moves = board.generate_valid_moves(board.next_move);
                for mve in moves {
                    if depth == 1 {
                        count.total += 1;
                        if board[mve.to].is_some() {
                            count.captures += 1;
                        }
                        match mve.typ {
                            crate::MoveType::Castle => count.castle += 1,
                            crate::MoveType::EnPassant => {
                                count.en_passant += 1;
                                count.captures += 1;
                            }
                            crate::MoveType::Promotion => count.promotions += 1,
                            _ => {}
                        }
                    } else {
                        board.with_temp_move(mve, |board| {
                            count += count_moves(board, depth - 1);
                        });
                    }
                }
                count
            }

            #[test]
            #[ignore = "slow test"]
            fn count_moves_from_start() {
                let mut board = Board::from_fen(START_BOARD_FEN).unwrap();

                let depth_to_count: &[(u32, CountedMoves)] = &[
                    (1, CountedMoves::new(20, 0, 0, 0, 0)),
                    (2, CountedMoves::new(400, 0, 0, 0, 0)),
                    (3, CountedMoves::new(8902, 34, 0, 0, 0)),
                    (4, CountedMoves::new(197281, 1576, 0, 0, 0)),
                    // (5, 4865609),
                    // (6, 119060324),
                    // (7, 3195901860),
                    // (8, 84998978956),
                    // (9, 2439530234167),
                ];

                for (depth, count) in depth_to_count {
                    assert_eq!(
                        count_moves(&mut board, *depth),
                        *count,
                        "Count from START for depth {depth} should be {count:?}"
                    );
                }
            }

            #[test]
            #[ignore = "slow test"]
            fn count_moves_from_position_2() {
                let mut board = Board::from_fen(POSITION_2).unwrap();

                let depth_to_count: &[(u32, CountedMoves)] = &[
                    (1, CountedMoves::new(48, 8, 0, 2, 0)),
                    (2, CountedMoves::new(2039, 351, 1, 91, 0)),
                    (3, CountedMoves::new(97862, 17102, 45, 3162, 0)),
                    (4, CountedMoves::new(4085603, 757163, 1929, 128013, 15172)),
                    // (5, 197281),
                    // (6, 8031647685),
                ];

                for (depth, count) in depth_to_count {
                    assert_eq!(
                        count_moves(&mut board, *depth),
                        *count,
                        "\nCount from POSITION_2 for depth {depth} should be {count:?}"
                    );
                }
            }

            #[test]
            #[ignore = "slow test"]
            fn count_moves_from_position_3() {
                let mut board = Board::from_fen(POSITION_3).unwrap();

                let depth_to_count: &[(u32, CountedMoves)] = &[
                    (1, CountedMoves::new(14, 1, 0, 0, 0)),
                    (2, CountedMoves::new(191, 14, 0, 0, 0)),
                    (3, CountedMoves::new(2812, 209, 2, 0, 0)),
                    (4, CountedMoves::new(43238, 3348, 123, 0, 0)),
                    // (5, 197281),
                    // (6, 8031647685),
                ];

                for (depth, count) in depth_to_count {
                    assert_eq!(
                        count_moves(&mut board, *depth),
                        *count,
                        "\nCount from POSITION_3 for depth {depth} should be {count:?}"
                    );
                }
            }

            #[test]
            #[ignore = "slow test"]
            fn count_moves_from_position_4() {
                let mut board_w = Board::from_fen(POSITION_4_W).unwrap();
                let mut board_b = Board::from_fen(POSITION_4_B).unwrap();

                let depth_to_count: &[(u32, CountedMoves)] = &[
                    (1, CountedMoves::new(6, 0, 0, 0, 0)),
                    (2, CountedMoves::new(264, 87, 0, 6, 48)),
                    (3, CountedMoves::new(9467, 1021, 4, 0, 120)),
                    (4, CountedMoves::new(422333, 131393, 0, 7795, 60032)),
                    // (5, 197281),
                    // (6, 8031647685),
                ];

                for (depth, count) in depth_to_count {
                    assert_eq!(
                        count_moves(&mut board_w, *depth),
                        *count,
                        "\nCount from POSITION_4_W for depth {depth} should be {count:?}"
                    );
                    assert_eq!(
                        count_moves(&mut board_b, *depth),
                        *count,
                        "\nCount from POSITION_4_B for depth {depth} should be {count:?}"
                    );
                }
            }
        }
    }
}
