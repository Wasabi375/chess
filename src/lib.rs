use core::fmt;
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    fmt::Display,
    hash::{Hash, Hasher},
    mem::{self, MaybeUninit},
    num::NonZeroU8,
    ops::{BitXorAssign, Index, IndexMut, Not},
};

use rand::{thread_rng, Rng};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promote_to: Option<Piece>,
    pub typ: MoveType,
}

impl Move {
    pub fn new(from: u8, to: u8) -> Self {
        Move {
            from,
            to,
            promote_to: None,
            typ: MoveType::Normal,
        }
    }

    pub fn en_passant(from: u8, to: u8) -> Self {
        Move {
            from,
            to,
            promote_to: None,
            typ: MoveType::EnPassant,
        }
    }

    pub fn castle(from: u8, to: u8) -> Self {
        Move {
            from,
            to,
            promote_to: None,
            typ: MoveType::Castle,
        }
    }

    pub fn promotion(from: u8, to: u8, target: Piece) -> Self {
        Move {
            from,
            to,
            promote_to: Some(target),
            typ: MoveType::Promotion,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiecePositions {
    #[allow(dead_code)]
    pub color: Color,
    pub king: u8,
    pub castle_king: bool,
    pub castle_queen: bool,
}

#[derive(Clone, Hash)]
pub struct ZobristHasherKingMoves<H> {
    castle_king: H,
    castle_queen: H,
}

#[derive(Clone, Hash)]
pub struct ZobristHasher<H> {
    black_move: H,
    white_king: ZobristHasherKingMoves<H>,
    black_king: ZobristHasherKingMoves<H>,
    en_passant_col: [H; 8],
    piece_hash: [H; 64 * 12],
}

impl<H: Hash> fmt::Debug for ZobristHasher<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish();
        f.debug_struct("ZobristHasher")
            .field("hash", &hash)
            .finish_non_exhaustive()
    }
}

impl<H: Clone> ZobristHasher<H> {
    fn king_move(&self, color: Color) -> ZobristHasherKingMoves<H> {
        match color {
            Color::White => &self.white_king,
            Color::Black => &self.black_king,
        }
        .clone()
    }

    fn piece_hash(&self, pos: impl Into<usize>, piece: Piece) -> H {
        self.piece_hash[pos.into() * piece.zobrist_index()].clone()
    }

    fn en_passant_hash(&self, pos: impl Into<usize>) -> H {
        self.en_passant_col[pos.into() % 8].clone()
    }
}

impl ZobristHasher<u64> {
    fn new_random(rng: &mut impl Rng) -> Self {
        let en_passant_row = {
            let mut data: [MaybeUninit<u64>; 8] = unsafe { MaybeUninit::uninit().assume_init() };
            for elem in &mut data {
                elem.write(rng.next_u64());
            }
            unsafe { mem::transmute(data) }
        };
        let piece_hash = {
            let mut data: [MaybeUninit<u64>; 64 * 12] =
                unsafe { MaybeUninit::uninit().assume_init() };
            for elem in &mut data {
                elem.write(rng.next_u64());
            }
            unsafe { mem::transmute(data) }
        };

        Self {
            black_move: rng.next_u64(),
            en_passant_col: en_passant_row,
            white_king: ZobristHasherKingMoves {
                castle_king: rng.next_u64(),
                castle_queen: rng.next_u64(),
            },
            black_king: ZobristHasherKingMoves {
                castle_king: rng.next_u64(),
                castle_queen: rng.next_u64(),
            },
            piece_hash,
        }
    }
}

impl<H: BitXorAssign + Clone + Default> ZobristHasher<H> {
    pub fn hash(
        &self,
        field: &[Option<Piece>; 64],
        black_pos: &PiecePositions,
        white_pos: &PiecePositions,
        en_passant: Option<u8>,
        next_move: Color,
    ) -> H {
        let mut hash = H::default();
        if next_move == Color::Black {
            hash ^= self.black_move.clone();
        }
        if let Some(en_passant) = en_passant {
            hash ^= self.en_passant_hash(en_passant);
        }
        if black_pos.castle_queen {
            hash ^= self.black_king.castle_queen.clone();
        }
        if black_pos.castle_king {
            hash ^= self.black_king.castle_king.clone();
        }
        if white_pos.castle_queen {
            hash ^= self.white_king.castle_queen.clone();
        }
        if white_pos.castle_king {
            hash ^= self.white_king.castle_king.clone();
        }

        for (i, piece) in field.iter().enumerate() {
            if let Some(piece) = *piece {
                hash ^= self.piece_hash(i, piece);
            }
        }

        hash
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Board {
    fields: [Option<Piece>; 64],

    pub white_pieces: PiecePositions,
    pub black_pieces: PiecePositions,

    pub next_move: Color,

    pub en_passant_square: Option<u8>,

    pub half_moves_since_capture: u8,
    pub full_move_count: u32,

    pub previous_positions: HashMap<u64, u8>,

    pub zobrist_hash: u64,
    pub zobrist_hasher: ZobristHasher<u64>,
}

impl PartialEq<Board> for Board {
    fn eq(&self, other: &Board) -> bool {
        // those should be all fields, except for zobrist hash and hasher
        self.fields == other.fields
            && self.white_pieces == other.white_pieces
            && self.black_pieces == other.black_pieces
            && self.next_move == other.next_move
            && self.en_passant_square == other.en_passant_square
            && self.half_moves_since_capture == other.half_moves_since_capture
            && self.full_move_count == other.full_move_count
    }
}
impl Eq for Board {}

impl Board {
    pub fn empty() -> Self {
        Board {
            fields: [None; 64],
            white_pieces: PiecePositions {
                color: Color::White,
                king: 0,
                castle_king: true,
                castle_queen: true,
            },
            black_pieces: PiecePositions {
                color: Color::Black,
                king: 0,
                castle_king: true,
                castle_queen: true,
            },
            next_move: Color::White,
            en_passant_square: None,
            half_moves_since_capture: 0,
            full_move_count: 0,
            previous_positions: HashMap::new(),
            zobrist_hash: 0,
            zobrist_hasher: ZobristHasher::new_random(&mut thread_rng()),
        }
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut fields = [None; 64];

        let mut idx = 0;

        let mut iter = fen.chars();

        let mut white_king = 0;
        let mut black_king = 0;

        // parse piece positions
        for char in &mut iter {
            if idx == 64 {
                if char != ' ' {
                    return Err("Expected ' ' after pieces".to_string());
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
                        return Err("'/' must come between 2 lines".to_string());
                    }
                }
                _ => return Err(format!("unexpected '{char}' instead of piece")),
            }
        }

        let next_move = match iter.next() {
            Some('b') => Color::Black,
            Some('w') => Color::White,
            _ => return Err("Expected either 'w' or 'b' to move".to_string()),
        };

        if iter.next() != Some(' ') {
            return Err("Expected ' ' after move color".to_string());
        }

        let mut white_castle_king: bool = false;
        let mut white_castle_queen: bool = false;
        let mut black_castle_king: bool = false;
        let mut black_castle_queen: bool = false;

        loop {
            match iter.next() {
                Some('-') => {
                    if iter.next() != Some(' ') {
                        return Err("Expected ' ' after '-' for castling".to_string());
                    }
                    break;
                }
                Some('k') => black_castle_king = true,
                Some('q') => black_castle_queen = true,
                Some('K') => white_castle_king = true,
                Some('Q') => white_castle_queen = true,
                Some(' ') => break,
                c => return Err(format!("Expected castling availability but got {c:?}")),
            }
        }

        let en_passant_square = {
            let col = iter.next();
            if col.is_none() {
                return Err("Expected en-passant".to_string());
            }
            let col = col.unwrap();
            if col != '-' {
                if !('a'..='h').contains(&col) {
                    return Err(format!("Expected 'a-h' for en-passant, got '{col}'"));
                }
                let col = col as u32 - 'a' as u32;

                let row = iter.next();
                if row.is_none() {
                    return Err("Expected en-passant row".to_string());
                }
                let row = row.unwrap();
                if !('1'..='8').contains(&row) {
                    return Err("Expected '1-8' for en-passant row".to_string());
                }
                let row = row as u32 - '1' as u32;

                let idx = col * 8 + row;
                assert!(idx < 64);

                Some(((7 - row) * 8 + col) as u8)
            } else {
                None
            }
        };

        if iter.next() != Some(' ') {
            return Err("Expected ' ' after en-passant".to_string());
        }

        let clocks_str: String = iter.collect();
        let mut clocks = clocks_str.trim_end().split(' ');

        let half_moves_since_capture: u8 = if let Some(half_moves) = clocks.next() {
            half_moves
                .parse()
                .map_err(|_| "Could not parse half-move-count".to_string())?
        } else {
            return Err("expected half move count".to_string());
        };

        let full_move_count: u32 = if let Some(full_moves) = clocks.next() {
            full_moves
                .parse()
                .map_err(|_| "Could not parse move-count".to_string())?
        } else {
            return Err("Expected move count".to_string());
        };

        if clocks.next().is_some() {
            return Err("Expected end of FEN".to_string());
        }

        let white_pieces = PiecePositions {
            color: Color::White,
            king: white_king as u8,
            castle_king: white_castle_king,
            castle_queen: white_castle_queen,
        };
        let black_pieces = PiecePositions {
            color: Color::Black,
            king: black_king as u8,
            castle_king: black_castle_king,
            castle_queen: black_castle_queen,
        };

        let zobrist_hasher = ZobristHasher::new_random(&mut thread_rng());
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
            half_moves_since_capture,
            full_move_count,
            previous_positions,
            zobrist_hash,
            zobrist_hasher,
        };

        Ok(board)
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

        fen.push_str(&self.half_moves_since_capture.to_string());
        fen.push(' ');
        fen.push_str(&self.full_move_count.to_string());

        fen
    }

    pub fn draw_by_repetition_or_50_moves(&self) -> bool {
        if self.previous_positions[&self.zobrist_hash] >= 3 {
            return true;
        }
        if self.half_moves_since_capture >= 100 {
            return true;
        }
        false
    }

    pub fn winner(&self) -> Option<Option<Color>> {
        let valid_moves = self.generate_valid_moves(self.next_move);
        if valid_moves.is_empty() {
            let attack_moves = self.generate_attacking_moves(!self.next_move);
            let king_pos = self.piece_positions(self.next_move).king;
            if attack_moves.iter().any(|m| m.to == king_pos) {
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
    pub fn is_valid_move(&self, mve: Move) -> bool {
        if let Some(piece) = self[mve.from] {
            let color = piece.color();

            let mut clone = self.clone();
            // NOTE: play asserts the color is matching
            clone.next_move = color;
            clone.play_move(mve);
            !clone.is_in_check(color)
        } else {
            false
        }
    }

    fn generate_valid_moves_ignore_checks(&self, color: Color) -> Vec<Move> {
        let mut moves = Vec::with_capacity(218);
        for pos in 0..64u8 {
            if let Some(piece) = self[pos] {
                if piece.color() == color {
                    moves = self.generate_vaild_moves_for_piece_int(pos, piece, true, moves);
                }
            }
        }
        moves
    }

    pub fn generate_valid_moves(&self, color: Color) -> Vec<Move> {
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
                    moves = self.generate_vaild_moves_for_piece_int(pos, piece, false, moves);
                }
            }
        }
        moves
    }

    pub fn generate_valid_moves_for_piece(&self, piece_at: u8) -> Vec<Move> {
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
        let mut moves = self.generate_vaild_moves_for_piece_int(piece_at, piece, true, moves);
        moves.retain(|m| self.is_valid_move(*m));
        moves
    }

    fn generate_vaild_moves_for_piece_int(
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
            let castle_row = match color {
                Color::White => 7,
                Color::Black => 0,
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

            let enemy_moves = self.generate_attacking_moves(!color);

            if castle_king
                && !enemy_moves
                    .iter()
                    .any(|m| m.to == bidx(5, castle_row) || m.to == bidx(6, castle_row))
            {
                moves.push(Move::castle(piece_at, bidx(6, castle_row)));
            }
            if castle_queen
                && !enemy_moves.iter().any(|m| {
                    m.to == bidx(1, castle_row)
                        || m.to == bidx(2, castle_row)
                        || m.to == bidx(3, castle_row)
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
        /// plays a given move. This assumes that the move is valid
        fn play_move_int(slf: &mut Board, mve: Move) {
            debug_assert!(slf[mve.from].is_some());
            debug_assert_eq!(slf[mve.from].unwrap().color(), slf.next_move);

            // toggle color for next move
            slf.next_move = !slf.next_move;
            slf.zobrist_hash ^= slf.zobrist_hasher.black_move;

            if slf.next_move == Color::White {
                slf.full_move_count += 1;
            }

            if mve.typ == MoveType::EnPassant {
                assert!(slf.en_passant_square.is_some());
                let en_passant = slf.en_passant_square.unwrap();

                slf.en_passant_square = None;
                slf.zobrist_hash ^= slf.zobrist_hasher.en_passant_hash(en_passant);

                slf[mve.to] = slf[mve.from];
                slf[mve.from] = None;

                let pawn = slf[mve.to].unwrap();
                debug_assert_eq!(pawn.typ(), PieceType::Pawn);
                slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(mve.from, pawn);
                slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(mve.to, pawn);

                let en_passant_col = en_passant % 8;
                let en_passant_row = mve.from / 8;
                let en_passant_idx = en_passant_col + en_passant_row * 8;

                if let Some(captured) = slf[en_passant_idx] {
                    debug_assert_eq!(captured.typ(), PieceType::Pawn);
                    slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(en_passant_idx, captured);
                }
                slf[en_passant_idx] = None;

                println!("EnPassant: {mve:?}, col: {en_passant_col}, row: {en_passant_row}, idx: {en_passant_idx}");

                slf.half_moves_since_capture = 0;

                return;
            }

            if mve.typ == MoveType::Castle {
                let castle_king = mve.to == (mve.from + 2);
                let (rook_from, rook_to) = if castle_king {
                    (mve.to + 1, mve.to - 1)
                } else {
                    (mve.to - 2, mve.to + 1)
                };

                debug_assert!(slf[rook_from].is_some());
                debug_assert_eq!(slf[rook_from].unwrap().typ(), PieceType::Rook);

                let king = slf[mve.from].unwrap();
                let rook = slf[rook_from].unwrap();

                // move king
                slf[mve.to] = slf[mve.from];
                slf[mve.from] = None;
                slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(mve.from, king);
                slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(mve.to, king);

                // move rook
                slf[rook_to] = slf[rook_from];
                slf[rook_from] = None;
                slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(rook_from, rook);
                slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(rook_to, rook);

                if let Some(en_passant_square) = slf.en_passant_square {
                    slf.zobrist_hash ^= slf.zobrist_hasher.en_passant_hash(en_passant_square);
                }
                slf.en_passant_square = None;

                let zobrist_king_hasher = slf.zobrist_hasher.king_move(king.color());
                if slf.piece_positions(king.color()).castle_queen {
                    slf.zobrist_hash ^= zobrist_king_hasher.castle_queen;
                }
                if slf.piece_positions(king.color()).castle_king {
                    slf.zobrist_hash ^= zobrist_king_hasher.castle_king;
                }
                slf.piece_positions_mut(king.color()).castle_queen = false;
                slf.piece_positions_mut(king.color()).castle_king = false;
                slf.piece_positions_mut(king.color()).king = mve.to;

                slf.half_moves_since_capture += 1;
                return;
            }

            if let Some(caputred) = slf[mve.to] {
                slf.half_moves_since_capture = 0;
                slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(mve.to, caputred);
            } else {
                slf.half_moves_since_capture += 1;
            }

            if mve.typ == MoveType::Promotion {
                assert!(mve.promote_to.is_some());
                slf[mve.to] = mve.promote_to;
            } else {
                slf[mve.to] = slf[mve.from];
            }
            let old_piece = slf[mve.from].unwrap();
            slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(mve.from, old_piece);
            slf[mve.from] = None;

            if old_piece.typ() == PieceType::Pawn {
                slf.half_moves_since_capture = 0;
            }

            // piece might be different from old pice, because of possible promotion
            let piece = slf[mve.to].unwrap();
            slf.zobrist_hash ^= slf.zobrist_hasher.piece_hash(mve.to, piece);

            let zobrist_king_hasher = slf.zobrist_hasher.king_move(piece.color());
            match piece.typ() {
                PieceType::King => {
                    slf.piece_positions_mut(piece.color()).king = mve.to;
                    if slf.piece_positions(piece.color()).castle_queen {
                        slf.zobrist_hash ^= zobrist_king_hasher.castle_queen;
                    }
                    slf.piece_positions_mut(piece.color()).castle_queen = false;
                    if slf.piece_positions(piece.color()).castle_king {
                        slf.zobrist_hash ^= zobrist_king_hasher.castle_king;
                    }
                    slf.piece_positions_mut(piece.color()).castle_king = false;
                }
                PieceType::Rook => {
                    let col = mve.from % 8;
                    let row = mve.from / 8;
                    let start_row = if piece.color() == Color::White { 7 } else { 0 };
                    if row == start_row {
                        if col == 0 {
                            if slf.piece_positions(piece.color()).castle_queen {
                                slf.zobrist_hash ^= zobrist_king_hasher.castle_queen;
                            }
                            slf.piece_positions_mut(piece.color()).castle_queen = false;
                        }
                        if col == 7 {
                            if slf.piece_positions(piece.color()).castle_king {
                                slf.zobrist_hash ^= zobrist_king_hasher.castle_king;
                            }
                            slf.piece_positions_mut(piece.color()).castle_king = false;
                        }
                    }
                }
                _ => {}
            }

            if let Some(old_en_passant) = slf.en_passant_square {
                slf.zobrist_hash ^= slf.zobrist_hasher.en_passant_hash(old_en_passant);
            }

            if piece.typ() == PieceType::Pawn && mve.to.abs_diff(mve.from) == 16 {
                slf.en_passant_square = Some(mve.to.min(mve.from) + 8);
                slf.zobrist_hash ^= slf.zobrist_hasher.en_passant_hash(mve.from);
            } else {
                slf.en_passant_square = None;
            }
        }
        play_move_int(self, mve);

        let count = self
            .previous_positions
            .entry(self.zobrist_hash)
            .or_insert(0);
        *count += 1;
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

#[cfg(test)]
mod test {
    use crate::{Board, Color, Piece, PieceType, START_BOARD_FEN};
    use std::collections::HashSet;

    const MOST_POSS_MOVES_FEN: &str = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNNK1B1 w - - 0 0";
    const MOST_POSS_MOVES_FEN_B: &str = "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNNK1B1 b - - 0 0";
    const MANY_POSS_MOVES_FEN_W: &str = "r6R/2pbpBk1/1p1B1N2/6q1/4Q3/2nn1p2/1PK1NbP1/R6r w - - 0 0";
    const MANY_POSS_MOVES_FEN_B: &str = "r6R/2pbpBk1/1p1B1N2/6q1/4Q3/2nn1p2/1PK1NbP1/R6r b - - 0 0";
    const EN_PASSANT_POS_W: &str = "rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 0";
    const EN_PASSANT_POS_W2: &str = "rnbqkbnr/pppp2pp/4p3/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 0";
    const EN_PASSANT_POS_B: &str = "rnbqkbnr/ppppp1pp/8/4P3/5pP1/8/PPPP1P1P/RNBQKBNR b KQkq g3 0 0";
    const CASTLE_QUEEN_W: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 0";
    const CASTLE_KING_W: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 0";
    const CASTLE_KING_B: &str = "rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 0";
    const CASTLE_QUEEN_B: &str = "r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 0";
    const WHITE_PROMOTION: &str = "rnbqkb1r/pppp2Pp/2n5/8/8/8/PPP1PPPP/RNBQKBNR w KQkq - 1 4";

    const ALL_TEST_FENS: &[&str] = &[
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
    ];

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
            println!("FEN: {generated_fen}");
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

    mod moves {
        use crate::{
            test::{
                CASTLE_KING_B, CASTLE_KING_W, CASTLE_QUEEN_B, CASTLE_QUEEN_W, EN_PASSANT_POS_B,
            },
            Board, Move,
        };

        use super::{EN_PASSANT_POS_W, EN_PASSANT_POS_W2};

        #[test]
        fn en_passant_white() {
            let mut board = Board::from_fen(EN_PASSANT_POS_W).unwrap();
            board.play_move(Move::en_passant(28, 19));
            let expected =
                Board::from_fen("rnbqkbnr/ppp2ppp/3Pp3/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 0")
                    .unwrap();
            assert_eq!(board, expected);
        }

        #[test]
        fn en_passant_white_2() {
            let mut board = Board::from_fen(EN_PASSANT_POS_W2).unwrap();
            board.play_move(Move::en_passant(28, 21));
            let expected =
                Board::from_fen("rnbqkbnr/pppp2pp/4pP2/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 0")
                    .unwrap();
            assert_eq!(board, expected);
        }

        #[test]
        fn en_passant_black() {
            let mut board = Board::from_fen(EN_PASSANT_POS_B).unwrap();
            board.play_move(Move::en_passant(37, 46));
            let expected =
                Board::from_fen("rnbqkbnr/ppppp1pp/8/4P3/8/6p1/PPPP1P1P/RNBQKBNR w KQkq - 0 1")
                    .unwrap();
            assert_eq!(board, expected);
        }

        #[test]
        fn castle_queen_white() {
            let mut board = Board::from_fen(CASTLE_QUEEN_W).unwrap();
            board.play_move(Move::castle(60, 58));
            let expected =
                Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/2KR1BNR b kq - 1 0").unwrap();
            assert_eq!(board, expected);
        }

        #[test]
        fn castle_king_white() {
            let mut board = Board::from_fen(CASTLE_KING_W).unwrap();
            board.play_move(Move::castle(60, 62));
            let expected =
                Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 b kq - 1 0").unwrap();
            assert_eq!(board, expected);
        }

        #[test]
        fn castle_queen_black() {
            let mut board = Board::from_fen(CASTLE_QUEEN_B).unwrap();
            board.play_move(Move::castle(4, 2));
            let expected =
                Board::from_fen("2kr1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 1 1").unwrap();
            assert_eq!(board, expected);
        }

        #[test]
        fn castle_king_black() {
            let mut board = Board::from_fen(CASTLE_KING_B).unwrap();
            board.play_move(Move::castle(4, 6));
            let expected =
                Board::from_fen("rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 1 1").unwrap();
            assert_eq!(board, expected);
        }
    }

    #[test]
    fn zobrist_hash_valid_after_move() {
        for fen in ALL_TEST_FENS {
            let board = Board::from_fen(fen).unwrap();
            for mve in board.generate_valid_moves(board.next_move) {
                let mut clone = board.clone();
                clone.play_move(mve);
                assert_eq!(
                    clone.zobrist_hash,
                    clone.calculate_zobrist_hash(),
                    "\nZobrist hash failed for move {mve:?} at fen \"{fen}\""
                );
            }
        }
    }

    mod move_gen {
        use crate::{
            test::{
                CASTLE_KING_B, CASTLE_KING_W, CASTLE_QUEEN_B, EN_PASSANT_POS_B, EN_PASSANT_POS_W,
                MANY_POSS_MOVES_FEN_B, MANY_POSS_MOVES_FEN_W, MOST_POSS_MOVES_FEN,
                MOST_POSS_MOVES_FEN_B,
            },
            Board, START_BOARD_FEN,
        };

        use super::{CASTLE_QUEEN_W, WHITE_PROMOTION};

        #[test]
        fn correct_move_count_1() {
            let start = Board::from_fen(START_BOARD_FEN).unwrap();
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
            let most_moves = Board::from_fen(MOST_POSS_MOVES_FEN).unwrap();
            assert_eq!(
                most_moves.generate_valid_moves(most_moves.next_move).len(),
                216
            );
            assert_eq!(
                most_moves.generate_valid_moves(!most_moves.next_move).len(),
                0
            );

            let most_moves = Board::from_fen(MOST_POSS_MOVES_FEN_B).unwrap();
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
            let board = Board::from_fen(MANY_POSS_MOVES_FEN_W).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 88);
            assert_eq!(board.generate_valid_moves(!board.next_move).len(), 90);
            let board = Board::from_fen(MANY_POSS_MOVES_FEN_B).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 90);
            assert_eq!(board.generate_valid_moves(!board.next_move).len(), 88);
        }

        #[test]
        fn correct_move_count_en_passant_w() {
            let board = Board::from_fen(EN_PASSANT_POS_W).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 30);
        }

        #[test]
        fn correct_move_count_en_passant_b() {
            let board = Board::from_fen(EN_PASSANT_POS_B).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 20);
        }

        #[test]
        fn correct_move_count_castle_queen_w() {
            let board = Board::from_fen(CASTLE_QUEEN_W).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 23)
        }

        #[test]
        fn correct_move_count_castle_king_w() {
            let board = Board::from_fen(CASTLE_KING_W).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 22)
        }

        #[test]
        fn correct_move_count_castle_queen_b() {
            let board = Board::from_fen(CASTLE_QUEEN_B).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 23)
        }

        #[test]
        fn correct_move_count_castle_king_b() {
            let board = Board::from_fen(CASTLE_KING_B).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 22)
        }

        #[test]
        fn correct_move_count_promotion() {
            let board = Board::from_fen(WHITE_PROMOTION).unwrap();
            assert_eq!(board.generate_valid_moves(board.next_move).len(), 43);
        }
    }
}
