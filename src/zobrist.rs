use crate::{Color, Piece, PiecePositions};
use core::fmt;
use lazy_static::lazy_static;
use rand::Rng;
use std::{
    collections::{hash_map::DefaultHasher, HashSet},
    hash::{Hash, Hasher},
    mem::{self, MaybeUninit},
    ops::BitXorAssign,
};

lazy_static! {
    pub static ref ZOBRIST_HASHER: ZobristHasher<u64> = {
        #[cfg(test)]
        let mut rng = {
            use rand::rngs::StdRng;
            use rand::SeedableRng;
            StdRng::seed_from_u64(123456789)
        };
        #[cfg(not(test))]
        let mut rng = rand::thread_rng();

        ZobristHasher::new_random(&mut rng)
    };
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ZobristHasherKingMoves<H> {
    pub castle_king: H,
    pub castle_queen: H,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct ZobristHasher<H> {
    pub black_move: H,
    pub white_king: ZobristHasherKingMoves<H>,
    pub black_king: ZobristHasherKingMoves<H>,
    pub en_passant_col: [H; 8],
    pub piece_hash: [H; 64 * 12],
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
    pub fn king_move(&self, color: Color) -> ZobristHasherKingMoves<H> {
        match color {
            Color::White => &self.white_king,
            Color::Black => &self.black_king,
        }
        .clone()
    }

    pub fn piece_hash(&self, pos: impl Into<usize>, piece: Piece) -> H {
        self.piece_hash[pos.into() * piece.zobrist_index()].clone()
    }

    pub fn en_passant_hash(&self, pos: impl Into<usize>) -> H {
        self.en_passant_col[pos.into() % 8].clone()
    }
}

impl ZobristHasher<u64> {
    pub fn new_random(rng: &mut impl Rng) -> Self {
        let mut used_hashes = HashSet::new();
        let mut rng = || loop {
            let result = rng.next_u64();
            if used_hashes.insert(result) {
                break result;
            }
        };
        let en_passant_row = {
            let mut data: [MaybeUninit<u64>; 8] = unsafe { MaybeUninit::uninit().assume_init() };
            for elem in &mut data {
                elem.write(rng());
            }
            unsafe { mem::transmute(data) }
        };
        let piece_hash = {
            let mut data: [MaybeUninit<u64>; 64 * 12] =
                unsafe { MaybeUninit::uninit().assume_init() };
            for elem in &mut data {
                elem.write(rng());
            }
            unsafe { mem::transmute(data) }
        };

        Self {
            black_move: rng(),
            en_passant_col: en_passant_row,
            white_king: ZobristHasherKingMoves {
                castle_king: rng(),
                castle_queen: rng(),
            },
            black_king: ZobristHasherKingMoves {
                castle_king: rng(),
                castle_queen: rng(),
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
