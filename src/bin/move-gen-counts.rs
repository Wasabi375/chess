use std::time::Instant;

use chess::engine::ella::EllaChess;
use chess::engine::Engine;
use chess::{Board, START_BOARD_FEN};

fn main() {
    search_stats(1);
    search_stats(3);
    search_stats(5);
    search_stats(6);
}

fn search_stats(depth: u32) {
    let board = Board::from_fen(START_BOARD_FEN).unwrap();

    let mut engine = EllaChess::new_from_board(board);

    let start = Instant::now();
    engine.search_to_depth(depth).unwrap();
    let elapsed = start.elapsed();

    let stats = engine.stats().unwrap();

    println!(
        "Stats after searching to depth {depth} from start pos:\ntime: {elapsed:?}\n{stats:#?}"
    );
}
