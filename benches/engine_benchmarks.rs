use chess::{Board, START_BOARD_FEN};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

mod ella {
    use chess::{
        engine::{ella::EllaChess, Engine},
        Board,
    };

    pub fn search_to_depth_depth(board: Board, depth: u32) {
        let mut ella = EllaChess::new_from_board(board);
        if let Err(e) = ella.search_to_depth(depth) {
            panic!("{e}");
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    const SEARCH_DEPTHS: &[u32] = &[1, 5];

    let board = Board::from_fen(START_BOARD_FEN).unwrap();

    let mut ella_group = c.benchmark_group("ella::search_to_depth");
    ella_group.sample_size(10);
    for &depth in SEARCH_DEPTHS {
        ella_group.bench_with_input(
            BenchmarkId::from_parameter(depth),
            &(depth, &board),
            |b, (depth, board)| {
                b.iter(|| ella::search_to_depth_depth((*board).clone(), *depth));
            },
        );
    }
    ella_group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
