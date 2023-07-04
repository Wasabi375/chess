mod assets;
mod square;

use chess::{Board, Move, MoveType, Piece, PieceType, START_BOARD_FEN};
use iced::{
    clipboard,
    widget::{button, container, image, text},
    Alignment, Application, Command, Element, Length, Settings,
};
use iced_native::{
    column,
    image::Handle,
    widget::{Column, Row},
};
use square::square;
use std::collections::HashMap;

fn main() -> iced::Result {
    Game::run(Settings {
        flags: GameInitialization {
            fen: Some(START_BOARD_FEN.to_string()),
        },
        ..Settings::default()
    })
}

#[derive(Debug, Default)]
struct GameInitialization {
    fen: Option<String>,
}

struct Game {
    board: Board,
    piece_images: HashMap<Piece, String>,
    state: GameState,
    highlighted: Vec<u8>,
}

#[derive(Default)]
enum GameState {
    #[default]
    WaitingForUser,
    PieceSelected {
        target_square: u8,
        valid_moves: Vec<Move>,
    },
    Promoting {
        promotion_square: u8,
        move_squrae: u8,
        valid_moves: Vec<Move>,
    },
}

impl GameState {
    fn handle_square_clicked(&mut self, square: u8, board: &Board) -> Option<Message> {
        let x = square % 8;
        let y = square / 8;
        println!("Clicked {square}: {x}-{y}: {:?}", board[square]);

        match self {
            GameState::WaitingForUser => {
                let target = board[square];
                if target.is_some() {
                    let valid_moves = board.generate_valid_moves_for_piece(square);
                    if !valid_moves.is_empty() {
                        *self = GameState::PieceSelected {
                            target_square: square,
                            valid_moves,
                        };
                    }
                }
                None
            }
            GameState::PieceSelected {
                target_square: original_square,
                valid_moves,
            } => {
                let selected = valid_moves
                    .iter()
                    .filter(|m| m.to == square && m.from == *original_square);
                if let Some(mve) = selected.clone().next() {
                    if mve.typ == MoveType::Promotion {
                        assert_eq!(selected.count(), 4);

                        *self = GameState::Promoting {
                            promotion_square: square,
                            move_squrae: mve.from,
                            // TODO clone should not be necessary here, since valid_moves is never modified
                            //  this might be fixed by rust in the future or I could use Rc instead
                            valid_moves: valid_moves.clone(),
                        };
                        None
                    } else {
                        Some(Message::Move(*mve))
                    }
                } else {
                    *self = GameState::WaitingForUser;
                    None
                }
            }
            GameState::Promoting {
                promotion_square,
                move_squrae,
                valid_moves,
            } => {
                if square == *promotion_square {
                    // promote to quene
                    let promo_move = valid_moves
                        .iter()
                        .find(|m| {
                            m.from == *move_squrae
                                && m.to == *promotion_square
                                && m.promote_to.map(|p| p.typ()) == Some(PieceType::Queen)
                        })
                        .expect("State is Promoting, but could not find promotion move");
                    Some(Message::Move(*promo_move))
                } else if square == *move_squrae {
                    // do nothing
                    None
                } else {
                    // promotion cancled
                    *self = GameState::PieceSelected {
                        target_square: *move_squrae,
                        // TODO clone should not be necessary here, since valid_moves is never modified
                        //  this might be fixed by rust in the future or I could use Rc instead
                        valid_moves: valid_moves.clone(),
                    };
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
    /// used to run update without specific message
    None,
    SquareClicked(u8),
    Move(Move),
    GenerateFen,
    LoadFen,
    RestartGame(Box<Board>),
    CountMoves,
}

impl Application for Game {
    type Executor = iced::executor::Default;

    type Message = Message;

    type Theme = iced::Theme;

    type Flags = GameInitialization;

    fn new(flags: GameInitialization) -> (Self, Command<Self::Message>) {
        let board = if let Some(fen) = flags.fen {
            Board::from_fen(&fen).expect("fen string is invalid")
        } else {
            Board::empty()
        };

        let mut game = Game {
            board,
            piece_images: assets::generate_piece_image_mapping(assets::ImageResolution::Px128),
            state: Default::default(),
            highlighted: Vec::new(),
        };
        let _ = game.update(Message::None);
        (game, Command::none())
    }

    fn title(&self) -> String {
        "Chess".to_string()
    }

    fn update(&mut self, message: Self::Message) -> Command<Message> {
        let new_message = match message {
            Message::None => None,
            Message::SquareClicked(idx) => self.state.handle_square_clicked(idx, &self.board),
            Message::Move(m) => {
                self.board.play_move(m);
                self.state = GameState::WaitingForUser;
                None
            }
            Message::GenerateFen => {
                let fen = self.board.generate_fen();
                println!("Fen generated: {fen}");
                return clipboard::write(fen);
            }
            Message::LoadFen => {
                return clipboard::read(|fen| {
                    if let Some(fen) = fen {
                        println!("Loading fen \"{fen}\"");
                        let board = Board::from_fen(&fen);
                        match board {
                            Ok(board) => Message::RestartGame(Box::new(board)),
                            Err(e) => {
                                println!("Could not load fen \"{fen}\": {e:?}");
                                Message::None
                            }
                        }
                    } else {
                        println!("Empty clipboard!");
                        Message::None
                    }
                })
            }
            Message::RestartGame(board) => {
                self.board = *board;
                self.state = Default::default();
                self.highlighted.clear();
                None
            }
            Message::CountMoves => {
                println!(
                    "There are {} possible moves for {}",
                    self.board.generate_valid_moves(self.board.next_move).len(),
                    self.board.next_move
                );
                None
            }
        };

        self.highlighted = self.board.en_passant_square.iter().copied().collect();

        if let Some(new_message) = new_message {
            self.update(new_message)
        } else {
            Command::none()
        }
    }

    fn view(&self) -> Element<'_, Self::Message> {
        let square_length = 80.0;

        let board = {
            let mut rows = Vec::<Element<Message>>::new();
            for row_index in 0..8 {
                let mut row = Vec::<Element<_>>::new();
                for column_index in 0..8 {
                    let bindex = (row_index * 8 + column_index) as u8;
                    let piece = self.board[bindex as usize];
                    let image = piece.map(|p| {
                        let handle = Handle::from_path(&self.piece_images[&p]);
                        image(handle)
                    });
                    let (selected, valid_move) = match &self.state {
                        GameState::WaitingForUser => (false, false),
                        GameState::PieceSelected {
                            target_square,
                            valid_moves,
                        } => {
                            let selected = bindex == *target_square;
                            let valid = !selected && valid_moves.iter().any(|m| m.to == bindex);
                            (selected, valid)
                        }
                        GameState::Promoting {
                            promotion_square,
                            move_squrae,
                            valid_moves: _,
                        } => {
                            let selected = bindex == *move_squrae;
                            let valid = *promotion_square == bindex;
                            (selected, valid)
                        }
                    };

                    row.push(
                        button(square(
                            (row_index + column_index) % 2 == 0,
                            square_length,
                            image,
                            selected,
                            valid_move,
                            self.highlighted.contains(&bindex),
                        ))
                        .on_press(Message::SquareClicked(bindex))
                        .padding(0.0)
                        .into(),
                    );
                }

                rows.push(Row::with_children(row).into());
            }
            Column::with_children(rows)
        }
        .padding(0)
        .spacing(0)
        .align_items(Alignment::Center);

        let mut row_entries = Vec::<Element<_>>::new();

        row_entries.push(board.into());

        if let Some(winner) = self.board.winner() {
            if let Some(winner) = winner {
                row_entries.push(text(format!("Winner {winner}")).into());
            } else {
                row_entries.push(text("Draw").into());
            }
        }

        if let GameState::Promoting {
            promotion_square,
            move_squrae,
            valid_moves,
        } = &self.state
        {
            let color = self.board.next_move;
            let promote_button = |piece_type| {
                let piece = Piece::new(piece_type, color);
                let mve = valid_moves
                    .iter()
                    .find(|m| {
                        m.to == *promotion_square
                            && m.from == *move_squrae
                            && m.promote_to == Some(piece)
                    })
                    .expect("Promotion not found");
                let handle = Handle::from_path(&self.piece_images[&piece]);
                button(image(handle)).on_press(Message::Move(*mve))
            };

            row_entries.push(
                column![
                    promote_button(PieceType::Queen),
                    promote_button(PieceType::Rook),
                    promote_button(PieceType::Bishop),
                    promote_button(PieceType::Knight)
                ]
                .padding(10)
                .spacing(10)
                .align_items(Alignment::Center)
                .into(),
            );
        }

        row_entries.push(
            column![
                image(Handle::from_path(
                    &self.piece_images[&Piece::new(chess::PieceType::King, self.board.next_move)]
                )),
                text(format!("Move: {}", self.board.full_move_count)),
                button(text("Generate Fen")).on_press(Message::GenerateFen),
                button(text("Load Fen")).on_press(Message::LoadFen),
                button(text("Restart Game")).on_press(Message::RestartGame(Box::new(
                    Board::from_fen(START_BOARD_FEN).unwrap()
                ))),
                button(text("Move Count")).on_press(Message::CountMoves),
            ]
            .padding(10)
            .spacing(10)
            .align_items(Alignment::Center)
            .into(),
        );

        let row = Row::with_children(row_entries)
            .padding(20)
            .spacing(20)
            .align_items(Alignment::Center);

        container(row)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x()
            .center_y()
            .into()
    }
}
