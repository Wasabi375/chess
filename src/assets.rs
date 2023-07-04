use std::collections::HashMap;

use chess::{Color, Piece, PieceType};

#[allow(dead_code)]
pub enum ImageResolution {
    X1,
    X2,
    Px128,
    Px256,
    Px512,
    Px1024,
}

impl ImageResolution {
    const fn folder(&self) -> &'static str {
        match self {
            ImageResolution::X1 => "1x",
            ImageResolution::X2 => "2x",
            ImageResolution::Px128 => "128px",
            ImageResolution::Px256 => "256px",
            ImageResolution::Px512 => "512px",
            ImageResolution::Px1024 => "1024px",
        }
    }

    const fn file_postfix(&self) -> &'static str {
        match self {
            ImageResolution::X1 => "1x",
            ImageResolution::X2 => "2x",
            ImageResolution::Px128 => "png_shadow_128px",
            ImageResolution::Px256 => "png_shadow_256px",
            ImageResolution::Px512 => "png_shadow_512px",
            ImageResolution::Px1024 => "png_shadow_1024px",
        }
    }
}

pub fn generate_piece_image_mapping(resolution: ImageResolution) -> HashMap<Piece, String> {
    let base_path = "assets/JohnPablok_Cburnett_Chess_set/PNGs/with_shadow";

    let mut result = HashMap::new();

    for color in Color::ALL_COLORS {
        let color_text = if color == Color::White { "w" } else { "b" };
        for typ in PieceType::ALL_TYPES {
            let typ_text = match typ {
                PieceType::King => "king",
                PieceType::Queen => "queen",
                PieceType::Bishop => "bishop",
                PieceType::Knight => "knight",
                PieceType::Rook => "rook",
                PieceType::Pawn => "pawn",
            };
            let piece = Piece::new(typ, color);
            result.insert(
                piece,
                format!(
                    "{}/{}/{}_{}_{}.png",
                    base_path,
                    resolution.folder(),
                    color_text,
                    typ_text,
                    resolution.file_postfix()
                ),
            );
        }
    }

    result
}
