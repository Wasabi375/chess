use iced::{Color, Length, Size};
use iced_native::{
    layout::Node,
    renderer::Quad,
    widget::{tree, Tree},
    Element, Renderer, Widget,
};

// Color pallet base: RGB(200, 160, 120)
const BLACK_SQUARE: Color = Color::from_rgb(0.29, 0.20, 0.11);
const WHITE_SQUARE: Color = Color::from_rgb(0.71, 0.50, 0.28);
const SELECTED_OVERLAY: Color = Color::from_rgba(0.68, 0.40, 0.76, 0.8);
const VALID_MOVE_OVERLAY: Color = Color::from_rgba(0.40, 0.76, 0.50, 0.8);
const HIGHLIGHT_OVERLAY: Color = Color::from_rgba(0.8, 0.0, 0.0, 0.8);

pub fn square<'a, Message, R: Renderer>(
    bg_black: bool,
    length: f32,
    content: Option<impl Into<Element<'a, Message, R>>>,
    selected: bool,
    valid_move: bool,
    highlight: bool,
) -> Square<'a, Message, R> {
    // only allow either selected or valid
    assert!(!selected || !valid_move);

    Square {
        background_black: bg_black,
        length,
        content: content.map(|c| c.into()),
        selected,
        valid_move,
        highlight,
    }
}

pub struct Square<'a, Message, R: Renderer> {
    background_black: bool,
    selected: bool,
    valid_move: bool,
    highlight: bool,
    length: f32,
    content: Option<Element<'a, Message, R>>,
}

impl<Message, R: Renderer> Widget<Message, R> for Square<'_, Message, R> {
    fn width(&self) -> Length {
        Length::Fixed(self.length)
    }

    fn height(&self) -> Length {
        Length::Fixed(self.length)
    }

    fn layout(
        &self,
        _renderer: &R,
        _limits: &iced_native::layout::Limits,
    ) -> iced_native::layout::Node {
        Node::new(Size::new(self.length, self.length))
    }

    fn draw(
        &self,
        state: &Tree,
        renderer: &mut R,
        theme: &<R as iced_native::Renderer>::Theme,
        style: &iced_native::renderer::Style,
        layout: iced_native::Layout<'_>,
        cursor_position: iced::Point,
        viewport: &iced::Rectangle,
    ) {
        renderer.with_layer(layout.bounds(), |renderer| {
            renderer.fill_quad(
                Quad {
                    bounds: layout.bounds(),
                    border_radius: 0.0.into(),
                    border_width: 2.0,
                    border_color: Color::BLACK,
                },
                if self.background_black {
                    BLACK_SQUARE
                } else {
                    WHITE_SQUARE
                },
            );
        });

        if self.selected || self.valid_move {
            let color = if self.selected {
                SELECTED_OVERLAY
            } else {
                assert!(self.valid_move);
                VALID_MOVE_OVERLAY
            };
            renderer.with_layer(layout.bounds(), |renderer| {
                renderer.fill_quad(
                    Quad {
                        bounds: layout.bounds(),
                        border_radius: 0.0.into(),
                        border_width: 0.0,
                        border_color: Color::TRANSPARENT,
                    },
                    color,
                );
            });
        }

        renderer.with_layer(layout.bounds(), |renderer| {
            if let Some(content) = self.content.as_ref() {
                content.as_widget().draw(
                    state,
                    renderer,
                    theme,
                    style,
                    layout,
                    cursor_position,
                    viewport,
                );
            }
        });
        if self.highlight {
            renderer.with_layer(layout.bounds(), |renderer| {
                renderer.fill_quad(
                    Quad {
                        bounds: layout.bounds(),
                        border_radius: 0.0.into(),
                        border_width: 0.0,
                        border_color: Color::TRANSPARENT,
                    },
                    HIGHLIGHT_OVERLAY,
                );
            });
        }
    }

    fn tag(&self) -> tree::Tag {
        tree::Tag::stateless()
    }

    fn state(&self) -> tree::State {
        tree::State::None
    }

    fn children(&self) -> Vec<Tree> {
        if let Some(content) = self.content.as_ref() {
            vec![Tree::new(content)]
        } else {
            vec![]
        }
    }

    fn diff(&self, tree: &mut Tree) {
        if let Some(content) = self.content.as_ref() {
            tree.diff(content)
        }
    }
}

impl<'a, M: 'a, R: 'a> From<Square<'a, M, R>> for Element<'a, M, R>
where
    R: Renderer,
{
    fn from(value: Square<'a, M, R>) -> Self {
        Self::new(value)
    }
}
