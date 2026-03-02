use axum_embed::ServeEmbed;
use rust_embed::RustEmbed;

#[derive(RustEmbed, Clone)]
#[folder = "static/"]
pub struct Assets;

pub fn serve() -> ServeEmbed<Assets> {
    ServeEmbed::<Assets>::new()
}
