use std::{
    mem,
    sync::atomic::{AtomicU32, Ordering},
};

#[derive(Debug, Default)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(f: f32) -> Self {
        AtomicF32(AtomicU32::new(unsafe { mem::transmute(f) }))
    }

    #[inline]
    pub fn load(&self, ord: Ordering) -> f32 {
        let u = self.0.load(ord);
        unsafe { mem::transmute(u) }
    }

    #[inline]
    pub fn store(&self, f: f32, ord: Ordering) {
        let u = unsafe { mem::transmute(f) };
        self.0.store(u, ord);
    }
}
