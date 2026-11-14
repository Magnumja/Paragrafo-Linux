use crate::extraction::Paragraph;
use std::collections::HashMap;
use std::ops::Range;

/// Page offsets and completed count are updated only when document data changes.
#[derive(Clone, Debug, Default)]
pub struct DocumentState {
    paragraphs: Vec<Paragraph>,
    page_ranges: HashMap<usize, Range<usize>>,
    pages: Vec<usize>,
    completed: usize,
    completed_by_page: HashMap<usize, usize>,
}

impl DocumentState {
    pub fn new(paragraphs: Vec<Paragraph>) -> Self {
        let mut state = Self { paragraphs, ..Self::default() };
        for (index, paragraph) in state.paragraphs.iter().enumerate() {
            let page = paragraph.page;
            match state.page_ranges.get_mut(&page) {
                Some(range) => range.end = index + 1,
                None => {
                    state.pages.push(page);
                    state.page_ranges.insert(page, index..index + 1);
                }
            }
            if !paragraph.translated.is_empty() {
                state.completed += 1;
                *state.completed_by_page.entry(page).or_default() += 1;
            }
        }
        state
    }

    pub fn paragraphs(&self) -> &[Paragraph] {
        &self.paragraphs
    }
    pub fn pages(&self) -> &[usize] {
        &self.pages
    }
    pub fn page_range(&self, page: usize) -> Option<Range<usize>> {
        self.page_ranges.get(&page).cloned()
    }
    pub fn completed(&self) -> usize {
        self.completed
    }
    pub fn completed_on(&self, page: usize) -> usize {
        self.completed_by_page.get(&page).copied().unwrap_or(0)
    }
    pub fn has_page_progress(&self) -> bool {
        !self.completed_by_page.is_empty()
    }
    pub fn is_empty(&self) -> bool {
        self.paragraphs.is_empty()
    }
    pub fn count(&self) -> usize {
        self.paragraphs.len()
    }
    pub fn is_complete(&self) -> bool {
        !self.is_empty() && self.completed == self.count()
    }

    pub fn apply(&mut self, text: &str, index: usize) {
        if index >= self.paragraphs.len() || text.is_empty() {
            return;
        }
        if self.paragraphs[index].translated.is_empty() {
            self.completed += 1;
            *self.completed_by_page.entry(self.paragraphs[index].page).or_default() += 1;
        }
        self.paragraphs[index].translated = text.to_string();
    }

    pub fn clear_translations(&mut self) {
        for paragraph in &mut self.paragraphs {
            paragraph.translated.clear();
        }
        self.completed = 0;
        self.completed_by_page.clear();
    }

    /// Pending paragraphs starting at the page being read, then wrapping to the beginning.
    pub fn pending_order(&self, page: usize) -> Vec<usize> {
        let start = self.page_ranges.get(&page).map_or(0, |range| range.start);
        (start..self.count()).chain(0..start).filter(|&index| self.paragraphs[index].translated.is_empty()).collect()
    }
}
