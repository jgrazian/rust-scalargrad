/// Wrapper over Vec<T> to store some T and provide access to that data
#[derive(Debug)]
pub struct Arena<T> {
    nodes: Vec<T>,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::default(),
        }
    }

    pub fn push(&mut self, data: T) -> usize {
        self.nodes.push(data);
        self.nodes.len() - 1
    }

    pub fn node(&self, id: usize) -> &T {
        &self.nodes[id]
    }

    pub fn node_mut(&mut self, id: usize) -> &mut T {
        &mut self.nodes[id]
    }
}
