#[derive(Debug)]
pub struct Graph<T> {
    nodes: Vec<T>,
}

impl<T> Graph<T> {
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
