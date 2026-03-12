The dataset created by `StaticGraphTemporalSignal` is a **temporal graph dataset** designed for node‑level regression tasks. It captures the dynamic behavior of a supply chain over 221 days. Let me break down its components and how they will be used to train your Graph Transformer.

---

## 1. What the Dataset Contains

The dataset consists of **221 snapshots**, one for each day from January 1 to August 9, 2023. Each snapshot is a static graph (same edges every day) with time‑varying node features and a target value per node.

### **Graph Structure (Static)**

- **Nodes**: 40 products (e.g., `SOS008L02P`, `SOS005L04P`, …).  
- **Edges**: Connections based on shared storage location (from `Edges (Storage Location).csv`).  
  - Each edge has a **weight** (the storage location code, e.g., 330.0, 1130.0).  
- The graph is **undirected** and **homogeneous** (only one node type).

### **Node Features (Dynamic)**

For each product (node) at each day, you have **4 features**:

1. **Production volume** (from `Production .csv`)  
2. **Factory issues** (from `Factory Issue.csv`) – raw material consumption  
3. **Delivery to distributor** (from `Delivery To distributor.csv`) – outbound flow  
4. **Sales orders** (from `Sales Order.csv`) – demand signal

Thus, every snapshot provides a matrix `X` of shape **(40, 4)** – 40 nodes × 4 features.

### **Targets (Dynamic)**

The target for each snapshot is the **sales value** for each product on that same day (shape: `(40,)`).  
So the task is: **given the current day’s features (production, issues, delivery, sales) and the graph structure, predict the same day’s sales for every product.**

> **Note:** This is a node‑level regression task. In your paper, you may later shift the targets to predict **next day’s sales** (forecasting). The dataset structure supports that easily.

---

## 2. How the Dataset Helps Train a Graph Transformer

### **Why Graph Neural Networks?**

In a supply chain, products are not independent. Their demand (sales) may be influenced by:

- Production constraints
- Shared raw materials
- Common storage locations
- Distribution patterns

The graph edges (based on storage location) encode these relationships. By propagating information along edges, a GNN can capture how one product’s state affects another’s.

### **Why a Graph Transformer?**

The Graph Transformer uses **global attention** – each node can attend to all other nodes, not just immediate neighbors. This is crucial for supply chains because:

- A disruption in a raw material can ripple across many products (long‑range dependencies).
- Demand shocks may propagate through complex, indirect relationships.

The dataset provides **rich, real‑world features** that allow the Transformer to learn these intricate dependencies. Each training step looks like this:

1. **Input**: a snapshot (graph + node features X)
2. **Model**: Graph Transformer applies multi‑head self‑attention (with edge bias) to update node representations.
3. **Output**: predicted sales for each node.
4. **Loss**: MSE between predictions and true sales (snapshot.y).

Because the graph structure is fixed, the model can learn **which storage‑location connections are most important** and **which products influence each other even if they are far apart in the graph**.

### **Temporal Aspect**

Even though each snapshot is processed independently, the dataset contains **221 snapshots**, allowing the model to see many different configurations of features. Over many days, it learns a general function that maps (features, graph) → sales. This is essentially a **spatio‑temporal** problem, but with a static graph.

If you later adapt it to forecasting (predicting t+1), the model would also need to capture temporal dynamics – but the current setup is a solid baseline for comparing GAT vs. Graph Transformer on the same spatial task.

---

## 3. Summary

| Component      | Shape               | Description                                                                 |
|----------------|---------------------|-----------------------------------------------------------------------------|
| Snapshots      | 221                 | Daily observations from Jan 1 to Aug 9, 2023                                |
| Node features  | (40, 4) per snapshot | Production, factory issues, delivery, sales for each product                |
| Edge index     | (2, num_edges)      | Undirected connections based on shared storage location                     |
| Edge weights   | (num_edges, 1)      | Storage location code (can be used as attention bias or ignored)            |
| Targets        | (40,) per snapshot  | Sales value for each product (same day)                                     |

This dataset is ideal for training your Graph Transformer because:

- It’s a **real‑world** supply chain graph with **domain‑meaningful edges**.
- Node features are **multivariate time series**, offering rich context.
- The task (predicting sales) is directly relevant to supply chain planning.
- The static graph allows you to focus on comparing **local vs. global attention** (GAT vs. Transformer) without extra complexity.

Once you train the model, you’ll be able to analyze whether the global attention of the Transformer captures long‑range dependencies that GAT misses, leading to better sales predictions.
