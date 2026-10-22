# Graph Neural Network-Based Personalized Fashion Recommendation System

**An end-to-end fashion recommendation system powered by Graph Neural Networks, GraphSAGE, Bayesian Personalized Ranking, and hard negative sampling.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.7-orange)](https://pytorch-geometric.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

## Overview

Traditional recommendation systems often struggle with sparse purchase histories, large product catalogues, and changing customer preferences. This project addresses these challenges by developing a **Graph Neural Network (GNN)-based personalized fashion recommendation system** using the H&M Personalized Fashion Recommendations dataset.

The system models customer–article interactions as a bipartite graph and learns meaningful representations of customers and fashion products. It combines GraphSAGE-based neighborhood aggregation, season-aware embeddings, Bayesian Personalized Ranking (BPR), and structured hard negative sampling to generate personalized product rankings.

The trained model is integrated into an interactive Streamlit application that allows users to explore customer purchase histories, generate personalized recommendations, filter results, and visualize learned embeddings.

## Key Features

- **Graph-based recommendation:** Models customer–article interactions using a heterogeneous bipartite graph.
- **GraphSAGE architecture:** Learns customer and article embeddings through two layers of neighborhood aggregation.
- **Season-aware personalization:** Incorporates learnable season embeddings to capture seasonal purchasing patterns.
- **BPR optimization:** Uses pairwise ranking loss to prioritize relevant products over negative samples.
- **Hard negative sampling:** Samples challenging negatives from popular products, similar product categories, and seasonal product pools.
- **Realistic evaluation:** Uses a candidate pool of 1,000 items per user while masking previously observed training interactions.
- **Interactive dashboard:** Provides personalized recommendations, customer profiles, purchase history, product filters, and embedding-space visualizations.
- **Fairness analysis:** Evaluates recommendation quality across customer age groups.
- **Lightweight inference:** Uses precomputed embeddings and NumPy-based scoring for deployment without requiring a GPU.

## Model Performance

The final model was evaluated on held-out test interactions using the candidate pool evaluation protocol described in the dissertation.

| Metric | Test Score |
|---|---:|
| ROC-AUC | 0.7126 |
| Recall@10 | 0.0321 |
| MAP@10 | 0.0121 |
| Precision@10 | 0.0168 |

**Evaluation setup:** A candidate pool of 1,000 articles per user, with previously observed training interactions excluded from recommendation rankings.

These results should be interpreted in the context of sparse implicit-feedback data. The metrics are specific to this evaluation protocol and are not directly comparable to results obtained through full-catalogue ranking or different candidate-sampling strategies.

## Dataset

This project uses the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations), released through Kaggle.

The original dataset contains transaction histories, customer attributes, article metadata, and product images.

### Dataset components

| File | Description |
|---|---|
| `transactions_train.csv` | Customer purchase interactions, transaction dates, article IDs, and prices |
| `customers.csv` | Customer demographic and membership information |
| `articles.csv` | Product metadata, categories, colours, departments, and garment attributes |
| Product images | Fashion article images used by the visualization and recommendation interface where available |

### Processed dataset

The final experimental dataset contains:

- **614,388** transactions
- **18,211** unique customers
- **28,111** unique articles
- Customer age range: **20–34 years**
- Interaction density: approximately **0.12%**

Data preprocessing included age-based filtering, minimum interaction thresholds for customers and articles, feature preparation, and customer–article interaction graph construction.

**Dataset access:** Download the data directly from Kaggle and comply with its competition rules and data-use terms. The raw dataset is not included in this repository.

## Methodology

### 1. Data preprocessing and exploratory analysis

Transaction records are joined with customer demographic information and article metadata. The analysis examines customer age distributions, product category frequencies, purchase activity over time, and seasonal purchasing patterns.

Season labels are assigned using the following mapping:

| Months | Season |
|---|---|
| December–February | Winter |
| March–May | Spring |
| June–August | Summer |
| September–November | Autumn |

### 2. Bipartite graph construction

The purchase history is represented as a graph containing two node types:

- **Customer nodes:** Represent individual customers, with standardized age features.
- **Article nodes:** Represent fashion products, with encoded categorical metadata and log-scaled popularity.

Observed purchases form customer-to-article edges, with reverse edges added to enable bidirectional message passing.

The graph is implemented using PyTorch Geometric.

### 3. GraphSAGE recommendation model

The model uses a two-layer GraphSAGE encoder with mean aggregation to learn 128-dimensional customer and article embeddings.

Key architectural components include:

- Type-specific input projections
- Learnable 8-dimensional season embeddings
- Two GraphSAGE convolutional layers
- Layer normalization, ReLU activation, and dropout
- L2-normalized embeddings
- Temperature-scaled dot-product scoring

The recommendation score between customer \(u\) and article \(i\) is computed as:

\[
\hat{s}_{ui}=\frac{\mathbf{z}_u^\top\mathbf{z}_i}{\tau}+b
\]

where \(\mathbf{z}_u\) and \(\mathbf{z}_i\) are normalized embeddings, \(\tau=0.07\) is the temperature parameter, and \(b\) is a learnable global bias.

### 4. Bayesian Personalized Ranking and hard negatives

The model is trained using Bayesian Personalized Ranking (BPR), which encourages a customer's observed positive items to receive higher scores than sampled negative items.

Each positive interaction is paired with eight hard negatives drawn from:

- **Popularity pool (40%):** Frequently purchased articles.
- **Category pool (40%):** Articles sharing the positive item's product type.
- **Season pool (20%):** Articles associated with the same season.

Previously purchased training items are excluded from negative sampling where applicable.

### 5. Candidate pool evaluation

Rather than relying exclusively on all-item ranking, the evaluation protocol constructs a candidate pool of 1,000 articles per user, incorporating held-out positive items, popular articles, and randomly sampled candidates.

Previously observed training interactions are masked before ranking.

Performance is measured using ROC-AUC, Recall@10, Precision@10, and MAP@10.

### 6. Fairness analysis

Recommendation quality is examined across three age groups:

| Age group | Approximate MAP@10 |
|---|---:|
| 20–24 | 0.0186 |
| 25–29 | 0.0083 |
| 30–34 | 0.0155 |

The results indicate differences in recommendation quality across age groups, with the 25–29 cohort performing less well than the other groups. This highlights the importance of evaluating recommender systems beyond aggregate performance.

## Streamlit Application

The trained model is integrated into an interactive Streamlit dashboard for exploring personalized fashion recommendations.

### Application capabilities

- Select a customer using their customer ID.
- View customer demographics and purchase statistics.
- Explore historical purchases and associated product images.
- Generate ranked product recommendations.
- Apply optional season and product-type filters.
- Adjust the number of recommendations displayed.
- Visualize recommendation scores and customer–article embedding relationships.
- Inspect item-level recommendation information and explanations.

The application uses precomputed embeddings, ID mappings, article metadata, and transaction history to perform lightweight inference.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Graph neural networks | PyTorch, PyTorch Geometric |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib, PCA |
| Application development | Streamlit |
| Training environment | Kaggle GPU runtime, CUDA 12.8 |
| Model optimization | Adam, BPR loss, ReduceLROnPlateau |

## Project Structure

The following is a suggested structure for organizing the repository. Adapt the paths and filenames to match the code already present in your GitHub project.

```text
fashion-recommendation-gnn/
│
├── app.py                    # Streamlit application entry point
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── notebooks/
│   └── model_training.ipynb  # Data analysis and model training
│
├── src/
│   ├── data_preprocessing.py  # Dataset preparation
│   ├── graph_builder.py       # Graph construction
│   ├── model.py               # GraphSAGE architecture
│   ├── train.py               # Training and BPR optimization
│   └── evaluate.py            # Ranking metrics and evaluation
│
├── artifacts/                 # Model artifacts and mappings
├── data/                      # Local dataset files
└── images/                    # Product images, where applicable
```

## Installation and Usage

### Prerequisites

- Python 3.12
- Git
- Access to the H&M dataset on Kaggle
- Compatible PyTorch and PyTorch Geometric installations

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the placeholder URL with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install a compatible PyTorch and PyTorch Geometric build if they are not already included in your dependency file. Refer to the official installation guides for the appropriate configuration.

### 4. Prepare the dataset

Download the H&M dataset from Kaggle and place the required CSV files and product images in the locations expected by your preprocessing scripts and application.

Update the dataset paths in the code if necessary.

### 5. Train the model

If your repository includes the suggested training script, run:

```bash
python src/train.py
```

Training requires the prepared dataset and a compatible environment. The trained model, embeddings, and mapping files must be saved to the locations expected by the application.

If your existing code uses a notebook or a different entry point, use that workflow instead.

### 6. Launch the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to interact with the recommendation dashboard.

**Note:** The commands and directory names above are illustrative until verified against the actual repository files. The application cannot run successfully until the expected data and model artifacts are available.

## Limitations

- **Sparse interactions:** Most possible customer–article combinations have no observed purchase history.
- **Cold-start problem:** New customers and articles with insufficient interactions are difficult to recommend reliably.
- **Limited feature representation:** Product images are available in the source dataset but are not used directly in the GNN architecture.
- **Candidate pool dependence:** Reported ranking metrics depend on the candidate construction and evaluation protocol.
- **Age-group performance differences:** Recommendation quality varies across the evaluated demographic subgroups.
- **Dataset scope:** The experiments focus on customers aged 20–34 and should not automatically be generalized to all customer populations.

## Future Improvements

Potential extensions include:

- Incorporating pretrained visual embeddings from product images.
- Comparing GraphSAGE with LightGCN and other graph-based recommenders.
- Improving cold-start recommendations using richer customer and product features.
- Exploring temporal graph models for evolving fashion preferences.
- Applying reweighting or calibration methods to reduce subgroup performance disparities.
- Evaluating performance with additional candidate-generation strategies and broader ranking benchmarks.
- Optimizing inference and deploying the application to a cloud platform.

## Research Context

This project was completed as part of the Master of Science in Statistics dissertation at St. Xavier’s University, Kolkata, under the supervision of Dr. Indrani Mukherjee.

The work combines statistical evaluation, graph representation learning, implicit-feedback recommendation, and interactive application development.

## References

1. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs*. NeurIPS.
2. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI.
3. Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop.
4. H&M Group. (2022). *H&M Personalized Fashion Recommendations*. Kaggle.

## Author

**Srijita Biswas**

Machine Learning | Graph Neural Networks | Recommendation Systems | Data Science

---

*Built with Python, PyTorch Geometric, and Streamlit to explore graph-based personalization for fashion retail.*
