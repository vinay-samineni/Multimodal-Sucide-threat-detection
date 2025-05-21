graph TD
    A[Input Modalities] --> B1[Text]
    A --> B2[Audio]
    A --> B3[Video]

    subgraph Feature Extraction
        B1 --> C1[Transformer Models: BERT / RoBERTa\nContextual Semantics & Sentiment]
        B2 --> C2[Mel Spectrograms]
        B2 --> C3[GRU Networks: Temporal Dynamics]
        B3 --> C4[CNN (ResNet): Spatial Features]
        B3 --> C5[LSTM / 3D CNN: Temporal Modeling]
    end

    subgraph Modality Fusion
        C1 --> D[Hybrid Fusion: Early + Late Fusion\nAttention Mechanisms]
        C2 --> D
        C3 --> D
        C4 --> D
        C5 --> D
    end

    D --> E[Risk Assessment Module]

    subgraph Risk Assessment Module
        E --> F1[Classification Layer: Fully Connected NN\nRisk Levels (Low, Medium, High)]
        E --> F2[Explainable AI Module: Feature & Modality Attribution]
    end

    subgraph Training and Evaluation
        F1 --> G1[Dataset: DAIC-WOZ\nSMOTE for Class Imbalance]
        F1 --> G2[Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC]
    end

    subgraph Implementation Considerations
        H1[Data Privacy: Regulatory Compliance]
        H2[Real-time Processing: Optimized Inference]
        H3[Continuous Learning: Model Updates]
    end
