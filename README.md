# AutoGluon Assignment

This repository contains AutoGluon notebooks for various machine learning tasks including tabular, text, image, and multimodal learning.

## 📁 Repository Structure

```
autogluon-assignment/
├── README.md
├── part1-kaggle/
│   ├── ieee-fraud-detection.ipynb      (Binary classification on fraud detection)
│   └── california-housing.ipynb        (Regression on housing prices)
├── part2-demos/
│   ├── tabular-quick-start.ipynb       (Quick start with Titanic dataset)
│   ├── tabular-multimodal.ipynb        (Tabular + text features)
│   └── tabular-feature-engineering.ipynb (Feature engineering comparison)
└── extra-credit/
    ├── tabular-multilabel.ipynb
    ├── tabular-gpu.ipynb
    ├── beginner_text.ipynb
    ├── multilingual_text.ipynb
    ├── ner.ipynb
    ├── beginner_image_cls.ipynb
    ├── clip_zeroshot.ipynb
    ├── quick_start_coco.ipynb
    ├── beginner_semantic_seg.ipynb
    ├── document_classification.ipynb
    ├── pdf_classification.ipynb
    ├── image_text_matching.ipynb
    ├── zero_shot_img_txt_matching.ipynb
    ├── text_semantic_search.ipynb
    ├── multimodal_text_tabular.ipynb
    ├── beginner_multimodal.ipynb
    ├── multimodal_ner.ipynb
    ├── forecasting-indepth.ipynb
    └── forecasting-chronos.ipynb
```

## 🚀 How to Run in Google Colab

### Step 1: Open Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Open notebook**
3. Select the **GitHub** tab
4. Enter: `https://github.com/YOUR_USERNAME/autogluon-assignment`
5. Choose the notebook you want to run

**Or use direct links:**
- Replace `YOUR_USERNAME` with your GitHub username
- Format: `https://colab.research.google.com/github/YOUR_USERNAME/autogluon-assignment/blob/main/part1-kaggle/ieee-fraud-detection.ipynb`

### Step 2: Set Your Target Label

Each notebook has a cell where you set the `LABEL` variable:

```python
# For classification (e.g., fraud detection)
LABEL = "isFraud"

# For regression (e.g., house prices)
LABEL = "median_house_value"
```

The notebook will automatically detect:
- **Classification** (binary/multiclass) → ROC-AUC metric
- **Regression** → RMSE metric

### Step 3: Run All Cells

1. Click **Runtime** → **Run all**
2. If prompted for data upload (Kaggle competitions), follow the instructions in the notebook
3. Wait for training to complete (typically 15-30 minutes depending on `time_limit`)

### Step 4: Download Outputs

After the notebook finishes, download these artifacts:

- **`autogluon_model.zip`** - Trained model archive
- **`leaderboard.csv`** - Model performance comparison
- **`feature_importance.csv`** - Top important features
- **`submission.csv`** - Predictions on test set (for Kaggle competitions)

Click the **folder icon** (📁) in the left sidebar, locate the files, and click the three dots (⋮) → **Download**.

## 📊 Notebook Details

### Part 1: Kaggle Competitions

#### ieee-fraud-detection.ipynb
- **Task**: Binary classification
- **Dataset**: IEEE-CIS Fraud Detection (from Kaggle)
- **Target**: `isFraud`
- **Metric**: ROC-AUC
- **Features**: Transaction and identity data

#### california-housing.ipynb
- **Task**: Regression
- **Dataset**: California Housing (from sklearn)
- **Target**: `median_house_value`
- **Metric**: RMSE
- **Features**: Housing attributes (location, rooms, income, etc.)

### Part 2: Demo Notebooks

#### tabular-quick-start.ipynb
- Minimal baseline with Titanic dataset
- Quick `load → fit → leaderboard` workflow

#### tabular-multimodal.ipynb
- Combines tabular and text features
- Demonstrates multimodal learning capabilities

#### tabular-feature-engineering.ipynb
- Compares performance with/without AutoGluon's automatic feature engineering
- Shows before/after model comparison

### Extra Credit

Advanced notebooks covering:
- **Tabular**: Multi-label classification, GPU acceleration
- **Text**: Sentiment analysis, NER, multilingual models
- **Image**: Classification, semantic segmentation, zero-shot learning
- **Multimodal**: Image-text matching, document classification, PDF analysis
- **Time Series**: Forecasting with classical and Chronos models

## 🔧 Configuration Options

Each notebook uses these default settings:

```python
predictor = TabularPredictor(
    label=LABEL,
    eval_metric="auto",  # Auto-detects based on problem type
    path=save_dir
)

predictor.fit(
    train_data,
    presets="medium_quality",  # Balance between speed and accuracy
    time_limit=900,            # 15 minutes (adjust as needed)
    verbosity=2                # Show detailed progress
)
```

### Adjust Training Time

To change training duration, modify the `time_limit` parameter:
- **Quick test**: `time_limit=300` (5 minutes)
- **Standard**: `time_limit=900` (15 minutes)
- **High quality**: `time_limit=3600` (1 hour)

### Change Presets

Available presets (speed vs accuracy tradeoff):
- `best_quality` - Highest accuracy, slower training
- `high_quality` - Good accuracy, moderate training time
- `medium_quality` - Balanced (default)
- `optimize_for_deployment` - Faster inference, smaller models
- `interpretable` - Simpler models, easier to understand

## 📝 Requirements

All notebooks auto-install dependencies:
```python
!pip install -q autogluon
```

Additional packages for specific notebooks:
- `kaggle` - For Kaggle dataset downloads
- `sklearn` - For built-in datasets
- `Pillow` - For image tasks

## 🤝 Contributing

Feel free to add your own notebooks or improve existing ones!

## 📄 License

MIT License - Feel free to use for educational purposes.

## 🆘 Troubleshooting

**Issue**: Notebook crashes or runs out of memory
- **Solution**: Reduce `time_limit` or use `presets="medium_quality"` instead of `best_quality`

**Issue**: Kaggle data download fails
- **Solution**: Upload your `kaggle.json` API token when prompted, or download data manually

**Issue**: Training is too slow
- **Solution**: Enable GPU in Colab (Runtime → Change runtime type → GPU)

**Issue**: Can't find output files
- **Solution**: Check the `Files` panel (📁 icon) in the left sidebar of Colab

## 📚 Resources

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [AutoGluon Tutorials](https://auto.gluon.ai/stable/tutorials/index.html)
- [Google Colab Guide](https://colab.research.google.com/)
