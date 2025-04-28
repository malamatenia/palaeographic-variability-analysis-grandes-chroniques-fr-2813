# <p align="center">Interpretable Deep Learning for Palaeographic Variability Analysis; revisiting the scribal hands of Charles V’ Grandes Chroniques de France (Paris, BnF, fr., 2813)</p> <p align="center"> !add webpage link here! </p> <sub> [Malamatenia Vlachou Efstathiou](https://malamatenia.github.io/)</sub> </p>

![a_PCA_methodo.png](./.media/a_PCA_methodo.png)

This repository contains the data and code developed for the paper Interpretable Deep Learning for Palaeographic Variability Analysis; revisiting the scribal hands of Charles V’ Grandes Chroniques de France (Paris, BnF, fr., 2813). 

### Abstract 
This study introduces an interpretable scribal hand characterization and variability analysis based on a deep learning framework combining graphic tools with statistical analysis of scribal practices. Designed to bridge the gap between traditional palaeographic methods — based on qualitative observations — and automatic computational models, our approach enables interpretable character-level inter- and intra-scribal variation analysis and graphic profiling. We demonstrate our method to Charles V’s copy of the Grandes Chroniques de France (Paris, BnF, fr. 2813), revisiting the traditional attribution to two royal scribes of Charles’ V, Henri de Trévou and Raoulet d’Orléans. Through the definition of graphic profiles and complementary statistical analysis of abbreviation usage and space management, we offer a more systematic and context-aware understanding of scribal behaviour. Beyond this case study, the approach opens new possibilities for palaeographic inquiry, from mapping script evolution to characterizing scribal variability in a communicable and interpretable way.

## Repository Structure

```
root/
├── data/                      # Ground Truth (GT) from selected folios of the Paris, BnF, fr. 2813 
│   ├── annotations/           # ALTO XML files with graphemic transcription and layout tagging (SegmOnto)
│   └── images/                # Gallica Images ©BnF
│
├── dataset/                   # The dataset created for our analysis
│   ├── annotation.json        # Annotations and metadata
│   ├── images/                # Extracted lines from the GT, organized per page folder
│       ├── <folio_ID>/
│       ├── <btv1b84472995_f1>
│       ├── <btv1b84472995_f2>/
│                ...             
│   ├── metadata_btv1b84472995.csv  # Metadata (includes details about the dataset)
│
├── prototypes/                # Prototypes created with Learnable Typewriter method
│   ├── <folio_ID>/
│   ├── <btv1b84472995_f1>/       # Folders for each page, containing prototypes for letters a-z
│   ├── <btv1b84472995_f2>/
│              ...
│
├── notebooks/              # Jupyter Notebooks for analysis
│   ├── PCA.ipynb
│   └── statistical_analysis.ipynb
│
└── results/                # Results folder for the outputs of the notebooks
    ├── <graphic_profiles/>
    ├── <statistical_analysis/> 
```

## Data: 
Available also on Zenodo: 10.5281/zenodo.15282371

## Run the analysis on our dataset and reproduce the paper's Results

### 🔤 Graphic Profile Identification and Analysis (Section 7.1) 
[PCA.ipynb](https://github.com/malamatenia/hand-variability-analysis/blob/8eadfdb4b95a999561a8626d5d7c9add724976ba/notebooks/PCA.ipynb) # PCA analysis for Graphic Profiles
### 📈 Statistical Analysis (Section 7.2) 
[statistical_analysis.ipynb]([https://github.com/malamatenia/hand-variability-analysis/blob/inference.ipynb](https://github.com/malamatenia/hand-variability-analysis/blob/8eadfdb4b95a999561a8626d5d7c9add724976ba/notebooks/statistical_analysis.ipynb)). 

## Cite us

```bibtex
@misc{vlachou2025variability,
    title = {Interpretable Deep Learning for Palaeographic Variability Analysis; revisiting the scribal hands of Charles V’ Grandes Chroniques de France (Paris, BnF, fr., 2813)},
    author = {Vlachou-Efstathiou, Malamatenia},
    publisher = {Scriptorium},
    year = {2025},
```

## Acknowledgements
This study was supported by the CNRS through MITI and the 80|Prime program (CrEMe Caractérisation des écritures médiévales), and by the European Research Council (ERC project DISCOVER, number 101076028).  I would like to express my deepest gratitude to my advisors, Prof. Dr. Dominique Stutzmann (IRHT-CNRS) and Prof. Dr. Mathieu Aubry (IMAGINE-ENPC), whose guidance, insightful feedback, and proofreading, as well as continuous support, were instrumental throughout the writing of this paper.

📝 Check out also: 
[Vlachou-Efstathiou, M., Siglidis, I., Stutzmann, D. & Aubry, M. (2024). An Interpretable Deep Learning Approach for Morphological Script Type Analysis.]([https://imagine.enpc.fr/~siglidii/learnable-typewriter/](https://learnable-handwriter.github.io/))
[Siglidis, I., Gonthier, N., Gaubil, J., Monnier, T., & Aubry, M. (2023). The Learnable Typewriter: A Generative Approach to Text Analysis.](https://imagine.enpc.fr/~siglidii/learnable-typewriter/)

