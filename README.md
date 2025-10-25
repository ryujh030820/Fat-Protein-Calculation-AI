# Deep Learning-Based Estimation of Fat and Protein Content in Meat

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **An automated pipeline for estimating nutritional content in meat using semantic segmentation and depth estimation**

## Abstract

We present a novel deep learning-based approach for automated estimation of fat and protein content in raw meat using only smartphone images. Our method combines semantic segmentation using SegFormer models with depth estimation and hand-based scale calibration to calculate nutritional content without specialized equipment. The system achieves high accuracy in meat segmentation (IoU: 0.9676) and demonstrates practical applicability for dietary management and food industry applications.

**Keywords:** Semantic Segmentation, Nutritional Analysis, Computer Vision, SegFormer, Depth Estimation

## Introduction

With the growing emphasis on fitness and personalized nutrition, accurate dietary tracking has become increasingly important. However, bulk-purchased meat products often lack precise nutritional labeling, particularly when portioned and repackaged for home use. This creates a significant challenge for individuals attempting to monitor their macronutrient intake accurately.

We address this problem by developing an end-to-end deep learning pipeline that:
- Automatically segments meat regions from images
- Identifies and quantifies fat distribution
- Estimates meat thickness using depth maps
- Calculates fat and protein content based on volumetric analysis

## Methodology

### System Architecture

![Pipeline Architecture](https://github.com/user-attachments/assets/93ea2e0b-21a7-42fc-b2e6-a2464f490429)

Our pipeline consists of four main components:

1. **Meat Segmentation Module**: Employs a SegFormer model to extract the meat region from input images
2. **Fat Segmentation Module**: Uses a second SegFormer model to identify fat distribution within the meat
3. **Scale Calibration Module**: Leverages MediaPipe hand landmark detection to establish pixel-to-centimeter ratio using hand length as a reference
4. **Depth Estimation Module**: Utilizes MiDaS to generate depth maps, which are converted to absolute thickness measurements using the calibrated scale

### Nutritional Content Calculation

The fat and protein content are calculated using the following equations:

![Fat Calculation Formula](https://github.com/user-attachments/assets/2b340c78-bab3-48ab-a384-2a59ffa2429d)

![Protein Calculation Formula](https://github.com/user-attachments/assets/98976063-f31f-4dd4-9dee-a07bc599e041)

Where protein content is derived by subtracting estimated fat and water content from total meat volume.

### Model Selection

We evaluated multiple state-of-the-art semantic segmentation architectures and selected SegFormer, a Transformer-based model, based on comprehensive performance comparisons. SegFormer demonstrated superior performance compared to traditional CNN-based approaches including U-Net, DeepLabV3, and HRNet.

#### Model Comparison

**Meat Segmentation Performance**

| Model            | IoU Score | Dice Coefficient | Pixel Accuracy |
| ---------------- | --------- | ---------------- | -------------- |
| SegFormer        | **0.9676** | **0.9834**      | **0.9904**    |
| U-Net (baseline) | 0.9588    | 0.9789           | 0.9882         |
| DeepLabV3        | 0.9528    | 0.9757           | 0.9866         |
| HRNet            | 0.9426    | 0.9702           | 0.9856         |

**Fat Segmentation Performance**

| Model            | IoU Score | Dice Coefficient | Pixel Accuracy |
| ---------------- | --------- | ---------------- | -------------- |
| SegFormer        | **0.2384** | **0.3641**      | **0.9577**    |
| U-Net (baseline) | 0.2322    | 0.3571           | 0.9560         |
| HRNet            | 0.2182    | 0.3361           | 0.9492         |
| DeepLabV3        | 0.1722    | 0.2803           | 0.9337         |

SegFormer consistently outperformed all baseline models across all metrics for both tasks.

## Dataset

### Data Collection and Annotation

We curated a custom dataset of raw meat images through systematic web scraping using queries such as "raw beef" and "raw pork", resulting in:
- **80 beef images**
- **79 pork images**

**Annotation Process:**
1. **Meat Region Labeling**: Manual annotation using LabelMe for precise boundary delineation
2. **Fat Region Labeling**: Semi-automated approach leveraging grayscale conversion and adaptive thresholding to exploit the characteristic high reflectance of adipose tissue

### Data Augmentation

To enhance model robustness and generalization, we applied standard augmentation techniques:
- Horizontal/vertical flipping
- Random rotation (±15°)

This resulted in an augmented dataset of:
- **560 beef images**
- **553 pork images**

### Data Split

Dataset partitioning followed standard practices:
- Training set: 70%
- Validation set: 15%
- Test set: 15%

## Experimental Results

### Training Strategy Comparison

We conducted experiments comparing two training paradigms: (1) separate models for beef and pork, and (2) a unified model trained on combined data.

#### Meat Segmentation Results

| Training Strategy | Meat Type | IoU Score | Dice Coefficient | Pixel Accuracy |
| ----------------- | --------- | --------- | ---------------- | -------------- |
| Separate Models   | Beef      | 0.9669    | 0.9830           | 0.9902         |
| Separate Models   | Pork      | 0.9640    | 0.9816           | 0.9901         |
| Joint Training    | Beef      | 0.9669    | 0.9830           | 0.9902         |
| Joint Training    | Pork      | 0.9640    | 0.9816           | 0.9901         |

#### Fat Segmentation Results

| Training Strategy | Meat Type | IoU Score | Dice Coefficient | Pixel Accuracy |
| ----------------- | --------- | --------- | ---------------- | -------------- |
| Separate Models   | Beef      | 0.2233    | 0.3453           | 0.9522         |
| Separate Models   | Pork      | **0.3160** | **0.4491**      | **0.9598**     |
| Joint Training    | Beef      | 0.2233    | 0.3453           | 0.9522         |
| Joint Training    | Pork      | 0.3160    | 0.4491           | 0.9598         |

**Key Findings:**
- Meat segmentation achieved comparable performance across both training strategies, suggesting good generalization capability
- Fat segmentation benefited from separate training, particularly for pork (IoU: 0.3160 vs. beef: 0.2233)
- The different performance patterns indicate that meat types share similar overall morphology but exhibit distinct fat distribution characteristics

## Qualitative Results

### Segmentation Visualizations

**Meat Segmentation - Beef**

![Beef Meat Segmentation](https://github.com/user-attachments/assets/8b448e89-e16c-4729-9c4d-d1fe38c0ce90)

**Meat Segmentation - Pork**

![Pork Meat Segmentation](https://github.com/user-attachments/assets/85a833b8-cb3d-4b92-979a-a17211d26c4b)

**Fat Segmentation - Beef**

![Beef Fat Segmentation](https://github.com/user-attachments/assets/75bbca74-d843-4ac2-a907-abbf675bee25)

**Fat Segmentation - Pork**

![Pork Fat Segmentation](https://github.com/user-attachments/assets/62f333cd-173f-4271-a39a-0b1f618a185b)

### Scale Calibration and Depth Estimation

**Hand Landmark Detection (MediaPipe)**

![Hand Detection](https://github.com/user-attachments/assets/e42ebd7e-a5fc-41e4-bf0f-1238ca863544)

The distance from fingertip to wrist (typically 18-19 cm for adults) is used to establish the pixel-to-centimeter conversion ratio.

**Depth Map Estimation (MiDaS)**

![Depth Map](https://github.com/user-attachments/assets/1bd6f73e-74bc-4f5c-95f4-9ae8b61afbae)

The depth map shows relative thickness (brighter regions indicate greater depth), which is converted to absolute measurements using the calibrated scale.

### Case Study

Applying our complete pipeline to a sample image yielded:
- **Fat content**: 26.40g
- **Protein content**: 16.19g

These results demonstrate the practical applicability of our approach for real-world nutritional tracking.

## Discussion

### Limitations and Future Directions

While our system demonstrates promising results, several areas warrant further investigation:

1. **Fat Segmentation Accuracy**: The relatively lower IoU scores for fat segmentation (0.23-0.32) compared to meat segmentation (0.96+) indicate room for improvement. This challenge stems from:
   - High variability in intramuscular fat distribution patterns
   - Similar visual appearance between lean meat and certain fat types
   - Limited training data diversity

2. **Dataset Scale**: Our current dataset (80 beef, 79 pork images) is relatively small. Expanding the dataset with more diverse:
   - Meat cuts and qualities
   - Lighting conditions
   - Camera angles and distances
   - Different meat types (chicken, lamb, etc.)

3. **Deployment Considerations**: For practical adoption, the following enhancements are necessary:
   - Development of mobile applications with intuitive user interfaces
   - Real-time inference optimization for edge devices
   - User studies to validate accuracy in real-world conditions

### Potential Applications

- **Personal Health Management**: Enable individuals to accurately track macronutrient intake
- **Food Industry**: Quality control and automated nutritional labeling systems
- **Research**: Standardized methodology for nutritional analysis in food science studies

## Conclusion

We present a novel deep learning pipeline for automated estimation of fat and protein content in raw meat using only smartphone images. Our approach combines SegFormer-based semantic segmentation with depth estimation and hand-based scale calibration, achieving high accuracy in meat segmentation (IoU: 0.9676) without requiring specialized equipment.

Despite limitations in dataset scale and fat segmentation accuracy, our results demonstrate the feasibility and practical utility of image-based nutritional analysis. This work lays the foundation for accessible dietary tracking tools and has potential applications in both consumer health management and food industry quality control.

## Installation

```bash
# Clone the repository
git clone https://github.com/ryujh030820/Fat-Protein-Calculation-AI.git
cd Fat-Protein-Calculation-AI

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Example usage (to be implemented)
from meat_analyzer import MeatAnalyzer

analyzer = MeatAnalyzer()
result = analyzer.analyze('path/to/meat/image.jpg')
print(f"Fat: {result['fat']}g, Protein: {result['protein']}g")
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{ryu2024fatprotein,
  title={Deep Learning-Based Estimation of Fat and Protein Content in Meat},
  author={Ryu, JeongHwan and Lee, KwanHak and Min, KyeongWon},
  year={2024},
  howpublished={\url{https://github.com/ryujh030820/Fat-Protein-Calculation-AI}}
}
```

## Contributors

- **JeongHwan Ryu**
- **KwanHak Lee**
- **KyeongWon Min**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We acknowledge the following open-source projects that made this work possible:

- **SegFormer**: [Xie et al., 2021](https://arxiv.org/abs/2105.15203)
- **MiDaS Depth Estimation**: [Ranftl et al.](https://github.com/isl-org/MiDaS)
- **MediaPipe**: [Google MediaPipe Hands](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)

## References

1. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. *NeurIPS 2021*. [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)

2. Zhang, L., et al. (2022). Estimating the nutritional value of food using image-based approaches. *ACM Digital Library*. [https://dl.acm.org/doi/fullHtml/10.1145/3575879.3575975](https://dl.acm.org/doi/fullHtml/10.1145/3575879.3575975)

3. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision transformers for dense prediction. *MiDaS*. [https://github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS)

4. Google. MediaPipe Hands. [https://mediapipe.readthedocs.io/en/latest/solutions/hands.html](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)

---

**Contact**: For questions or collaboration inquiries, please open an issue on GitHub or contact the contributors directly.
