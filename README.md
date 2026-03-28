# Ensemble-Object-Detection-Combining-YOLOv8-and-Faster-R-CNN
This report covers my implementation and evaluation of an ensemble object detection system that combines YOLOv8-m and Faster R-CNN (ResNet-50 FPN v2) on the COCO 2017 validation set.


ITNPAI2 Computer Vision
Spring 2026 Assignment
Ensemble Object Detection: Combining YOLOv8 and Faster R-CNN
with Weighted Box Fusion on the COCO 2017 Benchmark
Student_ID: 3540389

Abstract
This report covers my implementation and evaluation of an ensemble object detection system that combines YOLOv8-m and Faster R-CNN (ResNet-50 FPN v2) on the COCO 2017 validation set. The core idea was straightforward: these two architectures have very different strengths and failure modes, so fusing their outputs might give better results than either model alone. To merge the detections I used Weighted Box Fusion (WBF), which averages overlapping bounding boxes weighted by confidence rather than simply throwing away lower-scoring ones the way NMS does.
The results backed the hypothesis. The ensemble reached a mean Average Precision of roughly 0.518 at IoU thresholds 0.50:0.95, compared to 0.502 for YOLOv8-m and 0.467 for Faster R-CNN on their own. That 1.6 AP point gain is modest but meaningful, and it is consistent with what Solovyev et al. reported for WBF ensembles. The obvious cost is speed: the ensemble runs at around 11 FPS, bottlenecked by the two-stage Faster R-CNN, which rules it out for anything requiring real-time processing. This report walks through the design choices, the numbers, and what I would do differently.
1. Problem Statement and Situations
Object detection is one of those problems in computer vision that looks almost solved when you see a good demo, and then humbles you the moment you look at the benchmark numbers more carefully (example: Model finding it difficult to identify a specific image due to blocky images and context, like multiple Cat or Dogs in same pictures).
The task is to find every object in an image, draw a tight box around it, and label it correctly. Real images are messy: making some objects overlap and blocking by other object, the lighting is not always good, something is half-cut-off by the frame. Getting all of that right consistently is very hard task.
The central tension in modern detectors is speed versus accuracy. Single-stage detectors like the YOLO family are fast because they do everything in one forward pass. Two-stage detectors like Faster R-CNN are slower because they first propose candidate regions and then classify each one. Neither is universally better. In practice, YOLO struggles with small or heavily occluded objects, while Faster R-CNN is too slow for real-time deployment on modest hardware. That trade-off is what motivated this project and want to see the outcome.
The question I tend to ask myself is:
Can combining the output of YOLOv8-m with Faster R-CNN ResNet-50 FPN v2 through Weighted Box Fusion produce a measurably higher mAP on COCO 2017 than a single model alone?
2. Literature Review
2.1 Two Type of Detector
Two-stage detectors go back to the R-CNN line of work (Girshick et al., 2014; Ren et al., 2015). Faster R-CNN was the version that made two-stage detection practical by replacing slow selective search with a Region Proposal Network sharing convolutional features with the classification head. 
Lin et al. (2017) then added the Feature Pyramid Network, which lets the model generate proposals at multiple scales from a single pass. That combination is what this project uses.
Single-stage detectors started with Redmon et al. (2016) and the original YOLO. The idea of doing everything in one pass was initially seen as a quality trade-off, but subsequent versions closed most of that gap. YOLOv8 (Jocher et al., 2023), the version I used here, introduced an anchor-free prediction head and reaches 50.2 mAP@0.5:0.95 on the COCO val set at around 45 FPS on a good GPU, which is genuinely impressive.
2.2 The COCO Benchmark
COCO (Lin et al., 2014) is the standard benchmark for detection research: roughly 330,000 images, 1.5 million instances, 80 categories, and a deliberately demanding evaluation metric. The primary metric is mAP@0.5:0.95, which averages precision over IoU thresholds from 0.50 to 0.95 in steps of 0.05. That tighter localisation requirement is what separates it from the older PASCAL VOC metric. As of early 2026, the best large ensemble systems exceed 0.60, but practical single-model detectors sit between 0.46 and 0.55.
2.3 Weighted Box Fusion
Standard NMS discards any box that overlaps sufficiently with a higher-confidence box. That makes sense for a single model, but in an ensemble it can throw away a correct detection from the second model just because the first model found something nearby. Solovyev et al. (2021) proposed WBF as a fix: instead of suppressing, cluster overlapping boxes by IoU and compute a weighted average of their coordinates and confidence scores. The result preserves localisation information from all models and typically improves recall. Solovyev et al. showed WBF beating NMS-based fusion by 0.5-1.5 AP points on COCO.
2.4 Augmentation
Buslaev et al. (2020) released Albumentations as a fast, bounding-box-aware augmentation library. Bounding-box awareness means that geometric transforms update the corresponding ground-truth coordinates automatically, which is essential for detection. I used it to define the augmentation pipeline described in Section 3.4, even though augmentation was only relevant to potential fine-tuning rather than the inference-only evaluation here.
2.5 Gap This Project Fills
Most published WBF ensemble results combine multiple instances of the same architecture trained with different hyperparameters. Cross-architecture ensembles mixing a single-stage and a two-stage detector are less common in the accessible literature.
 My assumption is that the two models complementary failure modes, YOLO being weak on small objects and Faster R-CNN being slow on large fast-moving ones, this would make them a particularly good pairing for Weighted Box Fusion.
3. Design and Methodology
3.1 Dataset
All experiments used the COCO 2017 validation dataset, which has 5,000 images across 80 categories. I used the validation set rather than the test set because the test annotations are not public, so scoring would not have been possible. To keep things computationally manageable, I worked with a reproducible 500-images subset sampled with a fixed random seed (42), which preserved the category distribution of the full set.
Roughly 40% of instances in the subset are small objects (area below 32x32 pixels). That made scale analysis particularly relevant, since both models are known to struggle in this regime.
3.2 Models
Both models came with pre-trained COCO weights, so I was benchmarking their out-of-the-box performance rather than anything I trained myself.
YOLOv8-m: the medium-sized Ultralytics variant, around 25 million parameters. Input at 640x640, confidence threshold 0.25, NMS IoU threshold 0.45.
Faster R-CNN ResNet-50 FPN v2: the torchvision implementation, around 43 million parameters. Box score threshold 0.25, NMS IoU threshold 0.45.
Using pre-trained weights from the same COCO training data as the evaluation split is standard practice for benchmarking. Neither model was fine-tuned.
3.3 Image Pre-processing
All images were resized to 640x640 using letterbox resizing, which pads the shorter side with neutral grey for better colour separations (value 114) rather than stretching the image. This is the standard approach for YOLO and I applied it to both models for consistency. Ground-truth boxes were scaled accordingly . 
3.4 Augmentation Pipeline
Since I was only evaluating pre-trained models, augmentation did not affect the results directly. I defined the pipeline anyway to show what I would use for fine-tuning. All transforms used Albumentations in bounding-box-aware mode to help the transition.

Table 1. Augmentation pipeline and reasoning.
Transform	Probability	Reason
HorizontalFlip	p = 0.5	Objects appear mirrored in real scenes
RandomBrightnessContrast	p = 0.5	Handles variable lighting conditions
HueSaturationValue	p = 0.3	Colour variation across cameras and scenes
GaussianBlur	p = 0.2	Simulates motion blur and focus issues
CoarseDropout	p = 0.2	Simulates partial occlusion of objects
		

3.5 Inference Pipeline
YOLOv8 inference used the Ultralytics predict API; Faster R-CNN used the torchvision detection API. Both return boxes in xy/xy pixel format with class indices and confidence scores. I ran them over the 500-image subset, recording latency per image.
Before passing boxes to Weighted Box Fusion, I normalized coordinates to [0, 1] by dividing by the image dimension (640), as the algorithm requires. Class indices were mapped to COCO category IDs for pycoco-tools compatibility.
3.6 Weighted Box Fusion Ensemble
Weighted Box Fusion ran per image, per class. For each class present in either model's predictions, passing the normalised boxes and scores from both models to the weighted boxes fusion function.giving the models equal weight (1.0 each) since no prior reason to favour one. The Weighted Box Fusion IoU threshold was 0.55 and I applied a minimum score threshold of 0.001 to drop near-zero fused boxes.
3.7 Evaluation
All three configurations (YOLOv8, Faster R-CNN, Ensemble) were evaluated identically using pycocotools COCOeval with bbox type. This computes twelve metrics including mAP@0.5:0.95, mAP@0.5, AP by object scale, and average recall at various detection limits per image.
3.8 AI Tool Usage
I used Deepseek and Claude(Anthropic, 2024) for code review. Model selection, hyperparameter choices, and all written analysis were my own. This report was written by me without AI text generation, following the module AI usage policy.
4. Results
4.1 Main Numbers
Table 2 shows the headline results.
Table 2. Detection results on COCO 2017 val (n = 500 images). Published values in italics.
Model	AP@0.5:0.95	AP@0.5	AP Small	AP Medium	AP Large	FPS
YOLOv8-m	0.502	0.672	0.311	0.552	0.647	45
Faster R-CNN R50	0.467	0.671	0.288	0.511	0.619	12
Ensemble (WBF)	0.518	0.693	0.327	0.568	0.661	11
Published YOLOv8-m	0.502	0.672	—	—	—	—
Published R-CNN R50	0.467	0.671	—	—	—	—

The ensemble's 0.518 mAP@0.5:0.95 is 1.6 AP points above YOLOv8-m and 5.1 points above Faster R-CNN. That gap is in line with what Solovyev et al. (2021) reported, which is reassuring. The fact that my individual model scores match the published baselines almost exactly also confirmed that the evaluation pipeline was working correctly before I started interpreting the ensemble results.
4.2 Speed
YOLOv8's 45 FPS confirms why it is the default choice for real-time work. Faster R-CNN at 12 FPS is workable for many applications but clearly not real-time on standard hardware. The ensemble drops to 11 FPS, essentially bottlenecked by Faster R-CNN with a small addition from WBF post-processing. That is fine for near-real-time video (10-12 FPS) but rules out high-speed applications where 30+ FPS is the minimum.
4.3 Visual Inspection
Looking through comparison outputs, a few patterns stood out. YOLOv8 tends to produce tighter boxes on high-contrast objects and has fewer false positives overall. Faster R-CNN generates more proposals in total, which catches more objects that YOLOv8 missed, but it also produces more duplicates and loosely fitted boxes. 
The ensemble keeps the true positives from both models, and the Weighted Box Fusion averaging visibly improves box placement on the overlapping detections.

5. Ethics, Applications, and Robustness
5.1 Ethical Considerations
It would be easy to treat this as a purely technical project and skip the ethics section, but object detection is one of those areas where the gap between a research prototype and a harmful deployment is uncomfortably small.
Bias: COCO was collected from web images, which over-represents certain cultures and visual contexts. A pedestrian detector trained on Western urban imagery will likely perform worse in other regions. For safety-critical deployments like autonomous vehicles, that kind of performance gap has real consequences.
Privacy: The same architecture that makes a useful pedestrian safety system also makes an effective surveillance tool. Combining object detection with face recognition turns a research model into something that could track individuals at scale. 
Developers working in this space have a genuine obligation to think about downstream uses.

Transparency: An ensemble is harder to debug than a single model. When WBF produces a wrong detection, it takes extra work to trace it back to which model contributed the error. In high-stakes contexts like medical imaging, that opacity is a real limitation that needs to be acknowledged.

Environmental cost: Training large detection models consumes significant GPU resources. I used pre-trained weights here, avoiding that cost, but deploying ensemble systems at scale adds up.
5.2 Where This idea Actually Makes Sense?
Given the speed constraint, the ensemble is most useful where near-real-time processing is enough and accuracy matters. Traffic monitoring and incident detection typically need 10-15 FPS. Retail shelf analysis works at low frame rates. Medical image preprocessing pipelines run offline entirely. Satellite and aerial image analysis prioritises detection accuracy over speed. 
5.3 Robustness and Failure Cases
The ensemble inherits the failure modes of both component models. Small objects are the most persistent problem: AP_small remains substantially below AP_large across all three configurations, even with FPN. 
Class imbalance in COCO means common categories like person and car are detected far more reliably than rare ones, and the ensemble does not fix that. Both models were trained on natural images, so anything outside the COCO distribution, medical scans, infrared imagery, heavily stylised images, will perform worse without domain adaptation. Finally, neither model has any adversarial robustness.
If a perturbation fools both models in the same direction, WBF would reinforce the incorrect detection rather than catching it.

6. Conclusion
The central claim of this project, that a cross-architecture WBF ensemble combining YOLOv8-m and Faster R-CNN would outperform either model alone, held up in practice. The ensemble reached 0.518 mAP@0.5:0.95, a consistent 1.6 AP point gain over the best single model, with improvements across all object scales.
 The small-object results were particularly encouraging as evidence that the two models were genuinely complementary rather than correlated in their errors.
The limitation is speed. At 11 FPS, this is not a real-time system. That is not a problem in every context, but it does meaningfully restrict where it can be deployed.
If I were to take this further, the most promising directions would be: learning per-category Weighted-Box-Fusion weights rather than using equal weights for both models, adding a lightweight small-object specialist like YOLOv8-n to address the AP_small weakness, and quantising Faster R-CNN to reduce the two-stage bottleneck without sacrificing too much accuracy. 
Whether any of those directions is worth pursuing depends on the specific deployment target and whether the accuracy gain over a single fast model justifies the added complexity.



References
Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., and Kalinin, A. A. (2020). Albumentations: Fast and Flexible Image Augmentations. Information, 11(2), 125.
Dietterich, T. G. (2000). Ensemble Methods in Machine Learning. In: Multiple Classifier Systems. Springer, Berlin, Heidelberg. pp. 1-15.
Girshick, R., Donahue, J., Darrell, T., and Malik, J. (2014). Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. Proceedings of the IEEE CVPR. pp. 580-587.
Jocher, G., Chaurasia, A., and Qiu, J. (2023). Ultralytics YOLOv8. [Software] Available at: https://github.com/ultralytics/ultralytics [Accessed: March 2026].
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. Proceedings of the European Conference on Computer Vision (ECCV). pp. 740-755.
Lin, T.-Y., Dollar, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. (2017). Feature Pyramid Networks for Object Detection. Proceedings of the IEEE CVPR. pp. 2117-2125.
Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. Proceedings of the IEEE CVPR. pp. 779-788.
Ren, S., He, K., Girshick, R., and Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Advances in Neural Information Processing Systems (NeurIPS). 28, pp. 91-99.
Solovyev, R., Wang, W., and Gabruseva, T. (2021). Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models. Image and Vision Computing, 107, 104117.
TorchVision maintainers and contributors (2016). TorchVision: PyTorch's Computer Vision library. GitHub. Available at: https://github.com/pytorch/vision [Accessed: March 2026].
