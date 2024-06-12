# Application of Multi-Resolution Mutual Learning Network in Multi-Label ECG Classification

Abstract—Electrocardiograms (ECG), which record the electrophysiological activity of the heart, have become a crucial tool for diagnosing these diseases. In recent years, the application of deep learning techniques has significantly improved the performance of ECG signal classification. Multi-resolution feature analysis, which captures and processes information at different time scales, can extract subtle changes and overall trends in ECG signals, showing unique advantages. However, common multiresolution analysis methods based on simple feature addition or concatenation may lead to the neglect of low-resolution features, affecting model performance. To address this issue, this paper proposes the Multi-Resolution Mutual Learning Network (MRMNet). MRM-Net includes a dual-resolution attention architecture and a feature complementary mechanism. The dual-resolution attention architecture processes high-resolution and low-resolution features in parallel. Through the attention mechanism, the highresolution and low-resolution branches can focus on subtle waveform changes and overall rhythm patterns, enhancing the ability to capture critical features in ECG signals. Meanwhile, the feature complementary mechanism introduces mutual feature learning after each layer of the feature extractor. This allows features at different resolutions to reinforce each other, thereby reducing information loss and improving model performance and robustness. Experiments on the PTB-XL and CPSC2018 datasets demonstrate that MRM-Net significantly outperforms existing methods in multi-label ECG classification performance. The code for our framework will be publicly available at https://github.com/wxhdf/MRM.

