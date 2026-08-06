# Generative Adversarial Networks Assignment

This project is about the implementation and evaluation of generative adversarial networks (GANs).
In synthetic two-dimensional data, medical images, cybersecurity network
Hand-drawn sketches and traffic. This implementation is done in PyTorch and designed
To be compatible with Google Colab.

## Project files

Before creating new notebooks with the same names as the submitted one, rename it to the following clean names:
ZIP for the final submission:

```text
GAN_Assignment/
├── README.md
├── GAN_Assignment_Report.pdf
├── 01_part1_2d_gans.ipynb
├── 02_octmnist_dcgan.ipynb
├── 03_cicids_tabular_gan.ipynb
└── 04_quickdraw_dcgan.ipynb
```

| File | Purpose |
|---|---|
Trains a sine-wave GAN and compares baseline and modified GANs on an 8-component Gaussian mixture.
Run `02_octmnist_dcgan.ipynb` to explore OCTMNIST, train a retinal-image DCGAN, and plot losses and calculate FID.
Download CICIDS2017, train separate MLP GANs for BENIGN and DoS traffic and evaluate with PCA and distributional metrics.
| `04_quickdraw_dcgan.ipynb` | Download QuickDraw birthday-cake sketches, train a DCGAN and evaluate the images generated visually and using FID. |
Presents the methodology, figures, numerical results, interpretation, limitations and references.

The optional OCT conditional GAN, all-days CICIDS experiment and additional security features are provided.
QuickDraw categories are added as extensions which are disabled by default.
In the experiments reported the following description will be found.

## Main software requirements

Most of the packages required are already installed in Google Colab. The notebooks install
Specialised packages as needed.

```text
Python 3.10+
PyTorch
torchvision
NumPy
pandas
Matplotlib
Seaborn
SciPy
scikit-learn
tqdm
MedMNIST
TorchMetrics
torch-fidelity
KaggleHub
```

For training of OCTMNIST and QuickDraw, it is highly recommended to use an NVIDIA GPU.
for FID calculation.

You can run the project in Google Colab.The project can be run in Google Colab.

1. Go to <https://colab.research.google.com/>.
2. Click on File > Upload notebook and upload a notebook.
3. Click on Runtime > Change runtime type.
4. Select T4 GPU or any other GPU (if present) and save the configuration.
5. Select **Runtime > Run all**.
6. Do not remove from the final cell until the experiment is complete.
7. Download the notebook that is executed using File > Download > Download .ipynb.
8. Do the same for the other 4 notebooks.

Run the notebooks in numerical order. They are independent, but this order
follows the format of the report:

```text
01_part1_2d_gans.ipynb
02_octmnist_dcgan.ipynb
03_cicids_tabular_gan.ipynb
04_quickdraw_dcgan.ipynb
```

Please leave the browser tab open during training. FID calculation may be temporarily interrupted
while Inception weights are downloaded and can take considerably longer than
ordinary plotting cells.

## Dataset acquisition

There is no data required for submission with this project.

### OCTMNIST

The `medmnist` package is used to download the dataset in the OCTMNIST notebook.
automatically from <https://medmnist.com/>.

### CICIDS2017

The CICIDS notebook is for downloading the public network intrusion data from KaggleHub.
dataset automatically:

<https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset>

The required experiment selects the CSV of working hours and models for Wednesday.
BENIGN and DoS traffic. If asked by Kaggle to login, log in to Kaggle and
Set up an API token for Kaggle in the Colab session. Please do not put a private API in the public domain.
The provided notebook contains a token.

### QuickDraw

This is the `birthday cake` NumPy bitmap file and was downloaded from the QuickDraw notebook.
The data source used for this is from the Google Creative Lab dataset:

<https://github.com/googlecreativelab/quickdraw-dataset>

## Reported experiment settings

| Experiment | Training configuration |
|---|---|
| Part 1 | Seed 42, latent dimension 16, batch size 256, 5,000 training steps |
| OCTMNIST | Seed 42, latent dimension 100, batch size 256, 20 epochs, 5,000 FID samples |
| CICIDS2017 | Seed 42, latent dimension 128, batch size 512, 50 epochs, 100,000 rows per class |
| QuickDraw | Seed 42, latent dimension 100, batch size 256, 30 epochs, 50,000 training sketches, 5,000 FID samples |

To evaluate the results, leave `QUICK_RUN = False. Quick-run mode is only for
It's used to verify that the pipeline is running, but does not generate the reported values.

## Main recorded results

| Experiment | Result |
|---|---|
| Sine-wave GAN | MMD 0.0409; final generator loss 0.7586; discriminator loss 0.6432 |
Gaussian-mixture baseline | MMD 0.0003, generator loss 0.7240, discriminator loss 0.6698
Gaussian-mixture modified | MMD 0.0003; generator loss 0.7890; discriminator loss 0.6718 |
OCTMNIST DCGAN FID 99.761 with 5k real and generated images |
| CICIDS BENIGN GAN | Median KS 0.3656; correlation error 0.1967; real/fake AUC 1.000 |
| CICIDS DoS GAN | Median KS 0.3402; correlation error 0.3206; real/fake AUC 1.000 |
QuickDraw DCGAN | FID 19.397 (with 5000 real and generated images)

The lower MMD, FID, KS and correlation error are generally better. For the CICIDS
If synthetic data is not informative, then the AUC of the real-versus-fake classifier would be close to 0.5.
are not easy to distinguish from real data. The observed AUC of 1.000 thus
But it does not reflect good performance of GANs, but poor synthetic fidelity.

## Output directories

Every notebook has its own output folder:

```text
outputs/part1/
outputs/octmnist/
outputs/cicids/
outputs/quickdraw/birthday_cake/
```

These directories include figures, metric tables, configuration files and
model checkpoints. If they are available, they can be downloaded from the Colab file browser.
required, but not necessary when the notebooks executed already
Include any figures or numerical findings in the report.

## Troubleshooting

### DataLoader cleanup traceback

If Colab shows you the following message:

```text
AssertionError: can only test a child process
```

change the configuration in the OCTMNIST, CICIDS and QuickDraw notebooks to:

```python
NUM_WORKERS = 0
```

Then rerun the runtime and rerun all the cells. This avoids multiprocessing
cleanup issue on notebooks.

### CUDA out-of-memory error

Make `BATCH_SIZE` smaller and restart the runtime and rerun the notebook. Changing the
The final losses and evaluation metrics might differ depending on the batch size, therefore rerun the
complete experiment and update the report if different values are produced.

An FID dependency or download error occurred.An FID dependency or download error has occurred.

Re-run the installation cell, check that the Colab session is connected to the internet.
access, and restart the runtime if newly installed packages do not show up.

### Kaggle download error

Check the dataset URL is accessible, and log in with the Kaggle API.
token if requested. If manually changing the selected CSV, do not do so when performing
the optional all days extension.

## Reproducibility notes

- All experiments use random seed 42.
Each notebook has configuration values clustered together towards the start of the notebook.
Functions are used for training and evaluation.
Model checkpoints, figures and metrics are saved automatically.
CUDA determinism is turned on when supported.
However, there may be slight numerical difference due to differences in hardware and libraries.

## Final submission checklist

- [ ] PDF or Word report submitted according to the module instructions.
The student's name, student ID, module code and submission date is present on Report.
- [ ] Report meets the 6-8 page requirement.
- [ ] All notebooks have appropriate clean names without duplicate suffixes like '(2)' or '(3)'.
- [ ] Each notebook has been executed from cell 1 to last cell.
- [ ] Notebooks contain data figures and data output.
- [ ] There is no red traceback output in submitted notebooks.
All reported experiments have a value of `QUICK_RUN: False`.
- [ ] The CICIDS notebook is included.
- [ ] No private Kaggle token, dataset archive or unnecessary model weights are submitted.
