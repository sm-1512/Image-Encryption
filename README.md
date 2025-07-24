# 🔐 Image Encryption Using Logistic & Duffing Maps with Hash-Based Diffusion

This project implements a robust grayscale image encryption and decryption scheme using:

* Logistic map-based permutation
* Duffing map-based diffusion
* SHA-based image-sensitive diffusion

The encryption ensures strong key sensitivity, randomness, and resistance against statistical and differential attacks.

## 📌 Features

- ✔️ **Three-stage encryption**:
  - 🔁 Permutation using the **Logistic Map**
  - 🔐 Diffusion using the **Duffing Map**
  - 🧬 Final hash-based diffusion using image content (**SHA-512 → SHA-256 keystream**)

- 📈 **Security Metrics Computed**:
  - 🔢 **Entropy**
  - 🔁 **NPCR (Number of Pixel Change Rate)**
  - 🎯 **UACI (Unified Average Changing Intensity)**
  - 📉 **Correlation Coefficients** (Horizontal, Vertical, Diagonal)
  - 📊 **PSNR & SSIM**

- 🧪 **Key Sensitivity Test** included (with 1-pixel input modification)

- 📊 **Histogram Analysis** for:
  - Original Image
  - Encrypted Image
  - Decrypted Image

## 📂 Project Structure

```text
Image-Encryption/
├── encryption.py          # Main encryption/decryption script
├── pepper.jpg             # Sample input grayscale image
├── encrypted.png          # Output: Encrypted image
├── decrypted.png          # Output: Decrypted image (matches original)
└── README.md              # Project documentation
```

## 🛠️ Requirements

Make sure you have **Python 3.x** installed.  
Then install the required dependencies using pip:

```bash
pip install numpy pillow matplotlib scikit-image
```

## ▶️ How to Run
Place your grayscale .jpg image (512x512 or any size) in the same directory and run:
```bash
python image_encryption.py
```
Make sure to update the main() function if you change the image name:
```python
main("your_image.jpg")
```
## 🔄 Encryption Workflow

The encryption process consists of the following three stages:

1. 🖼️ **Image Preprocessing**  
   - Convert the input image to grayscale  
   - Resize to 512×512 pixels (if needed)

2. 🔐 **Key Generation using SHA-256**  
   - Generate seeds by hashing the original image bytes  
   - Seeds are used to initialize chaotic maps (logistic and duffing)

3. 🔁 **Permutation using Logistic Map**  
   - Shuffle pixel positions based on a chaotic sequence

4. 🔒 **Diffusion using Duffing Map**  
   - Generate a pseudo-random key stream from chaotic Duffing dynamics  
   - XOR-based pixel diffusion with an initialization vector (IV)

5. 🔬 **Final Diffusion using SHA-Based Keystream**  
   - Generate a keystream from SHA-512 and SHA-256 hashes of the image  
   - Apply final XOR-based diffusion for enhanced randomness

📌 The result is a highly encrypted image with strong statistical and differential security.

## 🔓 Decryption Process

The decryption follows the reverse order of the encryption stages:

1. 🔄 **Reverse Hash-Based Diffusion**  
   Undo the final XOR diffusion using the same SHA-512/256-derived keystream.

2. 🔄 **Reverse Duffing Map Diffusion**  
   Use the same key stream and initial vector (IV) to decrypt pixel values.

3. 🔄 **Inverse Permutation**  
   Reconstruct the original pixel positions using the stored permutation indices.

✅ The decrypted image (`decrypted.png`) should be **visually and numerically identical** to the original input image.

## 💡 Notes

- 🖤 This scheme only supports **grayscale images**. Images are converted using `.convert("L")`.

- 🔐 **Image-dependent encryption**: Even a 1-pixel change in the input image generates an entirely different cipher image due to SHA-based seed generation.

- ⚙️ Uses **SHA-256** and **SHA-512** for deterministic randomness and keystream generation.

- 📷 Encrypted image has:
  - High **entropy**
  - Low **pixel correlation**
  - Strong resistance to statistical and differential attacks

- 🧪 Decryption is only possible if all parameters (maps, seeds, IV) are identical to those used during encryption.
