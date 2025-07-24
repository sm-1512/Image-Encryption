import numpy as np
from PIL import Image
import hashlib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import math
import os

# ---------- Preprocessing ----------
def preprocess_image(path, size=(512, 512)):
    print("Loading and preprocessing image...")
    img = Image.open(path).convert("L").resize(size)
    return np.array(img, dtype=np.uint8)

# ---------- SHA256-Based Seeding ----------
def image_hash_seed(image):
    h = hashlib.sha256(image.tobytes()).hexdigest()
    return [int(h[i:i+8], 16)/0xFFFFFFFF for i in range(0, 64, 8)]

# ---------- Logistic Map (Permutation) ----------
def logistic_map(size, x0, r=3.99):
    seq = []
    x = x0
    for _ in range(size):
        x = r * x * (1 - x)
        seq.append(x)
    return np.array(seq)

# ---------- Duffing Map (Diffusion) ----------
def duffing_map(size, x0, y0, a=2.75, b=0.2):
    x, y = x0, y0
    seq = []
    for _ in range(size):
        x_new = y
        y_new = -b * x + a * y - y**3
        x, y = x_new, y_new
        seq.append((x + y) % 1)
    return np.array(seq)

# ---------- Permutation ----------
def permute_image(img, seq):
    print("Permuting image...")
    flat = img.flatten()
    indices = np.argsort(seq)
    return flat[indices], indices

def inverse_permute(flat_img, indices):
    inv_indices = np.zeros_like(indices)
    inv_indices[indices] = np.arange(len(indices))
    return flat_img[inv_indices]

# ---------- Diffusion (Duffing) ----------
def diffuse_image(flat_img, key_stream, iv):
    print("Diffusing image...")
    cipher = np.zeros_like(flat_img, dtype=np.uint8)
    prev = iv
    for i in range(len(flat_img)):
        val = (int(flat_img[i]) ^ int(key_stream[i]) ^ int(prev)) % 256
        cipher[i] = np.uint8(val)
        prev = cipher[i]
    return cipher

def reverse_diffusion(cipher, key_stream, iv):
    print("Reversing diffusion...")
    recovered = np.zeros_like(cipher, dtype=np.uint8)
    prev = iv
    for i in range(len(cipher)):
        val = (int(cipher[i]) ^ int(key_stream[i]) ^ int(prev)) % 256
        recovered[i] = np.uint8(val)
        prev = cipher[i]
    return recovered

# ---------- Image-Based Hash Diffusion ----------
def generate_image_based_keystream(image_bytes, length):
    key = hashlib.sha512(image_bytes).digest()
    keystream = bytearray()
    while len(keystream) < length:
        key = hashlib.sha256(key).digest()
        keystream.extend(key)
    return np.array(keystream[:length], dtype=np.uint8)

def password_diffusion(data, keystream):
    print("Applying image-based diffusion...")
    out = np.zeros_like(data, dtype=np.uint8)
    prev = 0
    for i in range(len(data)):
        out[i] = data[i] ^ keystream[i] ^ prev
        prev = out[i]
    return out

def reverse_password_diffusion(data, keystream):
    print("Reversing image-based diffusion...")
    out = np.zeros_like(data, dtype=np.uint8)
    prev = 0
    for i in range(len(data)):
        out[i] = data[i] ^ keystream[i] ^ prev
        prev = data[i]
    return out

# ---------- Metrics ----------
def calculate_entropy(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def calculate_npcr_uaci(img1, img2):
    diff = img1 != img2
    npcr = 100 * np.sum(diff) / diff.size
    uaci = 100 * np.sum(np.abs(img1.astype(np.int16) - img2.astype(np.int16))) / (img1.size * 255)
    return npcr, uaci

def correlation_coefficient(img):
    print("Calculating correlation coefficients...")
    img = img.astype(np.int32)
    x = img[:, :-1].flatten()
    y = img[:, 1:].flatten()
    rh = np.corrcoef(x, y)[0, 1]
    x = img[:-1, :].flatten()
    y = img[1:, :].flatten()
    rv = np.corrcoef(x, y)[0, 1]
    x = img[:-1, :-1].flatten()
    y = img[1:, 1:].flatten()
    rd = np.corrcoef(x, y)[0, 1]
    return rh, rv, rd

def plot_histograms(original, encrypted, decrypted):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.hist(original.flatten(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title("Original Image Histogram")
    plt.xlabel("Pixel Value"); plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(encrypted.flatten(), bins=256, range=(0, 256), color='red', alpha=0.7)
    plt.title("Encrypted Image Histogram")
    plt.xlabel("Pixel Value")

    plt.subplot(1, 3, 3)
    plt.hist(decrypted.flatten(), bins=256, range=(0, 256), color='green', alpha=0.7)
    plt.title("Decrypted Image Histogram")
    plt.xlabel("Pixel Value")

    plt.tight_layout()
    plt.show()

# ---------- Main Function ----------
def main(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    img = preprocess_image(image_path, size=(512, 512))
    rows, cols = img.shape
    flat_img = img.flatten()

    # Stage 1: Permutation
    seeds = image_hash_seed(img)
    logistic_seq = logistic_map(flat_img.size, seeds[0])
    permuted_flat, indices = permute_image(img, logistic_seq)

    # Stage 2: Duffing Diffusion
    key_stream = (duffing_map(flat_img.size, seeds[1], seeds[2]) * 255).astype(np.uint8)
    iv = np.uint8(int((seeds[3] * 1e6)) % 256)
    diffused_flat = diffuse_image(permuted_flat, key_stream, iv)

    # Stage 3: Image-Based Diffusion
    image_keystream = generate_image_based_keystream(img.tobytes(), len(diffused_flat))
    encrypted_flat = password_diffusion(diffused_flat, image_keystream)

    encrypted_img = encrypted_flat.reshape((rows, cols))
    Image.fromarray(encrypted_img).save("encrypted.png")

    # Decryption
    decrypted_flat_stage1 = reverse_password_diffusion(encrypted_flat, image_keystream)
    decrypted_flat_stage2 = reverse_diffusion(decrypted_flat_stage1, key_stream, iv)
    recovered_flat = inverse_permute(decrypted_flat_stage2, indices)
    decrypted_img = recovered_flat.reshape((rows, cols))
    Image.fromarray(decrypted_img).save("decrypted.png")

    # Metrics
    entropy = calculate_entropy(encrypted_img)
    print(f"\n=== Encryption Quality Metrics ===")
    print(f"Entropy: {entropy:.4f}")

    # Key Sensitivity Test (NPCR & UACI)
    img2 = img.copy()
    img2[0, 0] = np.uint8((int(img2[0, 0]) + 1) % 256)
    seeds2 = image_hash_seed(img2)
    logistic_seq2 = logistic_map(flat_img.size, seeds2[0])
    permuted_flat2, _ = permute_image(img2, logistic_seq2)
    key_stream2 = (duffing_map(flat_img.size, seeds2[1], seeds2[2]) * 255).astype(np.uint8)
    iv2 = np.uint8(int((seeds2[3] * 1e6)) % 256)
    diffused_flat2 = diffuse_image(permuted_flat2, key_stream2, iv2)
    image_keystream2 = generate_image_based_keystream(img2.tobytes(), len(diffused_flat2))
    encrypted_flat2 = password_diffusion(diffused_flat2, image_keystream2)
    encrypted_img2 = encrypted_flat2.reshape((rows, cols))

    npcr, uaci = calculate_npcr_uaci(encrypted_img, encrypted_img2)
    print(f"NPCR: {npcr:.4f}%")
    print(f"UACI: {uaci:.4f}%")

    rh, rv, rd = correlation_coefficient(encrypted_img)
    print("Correlation Coefficients (Encrypted Image):")
    print(f"  Horizontal: {rh:.4f}")
    print(f"  Vertical:   {rv:.4f}")
    print(f"  Diagonal:   {rd:.4f}")

    psnr_enc = psnr(img, encrypted_img)
    ssim_enc = ssim(img, encrypted_img, data_range=255)
    print(f"PSNR (Original vs Encrypted): {psnr_enc:.4f} dB")
    print(f"SSIM (Original vs Encrypted): {ssim_enc:.4f}")

    plot_histograms(img, encrypted_img, decrypted_img)

if __name__ == "__main__":
    main("pepper.jpg")
