# Indian Sign Language (ISL) Recognition & English Converter

Real-time Indian Sign Language (ISL) fingerspelling recognition system powered by Computer Vision (MediaPipe Hands + Canny Edge CNN).

---

## Quick Start (Run the App)

### Option 1: One-Click Run (Windows)
Double-click **`run.bat`** (or run `.\run.bat` in your terminal).
- On first launch, it will automatically set up the virtual environment (`.venv`) and install all dependencies.
- It will then immediately open the desktop application with your live webcam.

### Option 2: Using Yarn / NPM
If you use Node/Yarn:
```bash
yarn run dev
# or
npm run dev
```

### Option 3: Manual Python Commands
```bash
# 1. Clone repository
git clone https://github.com/amrindermatti01-creator/ISL-to-English.git
cd ISL-to-English

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows (or source .venv/bin/activate on Linux/Mac)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Desktop UI
python src/ui_app.py
```

---

## App Features
- **Live Recognition**: Real-time sign classification across 36 classes (alphabets `A`–`Z` and numbers `0`–`9`).
- **MediaPipe Hand Tracking**: Dynamic single and dual-handed region-of-interest extraction with clutter elimination.
- **Fingerspelling Word Buffer**: Hold any sign steadily for ~1 second to accumulate letters into full words.
- **Desktop UI**: Interactive GUI with confidence meter, latency tracking, and snapshot capture.
