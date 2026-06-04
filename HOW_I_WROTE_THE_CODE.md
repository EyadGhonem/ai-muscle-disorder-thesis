# How I Wrote This Code

A summary of my coding approach and methodology for the Radiomics Feature Extraction Pipeline.

## 1. **Problem I'm Solving**

I needed to extract quantitative features from 3D medical images (MRI) to analyze disease characteristics. Rather than manually measuring images, I automated this using radiomics—a scientific field that extracts 100+ statistical features from imaging data.

## 2. **My Coding Approach**

### **Step 1: Understand the Problem**
- Learned about PyRadiomics library (a scientific tool for feature extraction)
- Understood medical image formats (NIfTI - a standard medical imaging format)
- Defined what features I needed to extract

### **Step 2: Set Up the Foundation**
- Created a Python 3.9 virtual environment (isolated workspace for the project)
- Installed required libraries:
  - **PyRadiomics** - the core feature extraction tool
  - **SimpleITK** - for reading/writing medical images
  - **Pandas** - for organizing results into tables
  - **NumPy** - for numerical computations

### **Step 3: Break Down the Problem into Modules**
Instead of writing one giant script, I split the work into separate, focused scripts:

| Script | Purpose |
|--------|---------|
| `extract_radiomics.py` | Main script - extracts features from single image pairs |
| `extract_radiomics_advanced.py` | Batch processing - runs on multiple images automatically |
| `convert_images_to_nifti.py` | Converts 2D images (JPG, PNG) into 3D medical format |
| `downsample_images.py` | Creates smaller test versions (1/3 resolution) for quick testing |
| `test_extraction.py` | Validates that extraction is working correctly |

### **Step 4: Implement With Testing in Mind**
- **Small data first**: Created downsampled versions of images for quick testing (30 seconds) before running full processing (15-30 minutes)
- **Modular design**: Each script does one job well, making it easier to test and fix
- **Error handling**: Added logging to track what the code is doing

### **Step 5: Organize Data Flow**

```
Input Data
    ↓
Input: 3D MRI Image + Binary Mask
    ↓
PyRadiomics Extractor
    ↓
Extract 107+ Features:
  - Shape (size, elongation, etc.)
  - Texture (pattern analysis)
  - First-order statistics (mean, median, etc.)
    ↓
Output: CSV File with Results
    ↓
Result: Quantified disease characteristics for analysis
```

## 3. **Key Design Decisions**

### **Why Virtual Environment?**
Keeps this project's dependencies separate from other Python projects on my computer. Prevents version conflicts.

### **Why Separate Scripts?**
- `extract_small.py` - Fast testing (30 seconds) before committing to 15+ minute runs
- `extract_full.py` - Production-level processing
- Different scripts let me control what runs when

### **Why Downsampling?**
Medical images are large (100+ MB). Downsampling to 1/3 resolution lets me:
- Test the full pipeline quickly
- Verify everything works before full processing
- Debug problems faster

### **Why This Structure?**
```
thesis_project/
├── data/                          ← Input location (images + masks)
├── output/                        ← Output location (results)
├── extract_*.py                   ← The actual code
├── radiomics_env/                 ← Isolated environment
└── Documentation files            ← Explains how to use it
```

## 4. **The Workflow (Step-by-Step)**

1. **Preparation**: Place MRI images and masks in `data/images/` and `data/masks/`
2. **Activation**: Run the virtual environment activation script
3. **Execution**: Python runs my script which:
   - Loads each image pair
   - Applies the binary mask (defines region of interest)
   - Runs 20+ radiomics algorithms on that region
   - Extracts 107+ numerical features
4. **Output**: Saves results as CSV file for analysis

## 5. **Code Quality Practices I Used**

✓ **Comments**: Explains what each section does
✓ **Logging**: Tracks progress and identifies errors
✓ **Modular**: Each function has one job
✓ **Testing**: Small test version before full run
✓ **Error Handling**: Graceful failures with clear messages
✓ **Documentation**: README and setup guides

## 6. **Why This Approach?**

This coding style follows best practices in scientific programming:
- **Reproducible**: Same inputs always produce same outputs
- **Maintainable**: Easy to modify or fix specific parts
- **Scalable**: Can easily add more images without rewriting code
- **Documented**: Future me (or collaborators) can understand the logic

## 7. **If Something Breaks**

I can debug because:
- Modular design = isolate problems to specific scripts
- Logging shows exactly where it failed
- Test scripts run first (fast validation)
- Clear data flow makes it easy to trace issues

---

**Bottom Line**: I wrote this code to be *fast to test, easy to maintain, and scientifically rigorous*—automating a complex analysis that would otherwise require many hours of manual work.
